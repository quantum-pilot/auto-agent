import json
import time
from collections.abc import Generator
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple, List, TypeVar, cast

from pydantic import BaseModel

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import (
    LLMModelConfig,
    LLMResult,
    LLMResultChunk,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    PromptMessageContentType,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage, ToolProviderType
from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentStrategy,
    ToolEntity,
    ToolInvokeMeta,
)

T = TypeVar("T")


class LogMetadata:
    """Metadata keys for logging"""
    STARTED_AT = "started_at"
    PROVIDER = "provider"
    FINISHED_AT = "finished_at"
    ELAPSED_TIME = "elapsed_time"
    TOTAL_PRICE = "total_price"
    CURRENCY = "currency"
    TOTAL_TOKENS = "total_tokens"


class ExecutionMetadata(BaseModel):
    """Execution metadata with default values"""
    total_price: float = 0.0
    currency: str = ""
    total_tokens: int = 0
    prompt_tokens: int = 0
    prompt_unit_price: float = 0.0
    prompt_price_unit: float = 0.0
    prompt_price: float = 0.0
    completion_tokens: int = 0
    completion_unit_price: float = 0.0
    completion_price_unit: float = 0.0
    completion_price: float = 0.0
    latency: float = 0.0

    @classmethod
    def from_llm_usage(cls, usage: Optional[LLMUsage]) -> "ExecutionMetadata":
        if usage is None:
            return cls()
        return cls(
            total_price=float(usage.total_price),
            currency=usage.currency,
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            prompt_unit_price=float(usage.prompt_unit_price),
            prompt_price_unit=float(usage.prompt_price_unit),
            prompt_price=float(usage.prompt_price),
            completion_tokens=usage.completion_tokens,
            completion_unit_price=float(usage.completion_unit_price),
            completion_price_unit=float(usage.completion_price_unit),
            completion_price=float(usage.completion_price),
            latency=usage.latency,
        )


PROVIDER_MODEL = {
    "GPT-mini":         ("openai",    "gpt-5-mini-2025-08-07"),
    "GPT":              ("openai",    "gpt-5-2025-08-07"),
    "Claude-Sonnet":    ("anthropic", "claude-sonnet-4-20250514"),
    "Opus":             ("anthropic", "claude-opus-4-1-20250805"),
}


class IncomingParams(BaseModel):
    user_model: str
    think: bool = False
    # agent scaffolding
    model: AgentModelConfig
    tools: List[ToolEntity] | None = None
    instruction: str
    query: str
    maximum_iterations: int = 15

    def to_params(self) -> "Params":
        if self.user_model.split(":")[-1] == "1":
            self.think = True
        self.user_model = self.user_model.split(":")[0]
        provider, model = PROVIDER_MODEL.get(self.user_model, PROVIDER_MODEL["GPT-mini"])
        tool_max_tokens   = 256
        final_max_tokens  = 2048
        thinking_budget = 0
        reasoning_effort = "minimal"
        verbosity        = "medium"
        if self.think:
            thinking_budget = 1024
            final_max_tokens = 4096
            reasoning_effort = "medium"
        itpm_limit = 2_000_000 if "-mini" in model else 450_000
        otpm_limit = 90_000 if provider == "anthropic" else None
        return Params(
            user_model=self.user_model,
            think=self.think,
            model=self.model,
            tools=self.tools,
            instruction=self.instruction,
            query=self.query,
            maximum_iterations=self.maximum_iterations,
            provider=provider,
            exact_model=model,
            thinking_budget=thinking_budget,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            tool_max_tokens=tool_max_tokens,
            final_max_tokens=final_max_tokens,
            itpm_limit=itpm_limit,
            otpm_limit=otpm_limit,
            memory_summary_max_tokens=256,
            summarize_history_when_chars_over=10_000,
            initial_delay_secs=5,
            delay_multiplier=2,
            retry_attempts=5,
        )


class Params(IncomingParams):
    provider: str
    exact_model: str
    thinking_budget: int
    reasoning_effort: str
    verbosity: str
    tool_max_tokens: int
    final_max_tokens: int
    itpm_limit: int
    otpm_limit: Optional[int]
    memory_summary_max_tokens: int
    summarize_history_when_chars_over: int
    initial_delay_secs: int
    delay_multiplier: int
    retry_attempts: int


class TokenBuckets:
    """
    Per-(provider:model) token buckets.
    - OpenAI: single TPM bucket; reserve max(input_estimate, max_tokens).
    - Anthropic: separate ITPM/OTPM; OTPM includes thinking budget.
    """
    def __init__(self, storage, key_prefix: str, itpm: int, otpm: Optional[int]):
        self.s = storage
        self.key_i = f"{key_prefix}:in"
        self.key_o = f"{key_prefix}:out" if otpm is not None else None
        self.capacity_i = float(itpm)
        self.capacity_o = float(otpm) if otpm is not None else None

    def _now(self) -> float:
        return time.time()

    def _load(self, key: str, capacity: float) -> Tuple[float, float]:
        try:
            raw = self.s.get(key)
            if raw:
                d = json.loads(raw.decode("utf-8"))
                return float(d["tokens"]), float(d["ts"])
        except Exception:
            pass
        return capacity, self._now()

    def _save(self, key: str, tokens: float, ts: float):
        self.s.set(key, json.dumps({"tokens": tokens, "ts": ts}).encode("utf-8"))

    def _refill(self, tokens: float, last_ts: float, capacity: float) -> Tuple[float, float]:
        now = self._now()
        rate = capacity / 60.0
        dt = max(0.0, now - last_ts)
        return min(capacity, tokens + dt * rate), now

    def _reserve(self, key: str, capacity: float, need: float, hard_wait: bool = True, timeout_s: float = 20.0) -> bool:
        tokens, ts = self._load(key, capacity)
        deadline = self._now() + timeout_s
        while tokens < need:
            tokens, ts = self._refill(tokens, ts, capacity)
            if tokens >= need:
                break
            if not hard_wait and self._now() >= deadline:
                return False
            shortfall = need - tokens
            wait_s = shortfall / (capacity / 60.0)
            time.sleep(min(wait_s, 0.5))
        tokens, ts = self._refill(tokens, ts, capacity)
        tokens -= min(tokens, need)
        self._save(key, tokens, ts)
        return True

    def reserve(self, need_in: float, need_out: Optional[float]) -> bool:
        ok_in = self._reserve(self.key_i, self.capacity_i, max(0.0, need_in), hard_wait=True)
        ok_out = True
        if self.key_o and need_out is not None:
            ok_out = self._reserve(self.key_o, self.capacity_o, max(0.0, need_out), hard_wait=True)
        return ok_in and ok_out


class DynamicRouterAgentStrategy(AgentStrategy):
    """
    History-aware, rate-limited, config-driven strategy.
    - Summarizes history when oversized
    - Reserves tokens using a per-(provider:model) token bucket
    - Adjusts max_tokens per iteration (tool vs final)
    - Supports "thinking" budgets and OpenAI verbosity/reasoning knobs
    """

    def _rough_chars(self, msgs: List[PromptMessage]) -> int:
        total = 0
        for m in msgs:
            c = getattr(m, "content", "")
            if isinstance(c, list):
                total += sum(len(getattr(x, "text", "") or "") for x in c)
            else:
                total += len(str(c))
        return total

    def _rough_tokens(self, msgs: List[PromptMessage], tools: List[PromptMessageTool]) -> int:
        chars = self._rough_chars(msgs)
        tools_chars = sum(len(t.model_dump_json()) for t in tools)
        return max(1, (chars + tools_chars) // 3)

    def _summarize_history(self, history: List[PromptMessage], provider: str, model: str, p: Params) -> Optional[str]:
        try:
            cfg = LLMModelConfig(
                provider=provider,
                model=model,
                mode="chat",
                completion_params={
                    "max_tokens": p.memory_summary_max_tokens,
                    "verbosity": "low",
                    "reasoning_effort": "minimal",
                },
            )
            prompt = [
                SystemPromptMessage(content="Summarize prior conversation into crisp facts, preferences, and task state. No fluff."),
                *history,
                UserPromptMessage(content="Return a concise bullet summary. Keep identifiers and numbers."),
            ]
            res = self.session.model.llm.invoke(model_config=cfg, prompt_messages=prompt, tools=[], stream=False)
            if isinstance(res, LLMResult) and res.message and res.message.content:
                return "".join(x.data for x in res.message.content if x.type == PromptMessageContentType.TEXT) if isinstance(res.message.content, list) else str(res.message.content)
        except Exception:
            return None
        return None

    def _clear_user_prompt_image_messages(self, prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        prompt_messages = deepcopy(prompt_messages)
        for prompt_message in prompt_messages:
            if isinstance(prompt_message, UserPromptMessage) and isinstance(prompt_message.content, list):
                prompt_message.content = "\n".join([
                    content.data if content.type == PromptMessageContentType.TEXT
                    else "[image]" if content.type == PromptMessageContentType.IMAGE
                    else "[file]"
                    for content in prompt_message.content
                ])
        return prompt_messages

    def _organize_prompt_messages(self, current_thoughts: list[PromptMessage], history_prompt_messages: list[PromptMessage]) -> list[PromptMessage]:
        prompt_messages = [*history_prompt_messages, *current_thoughts]
        if len(current_thoughts) != 0:
            prompt_messages = self._clear_user_prompt_image_messages(prompt_messages)
        return prompt_messages

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage, None, None]:
        p = IncomingParams(**parameters).to_params()
        provider, model = p.provider, p.exact_model

        # Build a storage for token buckets
        storage = self.session.storage
        bucket_key = f"rl:{provider}:{model}"
        buckets = TokenBuckets(storage=storage, key_prefix=bucket_key, itpm=p.itpm_limit, otpm=p.otpm_limit)

        # Prepare history (optional summarization for budget)
        history = p.model.history_prompt_messages or []
        history.insert(0, SystemPromptMessage(content=p.instruction))
        history.append(UserPromptMessage(content=p.query))
        if self._rough_chars(history) > p.summarize_history_when_chars_over:
            yield self.create_log_message(label="Summarizing history", data={"chars": self._rough_chars(history)}, status=ToolInvokeMessage.LogMessage.LogStatus.START)
            summary = self._summarize_history(history, provider, model, p)
            if summary:
                history = [SystemPromptMessage(content=f"Conversation summary:\n{summary}")]
            yield self.finish_log_message(log=self.create_log_message(label="History summarized", data={}), data={}, metadata={})

        # convert tool messages
        tools = p.tools
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}
        prompt_messages_tools = self._init_prompt_tools(tools)

        # stream = (ModelFeature.STREAM_TOOL_CALL in p.model.entity.features) if (p.model.entity and p.model.entity.features) else False
        stream = True  # force stream cuz both providers support it
        stop = (p.model.completion_params.get("stop", []) if p.model.completion_params else [])

        # function-calling loop
        iteration_step = 1
        max_iteration_steps = p.maximum_iterations
        current_thoughts: list[PromptMessage] = []
        function_call_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""

        # Retry/backoff helpers
        def _with_retries(fn: Callable[[], T], *, label: str) -> Generator[AgentInvokeMessage, None, T]:
            delay = max(0.0, float(p.initial_delay_secs))
            for attempt in range(1, int(p.retry_attempts) + 1):
                try:
                    return fn()
                except Exception as e:  # include 429s
                    emsg = str(e)
                    should_retry = any(tok in emsg.lower() for tok in ["rate", "429", "529", "quota", "overloaded", "throttl"]) or attempt < p.retry_attempts
                    yield self.create_log_message(label=f"{label} retry {attempt}", data={"error": emsg}, status=ToolInvokeMessage.LogMessage.LogStatus.START)
                    if not should_retry or attempt == p.retry_attempts:
                        raise
                    time.sleep(delay)
                    delay = delay * max(1, int(p.delay_multiplier))

        while function_call_state and iteration_step <= max_iteration_steps:
            # start a new round
            function_call_state = False
            round_started_at = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={LogMetadata.STARTED_AT: round_started_at},
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log

            # Decide token caps for this iteration
            is_last_round = iteration_step == max_iteration_steps
            cap_out_base = p.final_max_tokens if is_last_round else p.tool_max_tokens

            # Build prompt
            prompt_messages = self._organize_prompt_messages(history_prompt_messages=history, current_thoughts=current_thoughts)

            # Compose completion params for this round
            completion_params = dict(p.model.completion_params or {})
            # Thinking and max_tokens interaction:
            # If "think" is enabled and provider is Anthropic, we treat max_tokens as thinking_budget + output_allowance
            effective_max_tokens = cap_out_base
            if p.think and provider == "anthropic":
                effective_max_tokens = max(cap_out_base + int(p.thinking_budget), cap_out_base)
                completion_params["thinking"] = {"enabled": True, "budget_tokens": int(p.thinking_budget)}
            elif p.think:
                # OpenAI GPT-5 style knobs (no separate thinking budget parameter)
                completion_params["reasoning_effort"] = p.reasoning_effort
            # Always set verbosity for OpenAI-like models
            completion_params["verbosity"] = p.verbosity
            completion_params["max_tokens"] = int(effective_max_tokens)

            # Estimate input and reserve tokens
            input_estimate = self._rough_tokens(prompt_messages, prompt_messages_tools)

            if p.otpm_limit:  # Anthropic-style: separate out-bucket
                need_in = float(max(1, input_estimate))
                need_out = float(max(1, effective_max_tokens))
            else:  # OpenAI-style: reserve the greater of in/out in single TPM
                need_in = float(max(max(1, input_estimate), max(1, effective_max_tokens)))
                need_out = None

            yield self.create_log_message(
                label="Rate-limit reservation",
                data={"need_in": need_in, "need_out": need_out, "provider": provider, "model": model},
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            buckets.reserve(need_in=need_in, need_out=need_out)

            # Invoke LLM (with retries)
            model_started_at = time.perf_counter()
            model_config = LLMModelConfig(
                provider=provider,
                model=model,
                mode="chat",
                completion_params=completion_params,
            )
            model_log = self.create_log_message(
                label=f"{provider}:{model} thought",
                data={},
                metadata={LogMetadata.STARTED_AT: model_started_at, LogMetadata.PROVIDER: provider},
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log

            def _invoke_once():
                try:
                    kwargs = {
                        "model_config": model_config,
                        "prompt_messages": prompt_messages,
                        "stop": stop,
                        "stream": stream,
                        "tools": prompt_messages_tools if not (is_last_round and max_iteration_steps > 1) else [],
                    }
                    return self.session.model.llm.invoke(**kwargs)
                except Exception as e:
                    if "context length" in str(e).lower() or "max context" in str(e).lower():
                        # shrink and retry once
                        summary = self._summarize_history(history, provider, model, p)
                        if summary:
                            history[:] = [SystemPromptMessage(content=f"Conversation summary:\n{summary}")]
                            kwargs["prompt_messages"] = self._organize_prompt_messages(history_prompt_messages=history, current_thoughts=current_thoughts)
                            return self.session.model.llm.invoke(**kwargs)
                    raise

            chunks_or_result = yield from _with_retries(_invoke_once, label="LLM invoke")

            tool_calls: list[tuple[str, str, dict[str, Any]]] = []
            response = ""
            tool_call_names = ""
            current_llm_usage: Optional[LLMUsage] = None

            if isinstance(chunks_or_result, Generator):
                for chunk in chunks_or_result:
                    if self._check_tool_calls(chunk):
                        function_call_state = True
                        tool_calls.extend(self._extract_tool_calls(chunk) or [])
                        tool_call_names = ";".join([tc[1] for tc in tool_calls])
                    if chunk.delta.message and chunk.delta.message.content:
                        if isinstance(chunk.delta.message.content, list):
                            for content in chunk.delta.message.content:
                                response += content.data
                                if (not function_call_state) or is_last_round:
                                    yield self.create_text_message(content.data)
                        else:
                            response += str(chunk.delta.message.content)
                            if (not function_call_state) or is_last_round:
                                yield self.create_text_message(str(chunk.delta.message.content))
                    if chunk.delta.usage:
                        self._increase_usage(llm_usage, chunk.delta.usage)
                        current_llm_usage = chunk.delta.usage
            else:
                result = chunks_or_result
                if self._check_blocking_tool_calls(result):
                    function_call_state = True
                    tool_calls.extend(self._extract_blocking_tool_calls(result) or [])
                    tool_call_names = ";".join([tc[1] for tc in tool_calls])
                if result.usage:
                    self._increase_usage(llm_usage, result.usage)
                    current_llm_usage = result.usage
                if result.message and result.message.content:
                    if isinstance(result.message.content, list):
                        for content in result.message.content:
                            response += content.data
                    else:
                        response += str(result.message.content)
                if not result.message.content:
                    result.message.content = ""
                if isinstance(result.message.content, str):
                    yield self.create_text_message(result.message.content)
                elif isinstance(result.message.content, list):
                    for content in result.message.content:
                        yield self.create_text_message(content.data)

            # Finish model log
            yield self.finish_log_message(
                log=model_log,
                data={
                    "output": response,
                    "tool_name": tool_call_names,
                    "tool_input": [{"name": n, "args": a} for _, n, a in tool_calls],
                },
                metadata={
                    LogMetadata.PROVIDER: provider,
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - model_started_at,
                    LogMetadata.TOTAL_PRICE: current_llm_usage.total_price if current_llm_usage else 0,
                    LogMetadata.CURRENCY: current_llm_usage.currency if current_llm_usage else "",
                    LogMetadata.TOTAL_TOKENS: current_llm_usage.total_tokens if current_llm_usage else 0,
                },
            )

            if response.strip():
                current_thoughts.append(AssistantPromptMessage(content=response, tool_calls=[]))
            final_answer += response + "\n"

            # Tool call execution
            tool_responses = []
            for tool_call_id, tool_call_name, tool_call_args in tool_calls:
                current_thoughts.append(
                    AssistantPromptMessage(
                        content="",
                        tool_calls=[
                            AssistantPromptMessage.ToolCall(
                                id=tool_call_id,
                                type="function",
                                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                    name=tool_call_name,
                                    arguments=json.dumps(tool_call_args, ensure_ascii=False),
                                ),
                            )
                        ],
                    )
                )
                tool_instance = tool_instances.get(tool_call_name)
                tool_call_started_at = time.perf_counter()
                tool_call_log = self.create_log_message(
                    label=f"CALL {tool_call_name}",
                    data={},
                    metadata={LogMetadata.STARTED_AT: time.perf_counter(), LogMetadata.PROVIDER: (tool_instance.identity.provider if tool_instance else "")},
                    parent=round_log,
                    status=ToolInvokeMessage.LogMessage.LogStatus.START,
                )
                yield tool_call_log

                if not tool_instance:
                    tool_response = {
                        "tool_call_id": tool_call_id,
                        "tool_call_name": tool_call_name,
                        "tool_response": f"there is not a tool named {tool_call_name}",
                        "meta": ToolInvokeMeta.error_instance(f"there is not a tool named {tool_call_name}").to_dict(),
                    }
                else:
                    try:
                        tool_invoke_responses = self.session.tool.invoke(
                            provider_type=ToolProviderType(tool_instance.provider_type),
                            provider=tool_instance.identity.provider,
                            tool_name=tool_instance.identity.name,
                            parameters={**tool_instance.runtime_parameters, **tool_call_args},
                        )
                        tool_result = ""
                        for tool_invoke_response in tool_invoke_responses:
                            if tool_invoke_response.type == ToolInvokeMessage.MessageType.TEXT:
                                tool_result += cast(ToolInvokeMessage.TextMessage, tool_invoke_response.message).text
                            elif tool_invoke_response.type == ToolInvokeMessage.MessageType.LINK:
                                tool_result += (
                                    "result link: "
                                    + cast(ToolInvokeMessage.TextMessage, tool_invoke_response.message).text
                                    + ". please tell user to check it."
                                )
                            elif tool_invoke_response.type in {ToolInvokeMessage.MessageType.IMAGE_LINK, ToolInvokeMessage.MessageType.IMAGE}:
                                # Attempt to stream blob if local
                                if hasattr(tool_invoke_response.message, "text"):
                                    file_info = cast(ToolInvokeMessage.TextMessage, tool_invoke_response.message).text
                                    try:
                                        if file_info.startswith("/files/"):
                                            import os
                                            if os.path.exists(file_info):
                                                with open(file_info, "rb") as f:
                                                    file_content = f.read()
                                                blob_response = self.create_blob_message(blob=file_content, meta={"mime_type": "image/png", "filename": os.path.basename(file_info)})
                                                yield blob_response
                                    except Exception as e:
                                        yield self.create_text_message(f"Failed to create blob message: {e}")
                                tool_result += "image has been created and sent to user already, you do not need to create it, just tell the user to check it now."
                                yield tool_invoke_response
                            elif tool_invoke_response.type == ToolInvokeMessage.MessageType.JSON:
                                text = json.dumps(cast(ToolInvokeMessage.JsonMessage, tool_invoke_response.message).json_object, ensure_ascii=False)
                                tool_result += f"tool response: {text}."
                            elif tool_invoke_response.type == ToolInvokeMessage.MessageType.BLOB:
                                tool_result += "Generated file ... "
                                yield tool_invoke_response
                            else:
                                tool_result += f"tool response: {tool_invoke_response.message!r}."
                    except Exception as e:
                        tool_result = f"tool invoke error: {e!s}"
                    tool_response = {
                        "tool_call_id": tool_call_id,
                        "tool_call_name": tool_call_name,
                        "tool_call_input": {**(tool_instance.runtime_parameters if tool_instance else {}), **tool_call_args},
                        "tool_response": tool_result,
                    }

                yield self.finish_log_message(
                    log=tool_call_log,
                    data={"output": tool_response},
                    metadata={
                        LogMetadata.STARTED_AT: tool_call_started_at,
                        LogMetadata.PROVIDER: (tool_instance.identity.provider if tool_instance else ""),
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter() - tool_call_started_at,
                    },
                )
                tool_responses.append(tool_response)
                if tool_response["tool_response"] is not None:
                    current_thoughts.append(ToolPromptMessage(content=str(tool_response["tool_response"]), tool_call_id=tool_call_id, name=tool_call_name))

            if tool_calls:
                yield self.create_text_message("\n")

            for prompt_tool in prompt_messages_tools:
                if prompt_tool.name in tool_instances:
                    self.update_prompt_message_tool(tool_instances[prompt_tool.name], prompt_tool)

            yield self.finish_log_message(
                log=round_log,
                data={"output": {"llm_response": response, "tool_responses": tool_responses}},
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_PRICE: (llm_usage["usage"].total_price if llm_usage.get("usage") else 0),
                    LogMetadata.CURRENCY: (llm_usage["usage"].currency if llm_usage.get("usage") else ""),
                    LogMetadata.TOTAL_TOKENS: (llm_usage["usage"].total_tokens if llm_usage.get("usage") else 0),
                },
            )

            if tool_responses and max_iteration_steps == 1:
                for resp in tool_responses:
                    yield self.create_text_message(str(resp["tool_response"]))

            iteration_step += 1

        metadata = ExecutionMetadata.from_llm_usage(llm_usage.get("usage"))
        yield self.create_json_message({"execution_metadata": metadata.model_dump()})

    def _check_tool_calls(self, llm_result_chunk: LLMResultChunk) -> bool:
        return bool(llm_result_chunk.delta.message.tool_calls)

    def _check_blocking_tool_calls(self, llm_result: LLMResult) -> bool:
        return bool(llm_result.message.tool_calls)

    def _extract_tool_calls(self, llm_result_chunk: LLMResultChunk) -> list[tuple[str, str, dict[str, Any]]]:
        tool_calls = []
        for prompt_message in llm_result_chunk.delta.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)
            tool_calls.append((prompt_message.id, prompt_message.function.name, args))
        return tool_calls

    def _extract_blocking_tool_calls(self, llm_result: LLMResult) -> list[tuple[str, str, dict[str, Any]]]:
        tool_calls = []
        for prompt_message in llm_result.message.tool_calls:
            args = {}
            if prompt_message.function.arguments != "":
                args = json.loads(prompt_message.function.arguments)
            tool_calls.append((prompt_message.id, prompt_message.function.name, args))
        return tool_calls

    # Usage aggregation helper (mirror FunctionCallingAgentStrategy)
    def _increase_usage(self, usage_holder: dict[str, Optional[LLMUsage]], delta: LLMUsage):
        if usage_holder.get("usage") is None:
            usage_holder["usage"] = delta
        else:
            u = usage_holder["usage"]
            u.total_tokens += delta.total_tokens
            u.prompt_tokens += delta.prompt_tokens
            u.completion_tokens += delta.completion_tokens
            u.total_price += delta.total_price
            u.prompt_price += delta.prompt_price
            u.completion_price += delta.completion_price
            u.latency = max(u.latency, delta.latency)
