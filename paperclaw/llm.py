"""LLM client — Anthropic (official SDK) or any OpenAI-compatible endpoint."""

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator, Callable

import httpx

from paperclaw.config import LLMSettings


class LLMNotConfigured(Exception):
    pass


class LLMError(Exception):
    pass


# Tool names that write a workspace file — mirrors tools.WRITE_TOOLS, kept as a
# local constant so the tool-call loop can flag spec/file changes without llm.py
# importing the (heavier) tools package.
_WRITE_TOOL_NAMES = frozenset({"apply_patch", "write_file"})


@dataclass
class ChatResult:
    text: str
    model: str  # the model that actually served the reply (from the response)
    files_modified: frozenset[str] = frozenset()  # relative paths touched by a write tool


async def chat(
    settings: LLMSettings,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
) -> ChatResult:
    """Send a conversation and return the assistant reply + served model.

    `messages` is a list of {"role": "user"|"assistant", "content": str}.
    """
    if not settings.api_key:
        raise LLMNotConfigured("No API key configured — open Settings to add one.")

    if settings.provider == "anthropic":
        return await _chat_anthropic(settings, system, messages, max_tokens)
    return await _chat_openai(settings, system, messages, max_tokens)


async def stream_chat(
    settings: LLMSettings,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
) -> AsyncIterator[str]:
    """Yield text chunks from a streaming LLM reply.

    Usage::
        async for chunk in stream_chat(settings, system, messages):
            print(chunk, end="", flush=True)
    """
    if not settings.api_key:
        raise LLMNotConfigured("No API key configured — open Settings to add one.")
    if settings.provider == "anthropic":
        async for chunk in _stream_anthropic(settings, system, messages, max_tokens):
            yield chunk
    else:
        async for chunk in _stream_openai(settings, system, messages, max_tokens):
            yield chunk


async def _stream_anthropic(
    settings: LLMSettings, system: str, messages: list[dict[str, str]], max_tokens: int
) -> AsyncIterator[str]:
    from anthropic import APIError, AsyncAnthropic

    kwargs: dict = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = AsyncAnthropic(**kwargs)
    try:
        async with client.messages.stream(
            model=settings.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except APIError as exc:
        raise LLMError(f"Anthropic API error: {exc.message}") from exc
    finally:
        await client.close()


async def stream_chat_thinking(
    settings: LLMSettings,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int = 6000,
    thinking_budget: int = 2000,
) -> AsyncIterator[dict]:
    """Stream a reply as TYPED events: ``{"type": "thinking"|"text", "text": str}``.

    On Anthropic, enables extended thinking so the model's reasoning streams as
    ``thinking`` events ahead of the ``text`` answer. Other providers — and
    Anthropic models that don't support thinking — yield only ``text`` events
    (best-effort fallback). Raises LLMNotConfigured / LLMError on failure.
    """
    if not settings.api_key:
        raise LLMNotConfigured("No API key configured — open Settings to add one.")
    if settings.provider == "anthropic":
        async for ev in _stream_anthropic_thinking(
            settings, system, messages, max_tokens, thinking_budget
        ):
            yield ev
    else:
        async for ev in _stream_openai_thinking(settings, system, messages, max_tokens):
            yield ev


async def _stream_anthropic_thinking(
    settings: LLMSettings, system: str, messages: list[dict[str, str]],
    max_tokens: int, thinking_budget: int,
) -> AsyncIterator[dict]:
    from anthropic import APIError, AsyncAnthropic

    kwargs: dict = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = AsyncAnthropic(**kwargs)
    try:
        try:
            async with client.messages.stream(
                model=settings.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
            ) as stream:
                async for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        yield {"type": "thinking", "text": delta.thinking}
                    elif delta.type == "text_delta":
                        yield {"type": "text", "text": delta.text}
        except APIError as exc:
            # Some models reject extended thinking — fall back to plain streaming.
            # Thinking is rejected at request time, so nothing has been yielded yet.
            msg = str(getattr(exc, "message", exc)).lower()
            if "thinking" in msg or "budget" in msg:
                async with client.messages.stream(
                    model=settings.model, max_tokens=max_tokens,
                    system=system, messages=messages,
                ) as stream:
                    async for text in stream.text_stream:
                        yield {"type": "text", "text": text}
            else:
                raise LLMError(f"Anthropic API error: {exc.message}") from exc
    finally:
        await client.close()


async def _stream_openai(
    settings: LLMSettings, system: str, messages: list[dict[str, str]], max_tokens: int
) -> AsyncIterator[str]:
    base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": settings.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                raise LLMError(
                    f"LLM endpoint returned {resp.status_code}: {text.decode()[:200]}"
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    delta = json.loads(raw)["choices"][0]["delta"].get("content") or ""
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def _stream_openai_thinking(
    settings: LLMSettings, system: str, messages: list[dict[str, str]], max_tokens: int
) -> AsyncIterator[dict]:
    """Like :func:`_stream_openai` but surfaces reasoning tokens as ``thinking``
    events. Reasoning models on OpenAI-compatible endpoints stream their
    chain-of-thought in ``delta.reasoning_content`` (DeepSeek/vLLM style) or
    ``delta.reasoning`` (OpenRouter style) BEFORE the answer in ``delta.content``
    — surfacing it gives live feedback during the (often long) reasoning phase."""
    base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": settings.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                raise LLMError(f"LLM endpoint returned {resp.status_code}: {text.decode()[:200]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    delta = json.loads(raw)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    yield {"type": "thinking", "text": reasoning}
                content = delta.get("content")
                if content:
                    yield {"type": "text", "text": content}


async def _chat_anthropic(
    settings: LLMSettings, system: str, messages: list[dict[str, str]], max_tokens: int
) -> ChatResult:
    from anthropic import APIError, AsyncAnthropic

    kwargs: dict = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url

    client = AsyncAnthropic(**kwargs)
    try:
        response = await client.messages.create(
            model=settings.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
    except APIError as exc:
        raise LLMError(f"Anthropic API error: {exc.message}") from exc
    finally:
        await client.close()

    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    text = "".join(block.text for block in response.content if block.type == "text")
    return ChatResult(text=text, model=response.model)


async def chat_with_tools(
    settings: LLMSettings,
    system: str,
    messages: list[dict],
    tools: list[dict],
    executor: "Callable[[str, dict], str]",
    max_tokens: int = 4096,
    max_rounds: int = 8,
) -> ChatResult:
    """Agentic tool-use loop: call the LLM, execute any tool calls, repeat.

    ``executor(tool_name, inputs)`` is called for each tool_use block.
    Returns the final assistant text after all tools are resolved.

    ``tools`` must be a list of Anthropic tool schemas (with ``input_schema``).
    Raises ``LLMNotConfigured`` / ``LLMError`` on failure.
    Only supported for the Anthropic provider; falls back to plain chat for others.
    """
    if not settings.api_key:
        raise LLMNotConfigured("No API key configured — open Settings to add one.")

    if settings.provider != "anthropic":
        return await _chat_with_tools_openai(
            settings, system, messages, tools, executor, max_tokens, max_rounds
        )

    from anthropic import APIError, AsyncAnthropic

    kwargs: dict = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = AsyncAnthropic(**kwargs)

    conversation: list[dict] = list(messages)
    final_text = ""
    files_modified: set[str] = set()

    try:
        for _round in range(max_rounds):
            response = await client.messages.create(
                model=settings.model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=conversation,
            )

            # Collect text and tool_use blocks from the response
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(block)

            final_text = "".join(text_parts)

            if response.stop_reason != "tool_use" or not tool_calls:
                break  # done — no more tool calls

            # Add the assistant turn (with tool_use blocks) to the conversation
            conversation.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and build the tool_result turn
            results = []
            for call in tool_calls:
                try:
                    output = executor(call.name, dict(call.input))
                except Exception as exc:
                    output = f"Error: {exc}"
                # Track files written by apply_patch / write_file so callers know
                # the spec (or another workspace file) changed.
                if call.name in _WRITE_TOOL_NAMES and not str(output).startswith("Error"):
                    path = dict(call.input).get("path", "")
                    if path:
                        files_modified.add(path)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": output,
                })
            conversation.append({"role": "user", "content": results})

    except APIError as exc:
        raise LLMError(f"Anthropic API error: {exc.message}") from exc
    finally:
        await client.close()

    return ChatResult(text=final_text, model=settings.model, files_modified=frozenset(files_modified))


async def stream_chat_with_tools(
    settings: LLMSettings,
    system: str,
    messages: list[dict],
    tools: list[dict],
    executor: "Callable[[str, dict], str]",
    max_tokens: int = 4096,
    max_rounds: int = 8,
) -> AsyncIterator[dict]:
    """Streaming variant of :func:`chat_with_tools`.

    Yields typed events so the caller can stream text to the UI as it is
    generated instead of waiting for the whole tool loop to finish:

      ``{"type": "delta", "text": str}``    — assistant text, chunk by chunk
      ``{"type": "final", "text": str,      — terminal (exactly once): the
         "paths": list[str]}``                canonical final-round reply text
                                              and the files apply_patch wrote.

    The streamed deltas may include intermediate-round commentary (e.g. "let me
    read the file…"); ``final.text`` is the LAST round's text only, matching the
    non-streaming :func:`chat_with_tools` semantics for block parsing/persistence.

    Anthropic streams natively; OpenAI-compatible providers fall back to the
    non-streaming loop and emit the reply as a single delta.
    """
    if not settings.api_key:
        raise LLMNotConfigured("No API key configured — open Settings to add one.")

    if settings.provider != "anthropic":
        result = await _chat_with_tools_openai(
            settings, system, messages, tools, executor, max_tokens, max_rounds
        )
        if result.text:
            yield {"type": "delta", "text": result.text}
        yield {"type": "final", "text": result.text, "paths": list(result.files_modified)}
        return

    from anthropic import APIError, AsyncAnthropic

    kwargs: dict = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = AsyncAnthropic(**kwargs)

    conversation: list[dict] = list(messages)
    final_text = ""
    files_modified: set[str] = set()

    try:
        for _round in range(max_rounds):
            async with client.messages.stream(
                model=settings.model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=conversation,
            ) as stream:
                async for event in stream:
                    if (event.type == "content_block_delta"
                            and event.delta.type == "text_delta"):
                        yield {"type": "delta", "text": event.delta.text}
                final = await stream.get_final_message()

            final_text = "".join(b.text for b in final.content if b.type == "text")
            tool_calls = [b for b in final.content if b.type == "tool_use"]

            if final.stop_reason != "tool_use" or not tool_calls:
                break  # done — no more tool calls

            conversation.append({"role": "assistant", "content": final.content})
            results = []
            for call in tool_calls:
                try:
                    output = executor(call.name, dict(call.input))
                except Exception as exc:
                    output = f"Error: {exc}"
                if call.name == "apply_patch" and not str(output).startswith("Error"):
                    path = dict(call.input).get("path", "")
                    if path:
                        files_modified.add(path)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": output,
                })
            conversation.append({"role": "user", "content": results})
    except APIError as exc:
        raise LLMError(f"Anthropic API error: {exc.message}") from exc
    finally:
        await client.close()

    yield {"type": "final", "text": final_text, "paths": list(files_modified)}


RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_DELAYS = (1.0, 3.0)  # two retries after the first attempt


def _error_snippet(resp: httpx.Response) -> str:
    """Human-readable upstream error — never raw HTML in the chat."""
    text = resp.text.lstrip()
    if text.startswith(("<!DOCTYPE", "<!doctype", "<html", "<HTML")):
        return "(HTML error page from the provider/gateway — likely transient, try again)"
    try:
        data = resp.json()
        msg = data.get("error", {}).get("message") or data.get("message")
        if msg:
            return str(msg)[:300]
    except Exception:
        pass
    return text[:300]


def _flatten_tool_output(output) -> str:
    """Collapse a tool result to a string for the OpenAI-compatible path, which
    can't carry images in a tool message. A `read_image` result (list of content
    blocks) becomes its text label plus a note that the image can't be shown."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for blk in output:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text", "")))
            elif isinstance(blk, dict) and blk.get("type") == "image":
                parts.append("[image not viewable on this provider]")
        return "\n".join(parts) or "[non-text tool result]"
    return str(output)


def _anthropic_to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schemas (input_schema) to OpenAI function format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


async def _chat_with_tools_openai(
    settings: LLMSettings,
    system: str,
    messages: list[dict],
    tools: list[dict],
    executor: "Callable[[str, dict], str]",
    max_tokens: int,
    max_rounds: int,
) -> ChatResult:
    """OpenAI-compatible tool-calling loop (mirrors the Anthropic version)."""
    base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
    oai_tools = _anthropic_to_openai_tools(tools)
    conversation: list[dict] = [{"role": "system", "content": system}, *messages]
    final_text = ""
    files_modified: set[str] = set()

    async with httpx.AsyncClient(timeout=120.0) as client:
        for _round in range(max_rounds):
            payload = {
                "model": settings.model,
                "max_tokens": max_tokens,
                "messages": conversation,
                "tools": oai_tools,
                "tool_choice": "auto",
            }
            for delay in (*RETRY_DELAYS, None):
                try:
                    resp = await client.post(
                        f"{base}/chat/completions",
                        headers={"Authorization": f"Bearer {settings.api_key}"},
                        json=payload,
                    )
                except httpx.HTTPError as exc:
                    if delay is None:
                        raise LLMError(f"Cannot reach LLM endpoint: {exc}") from exc
                    await asyncio.sleep(delay)
                    continue
                if resp.status_code == 200:
                    break
                if resp.status_code in RETRY_STATUSES and delay is not None:
                    await asyncio.sleep(delay)
                    continue
                raise LLMError(
                    f"LLM endpoint returned {resp.status_code}: {_error_snippet(resp)}"
                )

            data = resp.json()
            try:
                choice = data["choices"][0]
                msg = choice["message"]
            except (KeyError, IndexError) as exc:
                raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from exc

            final_text = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            if choice.get("finish_reason") != "tool_calls" or not tool_calls:
                break  # no more tool calls — done

            # Add the assistant turn (with tool_calls) to the conversation
            conversation.append({"role": "assistant", "content": final_text, "tool_calls": tool_calls})

            # Execute each tool call and inject results
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                try:
                    inputs = json.loads(fn.get("arguments") or "{}")
                    output = executor(name, inputs)
                except Exception as exc:
                    output = f"Error: {exc}"
                if name in _WRITE_TOOL_NAMES and not str(output).startswith("Error"):
                    path = json.loads(fn.get("arguments") or "{}").get("path", "")
                    if path:
                        files_modified.add(path)
                conversation.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _flatten_tool_output(output),
                })

    return ChatResult(
        text=final_text,
        model=str(data.get("model") or settings.model),
        files_modified=frozenset(files_modified),
    )


async def _chat_openai(
    settings: LLMSettings, system: str, messages: list[dict[str, str]], max_tokens: int
) -> ChatResult:
    base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": settings.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, *messages],
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        for delay in (*RETRY_DELAYS, None):
            try:
                resp = await client.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.api_key}"},
                    json=payload,
                )
            except httpx.HTTPError as exc:
                if delay is None:
                    raise LLMError(f"Cannot reach LLM endpoint: {exc}") from exc
                await asyncio.sleep(delay)
                continue
            if resp.status_code == 200:
                break
            if resp.status_code in RETRY_STATUSES and delay is not None:
                await asyncio.sleep(delay)
                continue
            raise LLMError(
                f"LLM endpoint returned {resp.status_code}: {_error_snippet(resp)}"
            )

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from exc
    return ChatResult(text=text, model=str(data.get("model") or settings.model))
