"""Token estimates for pre-charging the tokens-per-minute bucket.

The real count is known only after the response, but admission has to happen before. We
therefore charge ``prompt estimate + max output`` up front and reconcile with the provider's
reported usage afterwards (ADR-012). The prompt estimate is a cheap ~4-chars-per-token
heuristic: running a real tokenizer per request would cost more CPU than the rest of the hot
path, the right tokenizer depends on which provider eventually serves the request, and the
error is corrected at reconciliation anyway.
"""

from __future__ import annotations

from switchyard.schemas import ChatCompletionRequest

CHARS_PER_TOKEN = 4
PER_MESSAGE_OVERHEAD = 4


def estimate_tokens_from_chars(chars: int) -> int:
    return -(-chars // CHARS_PER_TOKEN)  # ceil division


def estimate_text_tokens(text: str) -> int:
    return estimate_tokens_from_chars(len(text))


def estimate_prompt_tokens(request: ChatCompletionRequest) -> int:
    total = 0
    for message in request.messages:
        total += PER_MESSAGE_OVERHEAD
        if isinstance(message.content, str):
            total += estimate_text_tokens(message.content)
        elif isinstance(message.content, list):
            for part in message.content:
                if isinstance(part.get("text"), str):
                    total += estimate_text_tokens(part["text"])
    return max(total, 1)


def estimate_request_tokens(request: ChatCompletionRequest, default_max_tokens: int) -> int:
    max_output = request.max_completion_tokens or request.max_tokens or default_max_tokens
    return estimate_prompt_tokens(request) + max_output * (request.n or 1)
