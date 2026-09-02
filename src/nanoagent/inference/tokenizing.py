"""Wrap a text-only transport so its replies still carry token ids — reconstructed, and labelled so.

This is what makes "every reply has tokens" true of OpenRouter and of any other chat API. It
renders the prompt and re-encodes the answer with the configured tokenizer, and stamps the result
:attr:`~nanoagent.inference.types.Fidelity.RECONSTRUCTED` so nothing downstream mistakes it for
what the sampler saw. Applied by :func:`~nanoagent.inference.backends.build_backend` when a config
names a ``tokenizer`` and the transport is not already native, so it covers plugin transports too
without any of them knowing it exists.

**What these ids are not.** Three separate reasons a reconstructed record can disagree with the
real one, all of them silent:

  * a provider that routes one model slug across several backends may not have used this
    vocabulary at all;
  * ``encode(decode(ids))`` is not the identity for every tokenizer, so even the right vocabulary
    can round-trip to a different segmentation;
  * ``completion_ids`` covers :attr:`Response.text` ONLY. A separated ``reasoning`` trace and the
    tool calls the model emitted were real generated tokens, but the chat API hands them back as
    parsed fields with their delimiters gone, so there is no honest way to put them back in
    sequence. For a tool-calling turn the completion is therefore not merely approximate, it is
    incomplete.

Which is the whole argument for the flag: these are useful for a length, a rough alignment or a
cache key, and unusable for a per-token loss.

Deliberately NOT under ``backends/``, which holds exactly the modules ``backend:`` can name — this
one wraps a transport rather than being one, and a name that appears in
:func:`~nanoagent.inference.plugins.available_backends` is a name a config is invited to try.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nanoagent.inference.backend import Backend
from nanoagent.inference.tokenizer import Tokenizer
from nanoagent.inference.types import Fidelity, Response, Tokens


class TokenizingBackend:
    """A :class:`~nanoagent.inference.backend.Backend` that adds reconstructed tokens to another's replies."""

    fidelity = Fidelity.RECONSTRUCTED

    def __init__(self, inner: Backend, tokenizer: Tokenizer) -> None:
        self._inner = inner
        self._tokenizer = tokenizer

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
    ) -> Response:
        """Delegate the call, then fill in :attr:`Response.tokens` from the text that came back.

        Templating and tokenizing are CPU work on the event loop, paid per call — which is why
        this is opt-in behind ``config.tokenizer`` rather than always on. A failed request keeps
        ``tokens`` at ``None``: there is no completion to encode, and a prompt-only record would
        just be a second copy of the request.
        """
        response = await self._inner.generate(messages, tools=tools, on_delta=on_delta)
        if response.error is not None:
            return response
        response.tokens = Tokens(
            prompt_ids=self._tokenizer.render(messages, tools),
            completion_ids=self._tokenizer.encode(response.text or ""),
            fidelity=Fidelity.RECONSTRUCTED,
            tokenizer=self._tokenizer.name,
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()
