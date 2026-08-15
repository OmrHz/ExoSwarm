from __future__ import annotations

from typing import Any

import httpx
import pytest

from exoswarm.agents.provider import (
    FeatherlessProvider,
    ProviderError,
    ProviderErrorCategory,
)


def _provider(api_key: str = "test-provider-secret") -> FeatherlessProvider:
    return FeatherlessProvider(
        api_key=api_key,
        model="deepseek-ai/DeepSeek-V4-Flash",
        timeout_seconds=3,
    )


def _response(status_code: int, *, payload: dict[str, Any]) -> httpx.Response:
    request = httpx.Request("POST", "https://api.featherless.ai/v1/chat/completions")
    return httpx.Response(status_code, request=request, json=payload)


def test_featherless_request_uses_json_object_mode_and_parses_envelope(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update({"url": url, **kwargs})
        return _response(
            200,
            payload={
                "id": "chatcmpl-test",
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"verdict":"APPROVE"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    response = _provider().complete(system="Return JSON.", user="{}")

    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["model"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert captured["headers"]["Authorization"] == "Bearer test-provider-secret"
    assert response.content == '{"verdict":"APPROVE"}'
    assert response.request_id == "chatcmpl-test"
    assert response.usage.total_tokens == 16


@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (401, ProviderErrorCategory.AUTHENTICATION),
        (403, ProviderErrorCategory.AUTHORIZATION),
        (404, ProviderErrorCategory.NOT_FOUND),
        (429, ProviderErrorCategory.RATE_LIMIT),
        (503, ProviderErrorCategory.SERVICE),
        (418, ProviderErrorCategory.HTTP_STATUS),
    ],
)
def test_http_failures_expose_only_safe_status_and_category(
    monkeypatch, status_code: int, category: ProviderErrorCategory
) -> None:
    secret = "credential-must-not-escape"
    provider_body = {"error": {"message": f"diagnostic includes {secret}"}}

    def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return _response(status_code, payload=provider_body)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ProviderError) as captured:
        _provider(secret).complete(system="system", user="user")

    error = captured.value
    assert error.category is category
    assert error.status_code == status_code
    assert f"HTTP {status_code}" in str(error)
    assert secret not in str(error)
    assert "diagnostic" not in str(error)


def test_timeout_is_categorized_without_request_details(monkeypatch) -> None:
    request = httpx.Request("POST", "https://api.featherless.ai/v1/chat/completions")

    def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        raise httpx.ReadTimeout("sensitive transport detail", request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ProviderError) as captured:
        _provider().complete(system="system", user="user")

    assert captured.value.category is ProviderErrorCategory.TIMEOUT
    assert captured.value.status_code is None
    assert str(captured.value) == "Featherless request timed out"
    assert "sensitive transport detail" not in str(captured.value)


def test_invalid_completion_envelope_is_categorized(monkeypatch) -> None:
    def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return _response(200, payload={"choices": []})

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ProviderError) as captured:
        _provider().complete(system="system", user="user")

    assert captured.value.category is ProviderErrorCategory.INVALID_RESPONSE
    assert captured.value.status_code is None
    assert "completion envelope" in str(captured.value)
