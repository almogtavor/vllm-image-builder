"""A backend pod restart must stall a request, never kill the session.

An engine crash takes the pod down for minutes. The old ladder gave up after
~1.85s, so the agent saw a 500 and the session died -- losing a benchmark cell
to a fault it could have waited out.
"""
import httpx
import pytest

import render_client as rc


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    slept = []
    monkeypatch.setattr(rc.time, "sleep", slept.append)
    return slept


def _req():
    return httpx.Request("POST", "http://backend/v1/chat/completions/render")


def _status_error(code):
    return httpx.HTTPStatusError(
        str(code), request=_req(), response=httpx.Response(code, request=_req())
    )


class TestBackendIsDown:
    def test_connect_failures_are_down(self):
        assert rc.backend_is_down(httpx.ConnectError("refused", request=_req()))
        assert rc.backend_is_down(httpx.ConnectTimeout("timeout", request=_req()))

    def test_wrapped_connect_error_is_down(self):
        # The OpenAI SDK re-raises httpx failures as APIConnectionError.
        try:
            try:
                raise httpx.ConnectError("refused", request=_req())
            except httpx.ConnectError as exc:
                raise RuntimeError("APIConnectionError") from exc
        except RuntimeError as outer:
            assert rc.backend_is_down(outer)

    def test_503_is_down_by_status_attr_or_response(self):
        assert rc.backend_is_down(_status_error(503))

        class SdkError(Exception):
            status_code = 503

        assert rc.backend_is_down(SdkError())

    def test_read_timeout_is_not_down(self):
        # The request was accepted; re-sending only adds load.
        assert not rc.backend_is_down(httpx.ReadTimeout("slow", request=_req()))

    def test_other_statuses_are_not_down(self):
        assert not rc.backend_is_down(_status_error(500))
        assert not rc.backend_is_down(_status_error(400))


class TestRidingOutRestart:
    def test_retries_until_the_backend_returns(self, _no_real_sleeping):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 4:
                raise httpx.ConnectError("refused", request=_req())
            return "ok"

        assert rc.riding_out_restart(flaky) == "ok"
        assert len(calls) == 4
        assert len(_no_real_sleeping) == 3

    def test_gives_up_at_the_deadline(self, monkeypatch):
        monkeypatch.setattr(rc, "_DOWN_WAIT", 0.0)

        def dead():
            raise httpx.ConnectError("refused", request=_req())

        with pytest.raises(httpx.ConnectError):
            rc.riding_out_restart(dead)

    def test_does_not_retry_an_unrelated_error(self):
        def boom():
            raise ValueError("bug in prompt build")

        with pytest.raises(ValueError):
            rc.riding_out_restart(boom)


class _Http:
    """Raises `fail` times, then returns `final`."""

    def __init__(self, exc, fail, final=None):
        self._exc, self._fail, self._final = exc, fail, final
        self.calls = 0

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        if self.calls <= self._fail:
            raise self._exc
        return self._final


class TestPostRidesOutRestart:
    def _client(self, http):
        return rc.RenderClient("http://backend/v1", "m", http)

    def test_outage_outlasting_the_burst_ladder_still_succeeds(self):
        ok = httpx.Response(200, json={"token_ids": [1]}, request=_req())
        http = _Http(httpx.ConnectError("refused", request=_req()), fail=20, final=ok)
        assert self._client(http).render_chat([{"role": "user", "content": "hi"}]) == [1]
        # 4 fast ladder attempts could never span a pod restart on their own.
        assert http.calls > len(rc._RETRY_BACKOFF) + 1

    def test_4xx_is_not_retried(self):
        bad = httpx.Response(400, json={}, request=_req())
        http = _Http(None, fail=0, final=bad)
        with pytest.raises(httpx.HTTPStatusError):
            self._client(http).render_chat([{"role": "user", "content": "hi"}])
        assert http.calls == 1

    def test_read_timeout_stops_at_the_ladder(self, monkeypatch):
        http = _Http(httpx.ReadTimeout("slow", request=_req()), fail=99)
        with pytest.raises(httpx.ReadTimeout):
            self._client(http).render_chat([{"role": "user", "content": "hi"}])
        assert http.calls == len(rc._RETRY_BACKOFF) + 1
