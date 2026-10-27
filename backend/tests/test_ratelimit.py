"""Sliding-window rate limiter (deterministic fake clock)."""

from __future__ import annotations

import pytest

from app.ratelimit import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_limit_then_blocks_with_retry_after():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(3, 60.0, clock=clock)
    assert [limiter.check("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry = limiter.check("a")
    assert not allowed and retry == 60


def test_window_slides_and_clients_are_independent():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(1, 60.0, clock=clock)
    assert limiter.check("a")[0]
    assert limiter.check("b")[0]  # separate client unaffected
    assert not limiter.check("a")[0]
    clock.now += 61
    assert limiter.check("a")[0]  # old hit expired


def test_client_table_is_bounded():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(5, 60.0, max_clients=3, clock=clock)
    for i in range(10):
        limiter.check(f"client-{i}")
    assert len(limiter._hits) <= 3


def test_invalid_limit_rejected():
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(0)
