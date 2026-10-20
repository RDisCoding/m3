import pytest

from trader.data.providers.ratelimit import SlidingWindowLimiter


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def test_ninth_call_in_a_second_waits_for_the_window():
    clock = FakeClock()
    limiter = SlidingWindowLimiter([(8, 1.0)], clock=clock.now, sleep=clock.sleep)
    waits = [limiter.acquire() for _ in range(9)]
    assert waits[:8] == [0.0] * 8
    assert waits[8] == pytest.approx(1.0)


def test_every_window_is_enforced():
    clock = FakeClock()
    limiter = SlidingWindowLimiter([(8, 1.0), (20, 60.0)], clock=clock.now, sleep=clock.sleep)
    for _ in range(20):
        limiter.acquire()
    assert clock.t == pytest.approx(2.0)  # 8 at t=0, 8 at t=1, 4 at t=2
    limiter.acquire()  # the 60 s window is full until the first call ages out
    assert clock.t == pytest.approx(60.0)


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        SlidingWindowLimiter([])
    with pytest.raises(ValueError):
        SlidingWindowLimiter([(0, 1.0)])
