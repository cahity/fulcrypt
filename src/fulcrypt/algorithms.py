"""Generic algorithms. They use only operators, so they work on int and every element class."""

from typing import Any


def xgcd(a: Any, b: Any) -> tuple[Any, Any, Any]:
    """Return (d, u, v) where d = gcd(a, b) = u*a + v*b, for nonnegative a and b."""
    cls = type(a)
    zero, one = cls(0), cls(1)

    if isinstance(a, int) and (a < zero or b < zero):
        raise ValueError("xgcd expects nonnegative inputs")

    # Invariant: old_r = old_u*a + old_v*b and r = u*a + v*b
    old_r, r = a, b
    old_s, s = one, zero
    old_t, t = zero, one

    while r:
        q, remainder = divmod(old_r, r)
        old_r, r = r, remainder
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    return old_r, old_s, old_t


def inverse_mod(a: Any, m: Any) -> Any:
    """Return the multiplicative inverse of a modulo m, if it exists."""
    d, u, _ = xgcd(a, m)
    if d != type(a)(1):
        raise ValueError(f"{a} has no inverse modulo {m}")
    return u % m


def power(a: Any, e: int) -> Any:
    """Exponentiation by squaring, for nonnegative integer e."""
    if not isinstance(e, int):
        return NotImplemented
    if e < 0:
        raise ValueError("power expects a nonnegative exponent")
    result = type(a)(1)
    base = a
    while e:
        if e & 1:
            result *= base
        base *= base
        e >>= 1
    return result
