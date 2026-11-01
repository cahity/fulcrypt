"""Tests for prime_divisors(n): the distinct primes dividing n, in increasing order."""

import math

import pytest

from fulcrypt.algorithms import prime_divisors


def brute_force(n):
    def is_prime(p):
        return p >= 2 and all(p % d for d in range(2, math.isqrt(p) + 1))

    return [p for p in range(2, n + 1) if n % p == 0 and is_prime(p)]


class TestPrimeDivisors:
    def test_one_has_none(self):
        assert prime_divisors(1) == []

    @pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13, 127, 163, 233, 997])
    def test_prime_divides_itself(self, p):
        # The case that matters for Rabin: a prime degree n has q = n.
        assert prime_divisors(p) == [p]

    @pytest.mark.parametrize(
        "n, primes",
        [
            (4, [2]),
            (8, [2]),
            (128, [2]),  # GCM: only q = 2
            (9, [3]),
            (125, [5]),
            (6, [2, 3]),  # the largest factor is above sqrt(n)
            (10, [2, 5]),
            (14, [2, 7]),
            (12, [2, 3]),
            (30, [2, 3, 5]),
            (50, [2, 5]),
            (210, [2, 3, 5, 7]),
            (254, [2, 127]),
            (1024, [2]),
            (1001, [7, 11, 13]),
        ],
    )
    def test_known(self, n, primes):
        assert prime_divisors(n) == primes

    def test_matches_brute_force_up_to_500(self):
        for n in range(1, 501):
            assert prime_divisors(n) == brute_force(n), n

    def test_each_prime_listed_once(self):
        # 2^7 * 3^4 has two distinct primes, not eleven.
        assert prime_divisors(2**7 * 3**4) == [2, 3]

    def test_sorted_and_distinct(self):
        for n in (360, 2310, 9699690 // 19):
            primes = prime_divisors(n)
            assert primes == sorted(set(primes))

    def test_product_rebuilds_the_radical(self):
        # The product of the distinct primes divides n, and n's prime factors are all there.
        for n in range(2, 300):
            radical = math.prod(prime_divisors(n))
            assert n % radical == 0
            m = n
            for p in prime_divisors(n):
                while m % p == 0:
                    m //= p
            assert m == 1, n

    def test_returns_list_of_ints(self):
        primes = prime_divisors(60)
        assert type(primes) is list
        assert all(type(p) is int for p in primes)
