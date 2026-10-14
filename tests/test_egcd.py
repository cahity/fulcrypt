"""Tests for the extended Euclidean algorithm over the integers and GF(2)[x].

Contract: egcd(a, b) -> (d, u, v) with d = u*a + v*b and d = gcd(a, b), normalized:
  - integers: inputs nonnegative, d >= 0
  - GF(2) polynomials: ints as bit vectors (0b101 = x^2 + 1), so d is automatically monic
gcd(0, 0) is 0.
"""

import math
import random

import pytest

from algebra import Field, Group
from algorithms import xgcd
from elements import GF2Poly

# --- Adapter: edit these two to match your signature -------------------------


def int_egcd(a, b):
    return xgcd(a, b)


def poly_egcd(a, b):
    return tuple(x.bits for x in xgcd(GF2Poly(a), GF2Poly(b)))


# --- GF(2)[x] reference helpers (test-side only) ------------------------------


def pmul(a, b):
    """Carry-less product over GF(2)."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def pmod(a, m):
    """Remainder of a divided by m over GF(2)."""
    while a and a.bit_length() >= m.bit_length():
        a ^= m << (a.bit_length() - m.bit_length())
    return a


def pdivides(d, a):
    return pmod(a, d) == 0 if d else a == 0


# --- Shared assertions --------------------------------------------------------


def check_int(a, b):
    d, u, v = int_egcd(a, b)
    assert all(isinstance(x, int) for x in (d, u, v))
    assert u * a + v * b == d, f"Bezout fails for ({a}, {b}): {u}*{a} + {v}*{b} != {d}"
    assert d == math.gcd(a, b), f"gcd({a}, {b}) should be {math.gcd(a, b)}, got {d}"
    return d, u, v


def check_poly(a, b):
    d, u, v = poly_egcd(a, b)
    assert all(isinstance(x, int) and x >= 0 for x in (d, u, v)), (
        "polynomials must be nonnegative ints"
    )
    assert pmul(u, a) ^ pmul(v, b) == d, f"Bezout fails for ({a:#b}, {b:#b})"
    # Divides both and is a combination of both => it is the gcd (monic is automatic over GF(2)).
    assert pdivides(d, a) and pdivides(d, b), f"{d:#b} does not divide both {a:#b} and {b:#b}"
    return d, u, v


# =============================================================================
# Integers
# =============================================================================


class TestIntegerKnownValues:
    @pytest.mark.parametrize(
        "a, b, d",
        [
            (240, 46, 2),
            (46, 240, 2),
            (12, 18, 6),
            (17, 5, 1),
            (99, 78, 3),
            (1071, 462, 21),
            (270, 192, 6),
            (1, 1, 1),
            (7, 7, 7),
            (13, 1, 1),
            (1, 13, 1),
        ],
    )
    def test_known(self, a, b, d):
        assert check_int(a, b)[0] == d


class TestIntegerEdgeCases:
    def test_both_zero(self):
        d, u, v = int_egcd(0, 0)
        assert d == 0

    @pytest.mark.parametrize("a", [1, 2, 5, 100, 2**61 - 1])
    def test_second_zero(self, a):
        assert check_int(a, 0)[0] == a

    @pytest.mark.parametrize("b", [1, 2, 5, 100, 2**61 - 1])
    def test_first_zero(self, b):
        assert check_int(0, b)[0] == b

    @pytest.mark.parametrize("a, b", [(5, 25), (25, 5), (3, 3 * 2**40), (1, 10**30)])
    def test_one_divides_other(self, a, b):
        assert check_int(a, b)[0] == min(a, b)

    @pytest.mark.parametrize("a", [1, 2, 9, 10**20])
    def test_equal(self, a):
        assert check_int(a, a)[0] == a

    def test_does_not_mutate_or_depend_on_order(self):
        assert int_egcd(84, 36)[0] == int_egcd(36, 84)[0] == 12


class TestIntegerHard:
    def test_consecutive_fibonacci(self):
        # Worst case for the number of division steps.
        f = [0, 1]
        while len(f) < 300:
            f.append(f[-1] + f[-2])
        for i in range(2, 299):
            assert check_int(f[i + 1], f[i])[0] == 1

    @pytest.mark.parametrize(
        "a, b",
        [
            (2**127 - 1, 2**89 - 1),  # Mersenne primes
            (2**521 - 1, 2**607 - 1),
            (10**50 + 151, 10**40 + 7),
            (2**200 * 3**50, 2**150 * 3**80 * 5),
        ],
    )
    def test_large(self, a, b):
        check_int(a, b)

    def test_common_factor(self):
        g = 2**64 + 13
        assert check_int(g * 1009, g * 2017)[0] == g


class TestIntegerRandom:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_small(self, seed):
        rng = random.Random(seed)
        for _ in range(500):
            check_int(rng.randrange(0, 1000), rng.randrange(0, 1000))

    @pytest.mark.parametrize("seed", range(5))
    def test_random_large(self, seed):
        rng = random.Random(100 + seed)
        for _ in range(200):
            check_int(rng.getrandbits(512), rng.getrandbits(256))

    def test_random_with_shared_factor(self):
        rng = random.Random(7)
        for _ in range(200):
            g = rng.randrange(1, 10**6)
            a, b = g * rng.randrange(0, 10**6), g * rng.randrange(0, 10**6)
            d = check_int(a, b)[0]
            assert d % g == 0 or (a == b == 0)


class TestIntegerModularInverse:
    @pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13, 101, 257, 7919])
    def test_inverse_mod_prime(self, p):
        for a in range(1, p):
            d, u, _ = int_egcd(a, p)
            assert d == 1
            assert (u * a) % p == 1
            assert u % p == pow(a, -1, p)

    def test_inverse_mod_large_prime(self):
        p = 2**127 - 1
        rng = random.Random(3)
        for _ in range(200):
            a = rng.randrange(1, p)
            d, u, _ = int_egcd(a, p)
            assert d == 1 and (u * a) % p == 1

    @pytest.mark.parametrize("a, m", [(4, 10), (6, 9), (15, 25), (0, 7)])
    def test_no_inverse_when_not_coprime(self, a, m):
        assert int_egcd(a, m)[0] != 1

    @pytest.mark.parametrize("p", [5, 7, 13])
    def test_agrees_with_field_inverse(self, p):
        f = Field(Group(set(range(p)), lambda a, b: (a + b) % p), lambda a, b: (a * b) % p)
        for a in range(1, p):
            assert int_egcd(a, p)[1] % p == f.inverse(a)


# =============================================================================
# GF(2)[x] polynomials (bit-vector ints)
# =============================================================================

AES = 0x11B  # x^8 + x^4 + x^3 + x + 1
IRREDUCIBLE = {
    2: 0b111,  # x^2 + x + 1
    3: 0b1011,  # x^3 + x + 1
    4: 0b10011,  # x^4 + x + 1
    5: 0b100101,  # x^5 + x^2 + 1
    8: AES,
}


class TestPolyHelpers:
    """Sanity checks on the test-side helpers, so failures point at your code."""

    def test_pmul(self):
        assert pmul(0b11, 0b11) == 0b101  # (x+1)^2 = x^2 + 1
        assert pmul(0b111, 0b11) == 0b1001  # (x^2+x+1)(x+1) = x^3 + 1

    def test_pmod(self):
        assert pmod(0b1001, 0b11) == 0
        assert pmod(0b101, 0b111) == 0b10


class TestPolyKnownValues:
    @pytest.mark.parametrize(
        "a, b, d",
        [
            (0b101, 0b11, 0b11),  # x^2+1 = (x+1)^2
            (0b11, 0b101, 0b11),
            (0b1001, 0b111, 0b111),  # x^3+1 = (x+1)(x^2+x+1)
            (0b1001, 0b101, 0b11),
            (0b111, 0b11, 0b1),  # x^2+x+1 is irreducible
            (0b10, 0b11, 0b1),  # x and x+1
            (0b100, 0b110, 0b10),  # x^2 and x^2+x share x
            (0b1, 0b1, 0b1),
            (AES, 0b11, 0b1),
        ],
    )
    def test_known(self, a, b, d):
        assert check_poly(a, b)[0] == d

    def test_aes_inverse_of_0x53(self):
        # Standard AES example: 0x53 * 0xCA = 1 in GF(2^8).
        d, u, _ = check_poly(0x53, AES)
        assert d == 1
        assert pmod(u, AES) == 0xCA


class TestPolyEdgeCases:
    def test_both_zero(self):
        assert poly_egcd(0, 0)[0] == 0

    @pytest.mark.parametrize("a", [0b1, 0b10, 0b1011, AES])
    def test_second_zero(self, a):
        assert check_poly(a, 0)[0] == a

    @pytest.mark.parametrize("b", [0b1, 0b10, 0b1011, AES])
    def test_first_zero(self, b):
        assert check_poly(0, b)[0] == b

    @pytest.mark.parametrize("a", [0b1, 0b11, 0b1011, AES])
    def test_equal(self, a):
        assert check_poly(a, a)[0] == a

    def test_constant_one(self):
        assert check_poly(0b1, AES)[0] == 1
        assert check_poly(AES, 0b1)[0] == 1

    @pytest.mark.parametrize("f, g", [(0b11, 0b111), (0b1011, 0b10011), (0b10, AES)])
    def test_one_divides_other(self, f, g):
        assert check_poly(f, pmul(f, g))[0] == f
        assert check_poly(pmul(f, g), f)[0] == f

    def test_same_degree(self):
        check_poly(0b1101, 0b1011)

    def test_powers_of_x(self):
        assert check_poly(1 << 10, 1 << 4)[0] == 1 << 4


class TestPolyStructured:
    def test_common_factor_is_found(self):
        # f * g and f * h with g, h distinct irreducibles => gcd is f.
        f = 0b1011
        g, h = 0b111, 0b10011
        assert check_poly(pmul(f, g), pmul(f, h))[0] == f

    def test_powers_of_x_plus_one(self):
        for i in range(1, 20):
            for j in range(1, 20):
                a = b = 1
                for _ in range(i):
                    a = pmul(a, 0b11)
                for _ in range(j):
                    b = pmul(b, 0b11)
                assert check_poly(a, b)[0] == (a if i <= j else b)

    def test_high_degree(self):
        a = (1 << 233) | (1 << 74) | 1  # x^233 + x^74 + 1 (NIST B-233 trinomial)
        b = (1 << 163) | 0b11001001  # x^163 + x^7 + x^6 + x^3 + 1 (B-163)
        check_poly(a, b)


class TestPolyRandom:
    @pytest.mark.parametrize("seed", range(5))
    def test_random_small(self, seed):
        rng = random.Random(seed)
        for _ in range(500):
            check_poly(rng.getrandbits(10), rng.getrandbits(10))

    @pytest.mark.parametrize("seed", range(3))
    def test_random_large(self, seed):
        rng = random.Random(50 + seed)
        for _ in range(100):
            check_poly(rng.getrandbits(128), rng.getrandbits(64))

    def test_random_with_shared_factor(self):
        rng = random.Random(9)
        for _ in range(200):
            f = rng.getrandbits(8) or 1
            a = pmul(f, rng.getrandbits(16))
            b = pmul(f, rng.getrandbits(16))
            d = check_poly(a, b)[0]
            assert pdivides(f, d) or (a == b == 0)


class TestPolyFieldInverse:
    @pytest.mark.parametrize("n", sorted(IRREDUCIBLE))
    def test_every_nonzero_element_invertible(self, n):
        m = IRREDUCIBLE[n]
        for a in range(1, 1 << n):
            d, u, _ = check_poly(a, m)
            assert d == 1
            assert pmod(pmul(pmod(u, m), a), m) == 1

    @pytest.mark.parametrize(
        "m, a",
        [
            (0b101, 0b11),  # x^2+1 is reducible: (x+1)^2
            (0b1001, 0b111),  # x^3+1 = (x+1)(x^2+x+1)
            (0b10001, 0b11),  # x^4+1 = (x+1)^4
        ],
    )
    def test_reducible_modulus_has_noninvertible(self, m, a):
        assert check_poly(a, m)[0] != 1

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_agrees_with_field_inverse(self, n):
        m = IRREDUCIBLE[n]
        add = Group(set(range(1 << n)), lambda a, b: a ^ b)
        f = Field(add, lambda a, b: pmod(pmul(a, b), m))
        for a in range(1, 1 << n):
            assert pmod(poly_egcd(a, m)[1], m) == f.inverse(a)


# =============================================================================
# gcd only: checks d and ignores u, v.  Run with: pytest -m gcd
# =============================================================================


def int_gcd(a, b):
    return int_egcd(a, b)[0]


def poly_gcd(a, b):
    return poly_egcd(a, b)[0]


def deg_of(a):
    return a.bit_length() - 1


def brute_poly_gcd(a, b):
    """Highest-degree common divisor by trying every candidate (small inputs only)."""
    if a == 0 or b == 0:
        return a or b
    best = 1
    for d in range(1, 1 << (min(a.bit_length(), b.bit_length()))):
        if pdivides(d, a) and pdivides(d, b):
            best = d  # candidates grow in degree, so the last hit is the gcd
    return best


@pytest.mark.gcd
class TestIntegerGcdOnly:
    @pytest.mark.parametrize(
        "a, b, d",
        [
            (240, 46, 2),
            (46, 240, 2),
            (12, 18, 6),
            (17, 5, 1),
            (99, 78, 3),
            (1071, 462, 21),
            (270, 192, 6),
            (1, 1, 1),
            (7, 7, 7),
            (13, 1, 1),
            (1, 13, 1),
            (0, 0, 0),
            (5, 0, 5),
            (0, 5, 5),
            (5, 25, 5),
            (25, 5, 5),
            (84, 36, 12),
        ],
    )
    def test_known(self, a, b, d):
        assert int_gcd(a, b) == d

    def test_returns_int(self):
        assert isinstance(int_gcd(12, 18), int)

    def test_consecutive_fibonacci(self):
        f = [0, 1]
        while len(f) < 300:
            f.append(f[-1] + f[-2])
        for i in range(2, 299):
            assert int_gcd(f[i + 1], f[i]) == 1

    @pytest.mark.parametrize(
        "a, b",
        [
            (2**127 - 1, 2**89 - 1),
            (2**521 - 1, 2**607 - 1),
            (2**200 * 3**50, 2**150 * 3**80 * 5),
            ((2**64 + 13) * 1009, (2**64 + 13) * 2017),
        ],
    )
    def test_large(self, a, b):
        assert int_gcd(a, b) == math.gcd(a, b)

    @pytest.mark.parametrize("seed", range(5))
    def test_random(self, seed):
        rng = random.Random(seed)
        for _ in range(500):
            a, b = rng.randrange(0, 1000), rng.randrange(0, 1000)
            assert int_gcd(a, b) == math.gcd(a, b), (a, b)
        for _ in range(200):
            a, b = rng.getrandbits(512), rng.getrandbits(256)
            assert int_gcd(a, b) == math.gcd(a, b), (a, b)

    def test_random_with_shared_factor(self):
        rng = random.Random(7)
        for _ in range(200):
            g = rng.randrange(1, 10**6)
            a, b = g * rng.randrange(0, 10**6), g * rng.randrange(0, 10**6)
            assert int_gcd(a, b) == math.gcd(a, b), (a, b)


@pytest.mark.gcd
class TestPolyGcdOnly:
    @pytest.mark.parametrize(
        "a, b, d",
        [
            (0b101, 0b11, 0b11),
            (0b11, 0b101, 0b11),
            (0b1001, 0b111, 0b111),
            (0b1001, 0b101, 0b11),
            (0b111, 0b11, 0b1),
            (0b10, 0b11, 0b1),
            (0b100, 0b110, 0b10),
            (0b1, 0b1, 0b1),
            (AES, 0b11, 0b1),
            (0x53, AES, 0b1),
            (0, 0, 0),
            (AES, 0, AES),
            (0, AES, AES),
            (AES, AES, AES),
            (1 << 10, 1 << 4, 1 << 4),
            (0b1101, 0b1011, 0b1),
        ],
    )
    def test_known(self, a, b, d):
        assert poly_gcd(a, b) == d

    def test_returns_nonnegative_int(self):
        d = poly_gcd(0b101, 0b11)
        assert isinstance(d, int) and d >= 0

    def test_brute_force_helper(self):
        assert brute_poly_gcd(0b1001, 0b101) == 0b11

    @pytest.mark.parametrize("f, g", [(0b11, 0b111), (0b1011, 0b10011), (0b10, AES)])
    def test_one_divides_other(self, f, g):
        assert poly_gcd(f, pmul(f, g)) == f
        assert poly_gcd(pmul(f, g), f) == f

    def test_common_factor_of_distinct_irreducibles(self):
        # f*g and f*h with g, h distinct irreducibles => gcd is exactly f.
        f = 0b1011
        assert poly_gcd(pmul(f, 0b111), pmul(f, 0b10011)) == f
        f = pmul(0b11, 0b100101)
        assert poly_gcd(pmul(f, 0b1011), pmul(f, AES)) == f

    def test_powers_of_x_plus_one(self):
        p = [1]
        for _ in range(20):
            p.append(pmul(p[-1], 0b11))
        for i in range(1, 20):
            for j in range(1, 20):
                assert poly_gcd(p[i], p[j]) == p[min(i, j)]

    def test_every_irreducible_pair_is_coprime(self):
        for m in IRREDUCIBLE.values():
            for a in range(1, 1 << deg_of(m)):
                assert poly_gcd(a, m) == 1, (a, m)

    @pytest.mark.parametrize("seed", range(5))
    def test_random_against_brute_force(self, seed):
        rng = random.Random(seed)
        for _ in range(150):
            a, b = rng.getrandbits(9), rng.getrandbits(9)
            assert poly_gcd(a, b) == brute_poly_gcd(a, b), (bin(a), bin(b))

    def test_random_with_shared_factor(self):
        rng = random.Random(9)
        for _ in range(200):
            f = rng.getrandbits(8) or 1
            a, b = pmul(f, rng.getrandbits(16)), pmul(f, rng.getrandbits(16))
            d = poly_gcd(a, b)
            assert pdivides(d, a) and pdivides(d, b) and (pdivides(f, d) or a == b == 0)
