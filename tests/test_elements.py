"""Tests for element classes. GF2Poly stores GF(2)[x] as a bit vector: 0b1011 = x^3 + x + 1."""

import dataclasses
import random

import pytest

from fulcrypt.elements import GF2Poly

P = GF2Poly


def rand_poly(rng, max_degree=12):
    return P(rng.getrandbits(max_degree + 1))


# =============================================================================
# GF2Poly: construction and basics
# =============================================================================


class TestGF2PolyBasics:
    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            P(-1)

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            P(3).bits = 5

    def test_equality_and_hash(self):
        assert P(5) == P(5)
        assert P(5) != P(6)
        assert len({P(5), P(5), P(6)}) == 2

    def test_not_equal_to_int(self):
        # Ints are not coerced, so a GF2Poly never equals an int.
        assert P(1) != 1
        assert P(0) != 0

    @pytest.mark.parametrize("bits, degree", [(0, -1), (1, 0), (0b10, 1), (0b1011, 3), (0x11B, 8)])
    def test_degree(self, bits, degree):
        assert P(bits).degree == degree

    def test_bool(self):
        assert not P(0)
        assert P(1)
        assert P(0b100)

    @pytest.mark.parametrize(
        "bits, text",
        [
            (0, "0"),
            (1, "1"),
            (0b10, "x"),
            (0b11, "x + 1"),
            (0b1011, "x^3 + x + 1"),
            (0x11B, "x^8 + x^4 + x^3 + x + 1"),
        ],
    )
    def test_repr(self, bits, text):
        assert repr(P(bits)) == text


# =============================================================================
# GF2Poly: arithmetic
# =============================================================================


class TestGF2PolyArithmetic:
    def test_add_is_xor(self):
        assert P(0b1100) + P(0b1010) == P(0b0110)

    def test_add_returns_poly(self):
        assert isinstance(P(1) + P(2), P)

    def test_add_self_is_zero(self):
        assert P(0b1011) + P(0b1011) == P(0)

    def test_sub_equals_add(self):
        assert P(0b1100) - P(0b1010) == P(0b1100) + P(0b1010)

    def test_neg_is_identity(self):
        assert -P(0b1011) == P(0b1011)

    @pytest.mark.parametrize(
        "a, b, product",
        [
            (0b11, 0b11, 0b101),  # (x+1)^2 = x^2 + 1
            (0b111, 0b11, 0b1001),  # (x^2+x+1)(x+1) = x^3 + 1
            (0b1011, 0b1, 0b1011),  # times 1
            (0b1011, 0b0, 0b0),  # times 0
            (0b10, 0b10, 0b100),  # x * x = x^2
        ],
    )
    def test_mul(self, a, b, product):
        assert P(a) * P(b) == P(product)
        assert P(b) * P(a) == P(product)

    @pytest.mark.parametrize(
        "a, b, q, r",
        [
            (0b1001, 0b11, 0b111, 0b0),  # x^3 + 1 = (x+1)(x^2+x+1)
            (0b1011, 0b11, 0b110, 0b1),  # x^3+x+1 = (x+1)(x^2+x) + 1
            (0b11, 0b1011, 0b0, 0b11),  # smaller degree: quotient 0
            (0b0, 0b11, 0b0, 0b0),  # zero numerator
            (0b1011, 0b1, 0b1011, 0b0),  # divide by 1
        ],
    )
    def test_divmod_known(self, a, b, q, r):
        assert divmod(P(a), P(b)) == (P(q), P(r))
        assert P(a) // P(b) == P(q)
        assert P(a) % P(b) == P(r)

    def test_divmod_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            divmod(P(0b101), P(0))
        with pytest.raises(ZeroDivisionError):
            P(0b101) % P(0)
        with pytest.raises(ZeroDivisionError):
            P(0b101) // P(0)

    @pytest.mark.parametrize(
        "base, e, result",
        [
            (0b11, 0, 0b1),
            (0b11, 1, 0b11),
            (0b11, 2, 0b101),  # (x+1)^2 = x^2 + 1
            (0b11, 4, 0b10001),  # (x+1)^4 = x^4 + 1
            (0b10, 10, 1 << 10),  # x^10
            (0b0, 0, 0b1),  # 0^0 = 1 by convention
            (0b0, 3, 0b0),
        ],
    )
    def test_pow(self, base, e, result):
        assert P(base) ** e == P(result)

    def test_pow_negative_exponent(self):
        with pytest.raises(ValueError):
            P(0b11) ** -1


# =============================================================================
# GF2Poly: type safety (no silent int coercion)
# =============================================================================


class TestGF2PolyTypeErrors:
    @pytest.mark.parametrize(
        "op",
        [
            lambda: P(1) + 1,
            lambda: 1 + P(1),
            lambda: P(1) - 1,
            lambda: P(3) * 2,
            lambda: 2 * P(3),
            lambda: P(3) % 2,
            lambda: P(3) // 2,
            lambda: divmod(P(3), 2),
            lambda: P(3) ** P(2),
        ],
    )
    def test_mixed_with_int_raises(self, op):
        with pytest.raises(TypeError):
            op()


# =============================================================================
# GF2Poly: ring laws on random inputs
# =============================================================================


class TestGF2PolyRingLaws:
    N = 300

    def test_add_commutative_associative(self):
        rng = random.Random(1)
        for _ in range(self.N):
            a, b, c = (rand_poly(rng) for _ in range(3))
            assert a + b == b + a
            assert (a + b) + c == a + (b + c)

    def test_mul_commutative_associative(self):
        rng = random.Random(2)
        for _ in range(self.N):
            a, b, c = (rand_poly(rng) for _ in range(3))
            assert a * b == b * a
            assert (a * b) * c == a * (b * c)

    def test_distributive(self):
        rng = random.Random(3)
        for _ in range(self.N):
            a, b, c = (rand_poly(rng) for _ in range(3))
            assert a * (b + c) == a * b + a * c

    def test_degree_of_product(self):
        rng = random.Random(4)
        for _ in range(self.N):
            a, b = rand_poly(rng), rand_poly(rng)
            if a and b:
                assert (a * b).degree == a.degree + b.degree

    def test_division_identity(self):
        # a = q*b + r with deg r < deg b
        rng = random.Random(5)
        for _ in range(self.N):
            a = rand_poly(rng, 20)
            b = rand_poly(rng, 10)
            if not b:
                continue
            q, r = divmod(a, b)
            assert q * b + r == a
            assert r.degree < b.degree

    def test_frobenius(self):
        # Over GF(2): (a + b)^2 = a^2 + b^2
        rng = random.Random(6)
        for _ in range(self.N):
            a, b = rand_poly(rng), rand_poly(rng)
            assert (a + b) ** 2 == a**2 + b**2

    def test_pow_matches_repeated_mul(self):
        rng = random.Random(7)
        for _ in range(50):
            a = rand_poly(rng, 6)
            e = rng.randrange(0, 12)
            expected = P(1)
            for _ in range(e):
                expected = expected * a
            assert a**e == expected


# =============================================================================
# GF2Poly: irreducibility
# =============================================================================


class TestGF2PolyIrreducible:
    @pytest.mark.parametrize(
        "bits",
        [
            0b10,
            0b11,  # x, x + 1
            0b111,  # x^2 + x + 1
            0b1011,
            0b1101,  # degree 3
            0b10011,
            0b11001,
            0b11111,  # degree 4
            0x11B,  # AES modulus
        ],
    )
    def test_irreducible(self, bits):
        assert P(bits).is_irreducible()

    @pytest.mark.parametrize(
        "bits",
        [
            0b0,
            0b1,  # zero and units are not irreducible
            0b100,  # x^2
            0b101,  # (x + 1)^2
            0b110,  # x(x + 1)
            0b1001,  # (x + 1)(x^2 + x + 1)
            0b10001,  # (x + 1)^4
            0b10101,  # (x^2 + x + 1)^2
        ],
    )
    def test_reducible(self, bits):
        assert not P(bits).is_irreducible()

    @pytest.mark.parametrize(
        "degree, count", [(1, 2), (2, 1), (3, 2), (4, 3), (5, 6), (6, 9), (7, 18), (8, 30)]
    )
    def test_count_by_degree(self, degree, count):
        # Known counts of irreducible polynomials over GF(2) (OEIS A001037).
        found = sum(P(b).is_irreducible() for b in range(1 << degree, 1 << (degree + 1)))
        assert found == count

    def test_product_of_two_is_reducible(self):
        rng = random.Random(8)
        for _ in range(100):
            a = P(rng.randrange(2, 1 << 6))
            b = P(rng.randrange(2, 1 << 6))
            assert not (a * b).is_irreducible()


# =============================================================================
# Step 13: Rabin's irreducibility test (fast enough for GF(2^128))
# =============================================================================
# f of degree n is irreducible iff x^(2^n) = x (mod f) and gcd(x^(2^(n/q)) - x, f) = 1
# for every prime q dividing n. Trial division would need ~2^65 divisions for degree 128.


def poly(*exponents):
    return P(sum(1 << e for e in exponents))


def trial_division(f):
    # The old definition, as an independent reference for small degrees.
    if f.bits < 2:
        return False
    return all(f % P(g) != P(0) for g in range(2, 1 << (f.degree // 2 + 1)))


GCM_POLY = poly(128, 7, 2, 1, 0)
DEG64_A = poly(64, 4, 3, 1, 0)  # irreducible
DEG64_B = poly(64, 11, 2, 1, 0)  # irreducible


class TestGF2PolyRabin:
    def test_agrees_with_trial_division_up_to_degree_10(self):
        for bits in range(1 << 11):
            assert P(bits).is_irreducible() == trial_division(P(bits)), P(bits)

    @pytest.mark.parametrize("degree, count", [(9, 56), (10, 99), (11, 186), (12, 335)])
    def test_count_by_degree(self, degree, count):
        # OEIS A001037, beyond what trial division could check quickly.
        found = sum(P(b).is_irreducible() for b in range(1 << degree, 1 << (degree + 1)))
        assert found == count

    @pytest.mark.parametrize(
        "f",
        [
            pytest.param(GCM_POLY, id="GCM x^128+x^7+x^2+x+1"),
            pytest.param(DEG64_A, id="x^64+x^4+x^3+x+1"),
            pytest.param(poly(127, 1, 0), id="x^127+x+1 (prime degree)"),
            pytest.param(poly(163, 7, 6, 3, 0), id="NIST B-163"),
            pytest.param(poly(233, 74, 0), id="NIST B-233 trinomial"),
        ],
    )
    def test_large_irreducible(self, f):
        assert f.is_irreducible()

    @pytest.mark.parametrize(
        "f",
        [
            # Factor degrees 64 and 64 both divide 128, so x^(2^128) = x (mod f) holds:
            # only the gcd condition (q = 2) catches these.
            pytest.param(DEG64_A * DEG64_B, id="deg 64 * deg 64"),
            pytest.param(DEG64_A * DEG64_A, id="deg 64 squared"),
            # Factor degree 3 doesn't divide 128: the first condition fails.
            pytest.param(poly(3, 1, 0) * poly(125, 72, 2, 1, 0), id="deg 3 * deg 125"),
            pytest.param(poly(128, 0), id="x^128+1 = (x+1)^128"),
            pytest.param(poly(128, 1, 0), id="x^128+x+1"),
            pytest.param(poly(128, 7, 2, 1), id="x^128+x^7+x^2+x (divisible by x)"),
        ],
    )
    def test_large_reducible(self, f):
        assert not f.is_irreducible()

    def test_gf2n_accepts_gcm_modulus(self):
        # Constructing the field runs the irreducibility check.
        from fulcrypt.elements import GF2n

        F = GF2n(GCM_POLY.bits)
        assert F.modulus.degree == 128

    def test_gf2n_rejects_large_reducible(self):
        from fulcrypt.elements import GF2n

        with pytest.raises(ValueError):
            GF2n((DEG64_A * DEG64_B).bits)
