"""Tests for ZMod(n): a class factory for the integers modulo n."""

import dataclasses
import math
import random

import pytest

from fulcrypt.algorithms import power
from fulcrypt.elements import GF2Poly, ZMod

Z7 = ZMod(7)
Z11 = ZMod(11)


# =============================================================================
# The factory
# =============================================================================


class TestFactory:
    def test_cached(self):
        # Same modulus -> same class, otherwise ZMod(7)(3) != ZMod(7)(3).
        assert ZMod(7) is ZMod(7)
        assert ZMod(7)(3) == ZMod(7)(3)

    def test_different_moduli_are_different_classes(self):
        assert ZMod(7) is not ZMod(11)

    def test_name(self):
        assert ZMod(7).__name__ == "Z7"

    def test_modulus_attribute(self):
        assert Z7.modulus == 7
        assert Z7(3).modulus == 7

    @pytest.mark.parametrize("n", [-5, 0, 1])
    def test_rejects_small_modulus(self, n):
        with pytest.raises(ValueError):
            ZMod(n)

    def test_all(self):
        assert ZMod(5).all() == [ZMod(5)(i) for i in range(5)]
        assert len(ZMod(12).all()) == 12


# =============================================================================
# Elements: construction, equality, hashing
# =============================================================================


class TestElement:
    @pytest.mark.parametrize(
        "raw, reduced", [(3, 3), (10, 3), (7, 0), (-1, 6), (-15, 6), (10**20, 10**20 % 7)]
    )
    def test_reduces_on_construction(self, raw, reduced):
        assert Z7(raw).value == reduced

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Z7(3).value = 4

    def test_hash(self):
        assert len({Z7(3), Z7(10), Z7(-4), Z7(4)}) == 2

    def test_not_equal_to_int(self):
        # Equal objects must have equal hashes. Z7(3) "equals" both 3 and 10,
        # but hash(3) != hash(10), so equality with int has to stay False.
        assert Z7(3) != 3

    def test_not_equal_across_moduli(self):
        assert Z7(3) != Z11(3)

    def test_int(self):
        assert int(Z7(10)) == 3

    def test_bool(self):
        assert not Z7(0)
        assert not Z7(7)
        assert Z7(1)

    def test_repr(self):
        assert repr(Z7(10)) == "3 (mod 7)"


# =============================================================================
# Arithmetic
# =============================================================================


class TestArithmetic:
    def test_add(self):
        assert Z7(4) + Z7(5) == Z7(2)

    def test_sub(self):
        assert Z7(2) - Z7(5) == Z7(4)

    def test_neg(self):
        assert -Z7(3) == Z7(4)
        assert -Z7(0) == Z7(0)

    def test_mul(self):
        assert Z7(3) * Z7(5) == Z7(1)

    def test_results_keep_class(self):
        a, b = Z7(3), Z7(5)
        for result in (a + b, a - b, a * b, -a, a / b, a**3):
            assert type(result) is Z7

    @pytest.mark.parametrize(
        "expr, expected",
        [
            (lambda: Z7(4) + 5, 2),
            (lambda: 5 + Z7(4), 2),
            (lambda: Z7(4) - 5, 6),
            (lambda: 5 - Z7(4), 1),  # needs __rsub__: 5 - 4, not 4 - 5
            (lambda: Z7(4) * 2, 1),
            (lambda: 2 * Z7(4), 1),
            (lambda: Z7(6) / 2, 3),
            (lambda: 1 / Z7(3), 5),  # needs __rtruediv__
        ],
    )
    def test_int_coercion(self, expr, expected):
        result = expr()
        assert type(result) is Z7
        assert result == Z7(expected)

    @pytest.mark.parametrize(
        "op",
        [
            lambda: Z7(1) + Z11(1),
            lambda: Z7(1) - Z11(1),
            lambda: Z7(1) * Z11(1),
            lambda: Z7(1) / Z11(1),
            lambda: Z7(1) + GF2Poly(1),
            lambda: Z7(1) * 1.5,
            lambda: Z7(3) ** Z7(2),  # exponents are ints, not elements of Z7
        ],
    )
    def test_mixing_raises_type_error(self, op):
        with pytest.raises(TypeError):
            op()


# =============================================================================
# Inverses and division
# =============================================================================


class TestInverse:
    def test_known(self):
        assert Z7(3).inverse() == Z7(5)

    @pytest.mark.parametrize("p", [2, 3, 5, 7, 13, 101])
    def test_every_nonzero_invertible_mod_prime(self, p):
        Zp = ZMod(p)
        for a in Zp.all()[1:]:
            assert a * a.inverse() == Zp(1)

    def test_no_inverse_when_not_coprime(self):
        with pytest.raises(ValueError):
            ZMod(6)(2).inverse()

    def test_zero_has_no_inverse(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            Z7(0).inverse()
        with pytest.raises((ValueError, ZeroDivisionError)):
            Z7(3) / Z7(0)

    @pytest.mark.parametrize("n", [12, 15, 30, 97])
    def test_units_count_is_euler_phi(self, n):
        Zn = ZMod(n)
        units = 0
        for a in Zn.all():
            try:
                a.inverse()
                units += 1
            except ValueError, ZeroDivisionError:
                pass
        assert units == sum(math.gcd(k, n) == 1 for k in range(1, n))

    def test_div(self):
        assert Z7(3) / Z7(3) == Z7(1)
        assert Z7(1) / Z7(3) == Z7(5)
        assert (Z7(2) / Z7(5)) * Z7(5) == Z7(2)


# =============================================================================
# Powers
# =============================================================================


class TestPower:
    @pytest.mark.parametrize(
        "a, e, expected", [(3, 0, 1), (3, 1, 3), (3, 2, 2), (3, 6, 1), (0, 0, 1), (0, 5, 0)]
    )
    def test_known(self, a, e, expected):
        assert Z7(a) ** e == Z7(expected)

    def test_negative_exponent(self):
        assert Z7(3) ** -1 == Z7(5)
        assert Z7(3) ** -2 == Z7(5) ** 2

    def test_negative_exponent_without_inverse(self):
        with pytest.raises(ValueError):
            ZMod(6)(2) ** -1

    def test_fermat_little_theorem(self):
        p = 2**127 - 1
        Zp = ZMod(p)
        rng = random.Random(1)
        for _ in range(50):
            a = Zp(rng.randrange(1, p))
            assert a ** (p - 1) == Zp(1)

    def test_matches_builtin_pow(self):
        rng = random.Random(2)
        for _ in range(100):
            n = rng.randrange(2, 10**12)
            a, e = rng.randrange(n), rng.randrange(10**6)
            assert ZMod(n)(a) ** e == ZMod(n)(pow(a, e, n))

    def test_toy_rsa(self):
        # n = 61 * 53, e * d = 1 mod lcm(60, 52)
        n, e, d = 3233, 17, 2753
        Zn = ZMod(n)
        assert Zn(65) ** e == Zn(2790)
        for m in range(0, n, 37):
            assert (Zn(m) ** e) ** d == Zn(m)

    def test_generic_power_works_on_zmod(self):
        # algorithms.power uses only *, so it works on any element.
        assert power(Z7(3), 6) == Z7(1)
        assert power(ZMod(3233)(65), 17) == ZMod(3233)(2790)


# =============================================================================
# Ring laws on random inputs
# =============================================================================


class TestRingLaws:
    @pytest.mark.parametrize("n", [97, 100, 2**61 - 1])
    def test_laws(self, n):
        Zn = ZMod(n)
        rng = random.Random(n)
        for _ in range(200):
            a, b, c = (Zn(rng.randrange(n)) for _ in range(3))
            assert a + b == b + a
            assert a * b == b * a
            assert (a + b) + c == a + (b + c)
            assert (a * b) * c == a * (b * c)
            assert a * (b + c) == a * b + a * c
            assert a - b == a + (-b)
            assert a + Zn(0) == a
            assert a * Zn(1) == a
