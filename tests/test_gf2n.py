"""Tests for GF2n(modulus): a class factory for GF(2^n) = GF(2)[x] / (modulus)."""

import dataclasses
import random

import pytest

from algorithms import power
from elements import GF2n, GF2Poly, ZMod

AES = GF2n(0x11B)  # x^8 + x^4 + x^3 + x + 1
GF8 = GF2n(0b1011)  # x^3 + x + 1
GF16 = GF2n(0b10011)  # x^4 + x + 1


# =============================================================================
# The factory
# =============================================================================


class TestFactory:
    def test_cached(self):
        assert GF2n(0x11B) is GF2n(0x11B)
        assert GF2n(0x11B)(0x57) == GF2n(0x11B)(0x57)

    def test_different_moduli_are_different_classes(self):
        # Same size, different irreducible polynomial -> different classes.
        assert GF2n(0b1011) is not GF2n(0b1101)

    @pytest.mark.parametrize(
        "modulus",
        [
            0,
            1,  # not polynomials of degree >= 1
            0b100,  # x^2
            0b101,  # (x + 1)^2
            0b1001,  # (x + 1)(x^2 + x + 1)
            0x11A,  # divisible by x
        ],
    )
    def test_rejects_reducible_modulus(self, modulus):
        with pytest.raises(ValueError):
            GF2n(modulus)

    @pytest.mark.parametrize(
        "modulus, name", [(0x11B, "GF(2^8)"), (0b1011, "GF(2^3)"), (0b10011, "GF(2^4)")]
    )
    def test_name_uses_degree(self, modulus, name):
        assert GF2n(modulus).__name__ == name

    def test_modulus_attribute(self):
        assert AES.modulus == GF2Poly(0x11B)
        assert AES(0x57).modulus == GF2Poly(0x11B)

    @pytest.mark.parametrize("F, size", [(GF8, 8), (GF16, 16), (AES, 256)])
    def test_all(self, F, size):
        elems = F.all()
        assert len(elems) == size
        assert len(set(elems)) == size
        assert elems[0] == F(0)


# =============================================================================
# Elements: construction, equality, hashing
# =============================================================================


class TestElement:
    @pytest.mark.parametrize(
        "raw, reduced",
        [
            (0x57, 0x57),
            (0x11B, 0x00),  # the modulus itself reduces to 0
            (0x100, 0x1B),  # x^8 = x^4 + x^3 + x + 1
            (0x200, 0x36),  # x^9 = x * x^8
        ],
    )
    def test_reduces_on_construction(self, raw, reduced):
        assert AES(raw).bits == reduced

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            AES(-1)

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            AES(3).bits = 4

    def test_hash(self):
        assert len({AES(0x57), AES(0x57), AES(0x57 ^ 0x11B), AES(0x83)}) == 2

    def test_not_equal_to_int(self):
        assert AES(1) != 1

    def test_not_equal_across_fields(self):
        assert GF2n(0b1011)(1) != GF2n(0b1101)(1)

    def test_bool(self):
        assert not AES(0)
        assert not AES(0x11B)
        assert AES(1)

    @pytest.mark.parametrize("bits, text", [(0x57, "0x57"), (0x05, "0x05"), (0x00, "0x00")])
    def test_repr(self, bits, text):
        assert repr(AES(bits)) == text

    def test_repr_in_containers(self):
        # Lists use repr, so they stay short.
        assert repr([AES(1), AES(0xCA)]) == "[0x01, 0xca]"

    @pytest.mark.parametrize(
        "bits, text",
        [
            (0x57, "x^6 + x^4 + x^2 + x + 1 (mod x^8 + x^4 + x^3 + x + 1)"),
            (0x01, "1 (mod x^8 + x^4 + x^3 + x + 1)"),
            (0x00, "0 (mod x^8 + x^4 + x^3 + x + 1)"),
        ],
    )
    def test_str(self, bits, text):
        assert str(AES(bits)) == text
        assert f"{AES(bits)}" == text


# =============================================================================
# Arithmetic
# =============================================================================


class TestArithmetic:
    def test_add_is_xor(self):
        assert AES(0x57) + AES(0x83) == AES(0xD4)

    def test_add_self_is_zero(self):
        assert AES(0x57) + AES(0x57) == AES(0)

    def test_sub_equals_add(self):
        assert AES(0x57) - AES(0x83) == AES(0x57) + AES(0x83)

    def test_neg_is_identity(self):
        assert -AES(0x57) == AES(0x57)

    @pytest.mark.parametrize(
        "a, b, product",
        [
            (0x57, 0x83, 0xC1),  # FIPS-197, section 4.2
            (0x57, 0x13, 0xFE),  # FIPS-197, section 4.2.1
            (0x57, 0x02, 0xAE),  # xtime chain from FIPS-197
            (0x57, 0x04, 0x47),
            (0x57, 0x08, 0x8E),
            (0x57, 0x10, 0x07),
            (0x57, 0x01, 0x57),
            (0x57, 0x00, 0x00),
        ],
    )
    def test_mul_fips197(self, a, b, product):
        assert AES(a) * AES(b) == AES(product)
        assert AES(b) * AES(a) == AES(product)

    def test_results_keep_class(self):
        a, b = AES(0x57), AES(0x83)
        for result in (a + b, a - b, a * b, -a, a / b, a**3, a.inverse()):
            assert type(result) is AES

    @pytest.mark.parametrize(
        "op",
        [
            lambda: AES(1) + 1,  # ints are ambiguous here: no coercion
            lambda: 1 + AES(1),
            lambda: AES(1) - 1,
            lambda: AES(3) * 2,
            lambda: 2 * AES(3),
            lambda: AES(3) / 1,
            lambda: 1 / AES(3),
            lambda: AES(1) + GF2Poly(1),
            lambda: AES(1) + ZMod(2)(1),
            lambda: GF8(1) + GF16(1),
            lambda: GF8(1) * GF16(1),
            lambda: AES(3) ** AES(2),
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
        assert AES(0x53).inverse() == AES(0xCA)  # FIPS-197, section 5.1.1
        assert AES(0xCA).inverse() == AES(0x53)
        assert AES(1).inverse() == AES(1)

    @pytest.mark.parametrize("F", [GF8, GF16, AES])
    def test_every_nonzero_invertible(self, F):
        for a in F.all()[1:]:
            assert a * a.inverse() == F(1)

    def test_zero_has_no_inverse(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            AES(0).inverse()
        with pytest.raises((ValueError, ZeroDivisionError)):
            AES(0x57) / AES(0)

    def test_div(self):
        a, b = AES(0x57), AES(0x83)
        assert (a / b) * b == a
        assert a / a == AES(1)
        assert AES(1) / AES(0x53) == AES(0xCA)


# =============================================================================
# Powers
# =============================================================================


class TestPower:
    def test_zero_and_one(self):
        assert AES(0x57) ** 0 == AES(1)
        assert AES(0) ** 0 == AES(1)
        assert AES(0x57) ** 1 == AES(0x57)

    def test_x_powers_reduce(self):
        assert AES(0x02) ** 8 == AES(0x1B)
        assert AES(0x02) ** 9 == AES(0x36)

    def test_negative_exponent(self):
        assert AES(0x53) ** -1 == AES(0xCA)
        assert AES(0x53) ** -2 == AES(0xCA) ** 2

    def test_negative_exponent_of_zero(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            AES(0) ** -1

    @pytest.mark.parametrize("F, n", [(GF8, 3), (GF16, 4), (AES, 8)])
    def test_lagrange(self, F, n):
        # The nonzero elements form a group of order 2^n - 1, so a^(2^n - 1) = 1
        # and a^(2^n) = a for every a, including 0.
        for a in F.all():
            if a:
                assert a ** ((1 << n) - 1) == F(1)
            assert a ** (1 << n) == a

    def test_generator(self):
        # 0x03 generates the multiplicative group of the AES field.
        seen = {AES(0x03) ** k for k in range(255)}
        assert len(seen) == 255

    def test_frobenius(self):
        rng = random.Random(1)
        for _ in range(200):
            a, b = AES(rng.randrange(256)), AES(rng.randrange(256))
            assert (a + b) ** 2 == a**2 + b**2

    def test_generic_power_works_on_gf2n(self):
        assert power(AES(0x02), 8) == AES(0x1B)


# =============================================================================
# Field laws
# =============================================================================


class TestFieldLaws:
    def test_gf8_exhaustive(self):
        elems = GF8.all()
        for a in elems:
            for b in elems:
                assert a + b == b + a
                assert a * b == b * a
                for c in elems:
                    assert (a * b) * c == a * (b * c)
                    assert a * (b + c) == a * b + a * c

    def test_aes_random(self):
        rng = random.Random(2)
        for _ in range(500):
            a, b, c = (AES(rng.randrange(256)) for _ in range(3))
            assert (a + b) + c == a + (b + c)
            assert (a * b) * c == a * (b * c)
            assert a * (b + c) == a * b + a * c
            assert a * AES(1) == a
            assert a + AES(0) == a


# =============================================================================
# Application: the AES S-box (inverse in GF(2^8), then an affine map over GF(2))
# =============================================================================


def sbox(byte):
    b = AES(byte).inverse().bits if byte else 0
    out = 0
    for i in range(8):
        bit = (
            (b >> i)
            ^ (b >> ((i + 4) % 8))
            ^ (b >> ((i + 5) % 8))
            ^ (b >> ((i + 6) % 8))
            ^ (b >> ((i + 7) % 8))
            ^ (0x63 >> i)
        )
        out |= (bit & 1) << i
    return out


class TestSbox:
    @pytest.mark.parametrize(
        "byte, expected",
        [
            (0x00, 0x63),
            (0x01, 0x7C),
            (0x53, 0xED),
            (0x10, 0xCA),
            (0xFF, 0x16),
            (0xC9, 0xDD),
        ],
    )
    def test_known_entries(self, byte, expected):
        assert sbox(byte) == expected

    def test_is_permutation(self):
        assert sorted(sbox(b) for b in range(256)) == list(range(256))
