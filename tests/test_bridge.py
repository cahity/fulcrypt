"""Tests for as_ring and as_field.

They check the ring and field axioms on element classes by brute force.

The verifiers are O(n^3), so keep sizes small (<= 16 elements).
"""

import functools

import pytest

from fulcrypt.algebra import Field, Ring, as_field, as_ring
from fulcrypt.elements import GF2n, ZMod

GF4 = GF2n(0b111)  # x^2 + x + 1
GF8 = GF2n(0b1011)  # x^3 + x + 1
GF8b = GF2n(0b1101)  # x^3 + x^2 + 1, same size, different polynomial
GF16 = GF2n(0b10011)  # x^4 + x + 1

PRIMES = [2, 3, 5, 7, 11, 13]
FIELDS = [ZMod(p) for p in PRIMES] + [GF4, GF8, GF8b, GF16]
NOT_FIELDS = [4, 6, 8, 9, 10, 12, 15]

# Each verification is O(n^3); build every field once and reuse it.
field = functools.cache(as_field)
ring = functools.cache(as_ring)


def ids(F):
    return F.__name__


# =============================================================================
# Accepts fields
# =============================================================================


class TestAcceptsFields:
    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_returns_field(self, F):
        assert isinstance(as_field(F), Field)

    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_elements(self, F):
        assert field(F).elements == set(F.all())

    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_zero_and_one(self, F):
        f = field(F)
        assert f.zero == F(0)
        assert f.one() == F(1)


# =============================================================================
# Rejects rings that are not fields
# =============================================================================


class TestRejectsNonFields:
    @pytest.mark.parametrize("n", NOT_FIELDS)
    def test_composite_modulus(self, n):
        with pytest.raises(AssertionError, match="Nonzero elements do not form a group"):
            as_field(ZMod(n))


# =============================================================================
# The verifier and the element classes must agree
# =============================================================================
# The verifier finds identities and inverses by searching every element.
# The element classes compute them with xgcd. Two independent methods, same answers.


class TestAgreesWithElements:
    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_add_and_multiply(self, F):
        f = field(F)
        for a in F.all():
            for b in F.all():
                assert f.add(a, b) == a + b
                assert f.multiply(a, b) == a * b

    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_additive_inverse(self, F):
        f = field(F)
        for a in F.all():
            assert f.additive_group.inverse(a) == -a

    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_multiplicative_inverse(self, F):
        f = field(F)
        for a in F.all()[1:]:
            assert f.inverse(a) == a.inverse()

    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_divide(self, F):
        f = field(F)
        for a in F.all():
            for b in F.all()[1:]:
                assert f.divide(a, b) == a / b

    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_divide_by_zero(self, F):
        with pytest.raises(ZeroDivisionError):
            field(F).divide(F(1), F(0))


# =============================================================================
# Structure facts
# =============================================================================


class TestStructure:
    @pytest.mark.parametrize("F", [GF4, GF8, GF8b, GF16, ZMod(2)], ids=ids)
    def test_characteristic_two(self, F):
        # In GF(2^n), every element is its own additive inverse.
        f = field(F)
        for a in F.all():
            assert f.additive_group.inverse(a) == a

    @pytest.mark.parametrize("F", FIELDS, ids=ids)
    def test_multiplicative_group(self, F):
        # The nonzero elements form an abelian group with one element fewer than the field.
        f = field(F)
        assert len(f.mult_group.elements) == len(F.all()) - 1
        assert f.mult_group.is_abelian()
        assert f.mult_group.identity() == F(1)

    def test_same_size_fields_are_separate(self):
        # GF8 and GF8b have the same size but are different classes,
        # so their element sets never overlap.
        assert field(GF8).elements.isdisjoint(field(GF8b).elements)


# =============================================================================
# as_ring
# =============================================================================

RINGS = [ZMod(n) for n in range(2, 13)] + [GF4, GF8]


class TestAsRing:
    @pytest.mark.parametrize("R", RINGS, ids=ids)
    def test_returns_ring(self, R):
        assert isinstance(as_ring(R), Ring)

    @pytest.mark.parametrize("n", NOT_FIELDS)
    def test_ring_but_not_field(self, n):
        # Z/nZ passes the ring checks for every n; the field check is what fails.
        r = ring(ZMod(n))
        assert type(r) is Ring
        assert not isinstance(r, Field)
        with pytest.raises(AssertionError):
            field(ZMod(n))

    @pytest.mark.parametrize("R", RINGS, ids=ids)
    def test_elements_zero_one(self, R):
        r = ring(R)
        assert r.elements == set(R.all())
        assert r.zero == R(0)
        assert r.one() == R(1)

    @pytest.mark.parametrize("R", RINGS, ids=ids)
    def test_agrees_with_operators(self, R):
        r = ring(R)
        for a in R.all():
            assert r.additive_group.inverse(a) == -a
            for b in R.all():
                assert r.add(a, b) == a + b
                assert r.multiply(a, b) == a * b

    def test_zero_divisors_z6(self):
        Z6 = ZMod(6)
        r = ring(Z6)
        pairs = {
            (int(a), int(b))
            for a in Z6.all()[1:]
            for b in Z6.all()[1:]
            if r.multiply(a, b) == r.zero
        }
        assert pairs == {(2, 3), (3, 2), (3, 4), (4, 3)}

    @pytest.mark.parametrize("R", [ZMod(p) for p in PRIMES] + [GF4, GF8], ids=ids)
    def test_fields_have_no_zero_divisors(self, R):
        r = ring(R)
        nonzero = R.all()[1:]
        assert all(r.multiply(a, b) != r.zero for a in nonzero for b in nonzero)

    @pytest.mark.parametrize("n", [6, 8, 9, 12])
    def test_units_match_element_inverse(self, n):
        # A unit has some b with a * b = 1. The brute-force search must agree with a.inverse().
        Zn = ZMod(n)
        r = ring(Zn)
        for a in Zn.all():
            has_partner = any(r.multiply(a, b) == r.one() for b in Zn.all())
            try:
                a.inverse()
                invertible = True
            except ValueError, ZeroDivisionError:
                invertible = False
            assert has_partner == invertible
