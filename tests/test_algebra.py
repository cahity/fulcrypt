import pytest

from algebra import Field, Group, Ring


def Zn_add(n):
    return Group(set(range(n)), lambda a, b: (a + b) % n)


def Zn_mul(n):
    return lambda a, b: (a * b) % n


# Group

def test_group_identity():
    assert Zn_add(5).identity() == 0


def test_group_inverses():
    g = Zn_add(5)
    assert {a: g.inverse(a) for a in range(5)} == {0: 0, 1: 4, 2: 3, 3: 2, 4: 1}


def test_group_is_abelian():
    assert Zn_add(5).is_abelian()


def test_group_rejects_non_closed():
    with pytest.raises(AssertionError):
        Group({0, 1, 2}, lambda a, b: (a + b) % 4)


def test_group_rejects_non_associative():
    # Subtraction mod 3 is closed but not associative
    with pytest.raises(AssertionError):
        Group(set(range(3)), lambda a, b: (a - b) % 3)


def test_group_rejects_missing_inverses():
    # Z5 under multiplication: 0 has no inverse
    with pytest.raises(AssertionError):
        Group(set(range(5)), Zn_mul(5))


def test_group_inverse_of_non_element():
    with pytest.raises(AssertionError):
        Zn_add(5).inverse(7)


def test_group_check_false_skips_validation():
    g = Group({0, 1, 2}, lambda a, b: (a + b) % 4, check=False)
    assert g.elements == {0, 1, 2}


# Ring

def test_ring_zero_and_one():
    r = Ring(Zn_add(6), Zn_mul(6))
    assert r.zero == 0
    assert r.one() == 1


def test_ring_add():
    assert Ring(Zn_add(6), Zn_mul(6)).add(4, 5) == 3


def test_ring_rejects_non_closed_multiplication():
    with pytest.raises(AssertionError):
        Ring(Zn_add(4), Zn_mul(5))


def test_ring_rejects_missing_one():
    # Zero multiplication has no multiplicative identity
    with pytest.raises(AssertionError):
        Ring(Zn_add(4), lambda a, b: 0)


def test_ring_check_false_skips_validation():
    r = Ring(Zn_add(4), lambda a, b: 0, check=False)
    with pytest.raises(AssertionError):
        r.one()


# Field

@pytest.mark.parametrize("p", [2, 3, 5, 7, 31])
def test_field_accepts_prime(p):
    f = Field(Zn_add(p), Zn_mul(p))
    for a in range(1, p):
        assert (a * f.inverse(a)) % p == 1


def test_field_inverses_Z7():
    f = Field(Zn_add(7), Zn_mul(7))
    assert {a: f.inverse(a) for a in range(1, 7)} == {1: 1, 2: 4, 3: 5, 4: 2, 5: 3, 6: 6}


@pytest.mark.parametrize("n", [4, 6, 8, 9])
def test_field_rejects_composite(n):
    with pytest.raises(AssertionError, match="Nonzero elements do not form a group"):
        Field(Zn_add(n), Zn_mul(n))


def test_field_rejects_trivial_ring():
    with pytest.raises(AssertionError, match="Nonzero elements do not form a group"):
        Field(Group({0}, lambda a, b: 0), lambda a, b: 0)


def test_field_inverse_of_zero():
    f = Field(Zn_add(5), Zn_mul(5))
    with pytest.raises(AssertionError):
        f.inverse(0)
