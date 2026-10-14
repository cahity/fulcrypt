"""Brute-force checks of the group, ring and field axioms on small finite structures."""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property


@dataclass
class Group:
    """One associative operation, an identity, and an inverse for each element."""

    elements: set
    operation: Callable
    check: bool = True

    def has_closure(self):
        """Check that the operation never leaves the element set."""
        for a in self.elements:
            for b in self.elements:
                if self.operation(a, b) not in self.elements:
                    return False
        return True

    def is_associative(self):
        """Check (a * b) * c == a * (b * c) for all elements."""
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.operation(self.operation(a, b), c) != self.operation(
                        a, self.operation(b, c)
                    ):
                        return False

        return True

    @cached_property
    def _identity(self):
        for e in self.elements:
            if all(self.operation(e, a) == a and self.operation(a, e) == a for a in self.elements):
                return e
        return None

    @cached_property
    def _inverses(self):
        # Maps each element to its inverse; elements without one are left out.
        e = self._identity
        inverses = {}
        for a in self.elements:
            for b in self.elements:
                if self.operation(a, b) == e and self.operation(b, a) == e:
                    inverses[a] = b
                    break
        return inverses

    def has_identity(self):
        """Check whether an identity element exists."""
        return self._identity is not None

    def identity(self):
        """Return the identity element."""
        assert self.has_identity(), "The group does not have an identity element."
        return self._identity

    def has_inverses(self):
        """Check that every element has an inverse."""
        return self.has_identity() and len(self._inverses) == len(self.elements)

    def inverse(self, a):
        """Return the inverse of a."""
        assert a in self._inverses, f"{a} has no inverse in the group."
        return self._inverses[a]

    def is_abelian(self):
        """Check that the operation is commutative."""
        for a in self.elements:
            for b in self.elements:
                if self.operation(a, b) != self.operation(b, a):
                    return False

        return True

    def is_group(self):
        """Check all group axioms."""
        return (
            self.has_closure()
            and self.is_associative()
            and self.has_identity()
            and self.has_inverses()
        )

    def is_abelian_group(self):
        """Check the group axioms and commutativity."""
        return self.is_group() and self.is_abelian()

    def __post_init__(self):
        if self.check:
            assert self.is_group(), "The provided elements and operation do not form a group."


@dataclass
class Ring:
    """A ring is an abelian additive group with associative, distributive multiplication."""

    additive_group: Group
    multiply: Callable
    check: bool = True

    def add(self, a, b):
        """Add two elements with the additive group operation."""
        return self.additive_group.operation(a, b)

    @property
    def elements(self):
        """Return the element set."""
        return self.additive_group.elements

    @property
    def zero(self):
        """Return the additive identity."""
        return self.additive_group.identity()

    def has_closure(self):
        """Check that multiplication never leaves the element set."""
        for a in self.elements:
            for b in self.elements:
                if self.multiply(a, b) not in self.elements:
                    return False
        return True

    def is_associative(self):
        """Check that multiplication is associative."""
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.multiply(self.multiply(a, b), c) != self.multiply(
                        a, self.multiply(b, c)
                    ):
                        return False

        return True

    @cached_property
    def _one(self):
        for e in self.elements:
            if all(self.multiply(e, a) == a and self.multiply(a, e) == a for a in self.elements):
                return e
        return None

    def has_one(self):
        """Check whether a multiplicative identity exists."""
        return self._one is not None

    def one(self):
        """Return the multiplicative identity."""
        assert self.has_one(), "The ring does not have an identity element."
        return self._one

    def is_distributive(self):
        """Check that multiplication distributes over addition on both sides."""
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.multiply(a, self.add(b, c)) != self.add(
                        self.multiply(a, b), self.multiply(a, c)
                    ):
                        return False
                    if self.multiply(self.add(b, c), a) != self.add(
                        self.multiply(b, a), self.multiply(c, a)
                    ):
                        return False

        return True

    def is_ring(self):
        """Check all ring axioms."""
        return (
            self.has_closure()
            and self.additive_group.is_abelian()  # group axioms checked by Group itself
            and self.has_one()
            and self.is_associative()
            and self.is_distributive()
        )

    def __post_init__(self):
        if self.check:
            assert self.is_ring(), (
                "The provided additive group and multiplication do not form a ring."
            )


@dataclass
class Field(Ring):
    """A field is a ring in which every nonzero element has a multiplicative inverse."""

    def __post_init__(self):
        super().__post_init__()
        self.mult_group = Group(self.elements - {self.zero}, self.multiply, check=False)
        if self.check:
            # Associativity is already covered by the ring checks.
            assert self.mult_group.has_closure() and self.mult_group.has_inverses(), (
                "Nonzero elements do not form a group under multiplication "
                "(zero divisors or missing inverses)."
            )
            assert self.mult_group.is_abelian(), "Multiplication is not commutative."

    def inverse(self, a):
        """Return the multiplicative inverse of a nonzero element."""
        return self.mult_group.inverse(a)

    def divide(self, a, b):
        """Return a / b, raising ZeroDivisionError for b = 0."""
        if b == self.zero:
            raise ZeroDivisionError("Zero has no multiplicative inverse")
        return self.multiply(a, self.inverse(b))


def as_field(cls) -> Field:
    """Build a checked Field from an element class with an all() method."""
    elems = set(cls.all())
    return Field(Group(elems, operator.add), operator.mul)


def as_ring(cls) -> Ring:
    """Build a checked Ring from an element class with an all() method."""
    elems = set(cls.all())
    return Ring(Group(elems, operator.add), operator.mul)
