import operator
from typing import Callable
from dataclasses import dataclass
from functools import cached_property


@dataclass
class Group:
    """One associative operation, an identity, and an inverse for each element."""
    elements: set
    operation: Callable
    check: bool = True

    def has_closure(self):
        for a in self.elements:
            for b in self.elements:
                if self.operation(a, b) not in self.elements:
                    return False
        return True

    def is_associative(self):
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.operation(self.operation(a, b), c) != self.operation(a, self.operation(b, c)):
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
        return self._identity is not None

    def identity(self):
        assert self.has_identity(), "The group does not have an identity element."
        return self._identity

    def has_inverses(self):
        return self.has_identity() and len(self._inverses) == len(self.elements)

    def inverse(self, a):
        assert a in self._inverses, f"{a} has no inverse in the group."
        return self._inverses[a]

    def is_abelian(self):
        for a in self.elements:
            for b in self.elements:
                if self.operation(a, b) != self.operation(b, a):
                    return False

        return True

    def is_group(self):
        return (
            self.has_closure()
            and self.is_associative()
            and self.has_identity()
            and self.has_inverses()
        )

    def is_abelian_group(self):
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
        return self.additive_group.operation(a, b)

    @property
    def elements(self):
        return self.additive_group.elements

    @property
    def zero(self):
        return self.additive_group.identity()

    def has_closure(self):
        for a in self.elements:
            for b in self.elements:
                if self.multiply(a, b) not in self.elements:
                    return False
        return True

    def is_associative(self):
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.multiply(self.multiply(a, b), c) != self.multiply(a, self.multiply(b, c)):
                        return False

        return True

    @cached_property
    def _one(self):
        for e in self.elements:
            if all(self.multiply(e, a) == a and self.multiply(a, e) == a for a in self.elements):
                return e
        return None

    def has_one(self):
        return self._one is not None

    def one(self):
        assert self.has_one(), "The ring does not have an identity element."
        return self._one

    def is_distributive(self):
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    if self.multiply(a, self.add(b, c)) != self.add(self.multiply(a, b), self.multiply(a, c)):
                        return False
                    if self.multiply(self.add(b, c), a) != self.add(self.multiply(b, a), self.multiply(c, a)):
                        return False

        return True

    def is_ring(self):
        return (
            self.has_closure()
            and self.additive_group.is_abelian()  # group axioms checked by Group itself
            and self.has_one()
            and self.is_associative()
            and self.is_distributive()
        )

    def __post_init__(self):
        if self.check:
            assert self.is_ring(), "The provided additive group and multiplication do not form a ring."


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
        return self.mult_group.inverse(a)

    def divide(self, a, b):
        if b == self.zero:
            raise ZeroDivisionError("Zero has no multiplicative inverse")
        return self.multiply(a, self.inverse(b))


def as_field(cls) -> Field:
    elems = set(cls.all())
    return Field(Group(elems, operator.add), operator.mul)

def as_ring(cls) -> Ring:
    elems = set(cls.all())
    return Ring(Group(elems, operator.add), operator.mul)

