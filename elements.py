"""Element classes: GF(2)[x] polynomials, integers mod n, and GF(2^n) fields."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from algorithms import inverse_mod, power


@dataclass(frozen=True)
class GF2Poly:
    """A polynomial over GF(2), stored as a bit vector (0b1011 = x^3 + x + 1)."""

    bits: int

    def __post_init__(self):
        if self.bits < 0:
            raise ValueError("Binary polynomials must be nonnegative in GF(2)")

    @property
    def degree(self):
        """Return the degree; the zero polynomial has degree -1."""
        return self.bits.bit_length() - 1

    def __divmod__(self, other: GF2Poly) -> tuple[GF2Poly, GF2Poly]:
        """Return the quotient and remainder from polynomial long division over F_2."""
        if not isinstance(other, GF2Poly):
            return NotImplemented
        if other == GF2Poly(0):
            raise ZeroDivisionError("Cannot divide by the zero polynomial")

        a, b = self.bits, other.bits
        quotient = 0
        while a and a.bit_length() >= b.bit_length():
            shift = a.bit_length() - b.bit_length()
            quotient ^= 1 << shift
            a ^= b << shift  # Subtract the multiple that cancels the leading term.
        return GF2Poly(quotient), GF2Poly(a)

    def is_irreducible(self) -> bool:
        """Check whether the polynomial is irreducible over GF(2).

        A reducible polynomial has a factor of degree at most half its degree.
        """
        if self.bits < 2:
            return False

        for factor in range(2, (1 << (self.degree // 2 + 1))):
            if self % GF2Poly(factor) == GF2Poly(0):
                return False
        return True

    def __repr__(self) -> str:
        """Write a binary polynomial in descending powers of x."""
        terms = []
        for degree in range(self.degree, -1, -1):
            if self.bits & (1 << degree):
                if degree == 0:
                    terms.append("1")
                elif degree == 1:
                    terms.append("x")
                else:
                    terms.append(f"x^{degree}")
        return " + ".join(terms) or "0"

    def __mul__(self, other: GF2Poly) -> GF2Poly:
        """Multiply over F_2. Adding terms uses XOR, so there are no carries."""
        if not isinstance(other, GF2Poly):
            return NotImplemented
        a, b = self.bits, other.bits
        product = 0
        while b:
            if b & 1:
                product ^= a
            a <<= 1
            b >>= 1
        return GF2Poly(product)

    def __mod__(self, other):
        if not isinstance(other, GF2Poly):
            return NotImplemented
        return divmod(self, other)[1]

    def __floordiv__(self, other):
        if not isinstance(other, GF2Poly):
            return NotImplemented
        return divmod(self, other)[0]

    def __pow__(self, other: int) -> GF2Poly:
        """Exponentiation by squaring."""
        if not isinstance(other, int):
            return NotImplemented
        if other < 0:
            raise ValueError("Negative exponents are not supported for GF(2) polynomials")
        return power(self, other)

    def __bool__(self):
        return self.bits != 0

    def __add__(self, other):
        if not isinstance(other, GF2Poly):
            return NotImplemented
        return GF2Poly(self.bits ^ other.bits)

    def __neg__(self):
        return self

    def __sub__(self, other):
        if not isinstance(other, GF2Poly):
            return NotImplemented
        return self + other


@functools.cache
def ZMod(n: int) -> type[Any]:
    """Return the element class of the integers modulo n (cached per n)."""
    if n < 2:
        raise ValueError(f"Zn with n={n} can not be constructed. n must be at least 2.")

    @dataclass(frozen=True)
    class Zn:
        value: int
        modulus = n  # No type annotation to not make it a dataclass field.

        def __post_init__(self):
            object.__setattr__(self, "value", self.value % n)

        def _coerce(self, other):
            if isinstance(other, type(self)):
                return other
            if isinstance(other, int) and not isinstance(other, bool):
                return type(self)(other)
            return None

        def __add__(self, other):
            o = self._coerce(other)
            if o is None:
                return NotImplemented
            return type(self)(self.value + o.value)

        def __radd__(self, other):
            return self.__add__(other)

        def __neg__(self):
            return type(self)(-self.value)

        def __sub__(self, other):
            o = self._coerce(other)
            if o is None:
                return NotImplemented
            return type(self)(self.value - o.value)

        def __rsub__(self, other):
            o = self._coerce(other)
            if o is None:
                return NotImplemented
            return type(self)(o.value - self.value)

        def __mul__(self, other):
            o = self._coerce(other)
            if o is None:
                return NotImplemented
            return type(self)(self.value * o.value)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __rtruediv__(self, other):
            o = self._coerce(other)
            if o is None:
                return NotImplemented
            return o * self.inverse()

        def __bool__(self):
            return self.value != 0

        def __int__(self):
            return self.value

        def __repr__(self):
            return f"{self.value} (mod {self.modulus})"

        def inverse(self):
            """Return the multiplicative inverse; ValueError if none exists."""
            return type(self)(inverse_mod(self.value, self.modulus))

        def __truediv__(self, other):
            o = self._coerce(other)
            if o is None:
                return NotImplemented
            return self * o.inverse()

        def __pow__(self, other):
            if not isinstance(other, int):
                return NotImplemented
            if other < 0:
                return self.inverse() ** -other
            return power(self, other)

        @classmethod
        def all(cls):
            """Return all elements of Z/nZ."""
            return [cls(i) for i in range(cls.modulus)]

    Zn.__name__ = Zn.__qualname__ = f"Z{n}"
    return Zn


@functools.cache
def GF2n(modulus: int) -> type[Any]:
    """Return the element class of GF(2)[x] modulo an irreducible polynomial (cached)."""
    m = GF2Poly(modulus)
    if not m.is_irreducible():
        raise ValueError(f"GF(2^{m.degree}) can not be constructed. {m} is reducible.")

    @dataclass(frozen=True)
    class F:
        bits: int
        modulus = m  # No type annotation to not make it a dataclass field.

        def __post_init__(self):
            object.__setattr__(self, "bits", (GF2Poly(self.bits) % m).bits)

        def __neg__(self):
            return self

        def __add__(self, other):
            if type(other) is not F:
                return NotImplemented
            return type(self)(self.bits ^ other.bits)

        def __sub__(self, other):
            if type(other) is not F:
                return NotImplemented
            return self + other

        def __mul__(self, other):
            if type(other) is not F:
                return NotImplemented
            return type(self)((GF2Poly(self.bits) * GF2Poly(other.bits)).bits)

        def inverse(self):
            """Return the multiplicative inverse; ValueError for zero."""
            return type(self)(inverse_mod(GF2Poly(self.bits), m).bits)

        def __truediv__(self, other):
            if type(other) is not F:
                return NotImplemented
            return self * other.inverse()

        def __pow__(self, other):
            if not isinstance(other, int):
                return NotImplemented
            if other < 0:
                return self.inverse() ** -other
            return power(self, other)

        def __bool__(self):
            return self.bits != 0

        def __repr__(self):
            return f"{self.bits:#04x}"

        def __str__(self):
            return f"{GF2Poly(self.bits)} (mod {self.modulus})"

        @classmethod
        def all(cls):
            """Return all elements of GF(2^n)."""
            return [cls(i) for i in range(1 << cls.modulus.degree)]

    F.__name__ = F.__qualname__ = f"GF(2^{m.degree})"
    return F
