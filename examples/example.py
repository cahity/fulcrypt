"""Printed walk-throughs of groups, rings and fields built from the element classes."""

from fulcrypt import algebra
from fulcrypt.elements import GF2n, GF2Poly, ZMod

Z2, Z6, Z7 = ZMod(2), ZMod(6), ZMod(7)


def ex1():
    """Show that Z6 under addition is a group."""
    group = algebra.as_ring(Z6).additive_group

    print("1. Addition forms a group (modulo 6)")
    print("Elements:", Z6.all())
    print("Identity:", group.identity())
    print("4 + 5 =", group.operation(Z6(4), Z6(5)))
    print("Additive inverse of 2:", group.inverse(Z6(2)))
    print("2 + its inverse =", group.operation(Z6(2), group.inverse(Z6(2))))
    print("Abelian:", group.is_abelian())


def ex2():
    """Show that Z6 is a ring but not a field."""
    ring = algebra.as_ring(Z6)

    print("2. A ring need not be a field (modulo 6)")
    print("Elements:", Z6.all())
    print("An inverse of 2 would satisfy 2 * b = 1 (mod 6).")
    print("Try every possible b:")

    for b in Z6.all():
        print(f"2 * {int(b)} mod 6 =", ring.multiply(Z6(2), b))

    print("No result is 1, so 2 has no multiplicative inverse.")
    print("A field requires an inverse for every nonzero element.")
    print("This is a ring, but it is not a field.")


def ex3():
    """Show that Z7 is a field."""
    field = algebra.as_field(Z7)

    print(
        "3. A field is a ring in which every nonzero element has a multiplicative inverse"
        " (modulo 7)"
    )
    print("Elements:", Z7.all())
    print("An inverse of 2 would satisfy 2 * b = 1 (mod 7).")
    print("Try every possible b:")

    for b in Z7.all():
        print(f"2 * {int(b)} mod 7 =", field.multiply(Z7(2), b))

    print("The result is 1 when b = 4, so 4 is the multiplicative inverse of 2.")
    print("A field requires an inverse for every nonzero element.")
    print("This is a field.")


def ex4():
    """Show division as multiplication by an inverse in Z7."""
    field = algebra.as_field(Z7)
    inverse = field.inverse(Z7(2))
    quotient = field.divide(Z7(3), Z7(2))

    print("4. Division is multiplication by an inverse (modulo 7)")
    print("Multiplicative inverse of 2:", inverse)
    print("3 * the inverse of 2 =", field.multiply(Z7(3), inverse))
    print("3 / 2 =", quotient)
    print("Check: 2 * the quotient =", field.multiply(Z7(2), quotient))

    try:
        field.divide(Z7(3), Z7(0))
    except ZeroDivisionError as error:
        print("3 / 0:", error)


def ex5():
    """Show that bits form the field Z2 (XOR and AND)."""
    field = algebra.as_field(Z2)

    print("5. Bits form a field (modulo 2)")
    print("a b | a + b | a * b")
    for a in Z2.all():
        for b in Z2.all():
            print(f"{int(a)} {int(b)} |   {int(field.add(a, b))}   |   {int(field.multiply(a, b))}")
    print("Addition is XOR; multiplication is AND.")
    print("1 + 1 =", field.add(Z2(1), Z2(1)))
    print("But zero and one are distinct:", field.zero != field.one())


def ex6():
    """Show arithmetic in GF(2^3) built from x^3 + x + 1."""
    print("ex6")
    modulus = GF2Poly(0b1011)  # x^3 + x + 1
    F = GF2n(modulus.bits)
    field = algebra.as_field(F)

    a = F(0b101)  # x^2 + 1
    b = F(0b111)  # x^2 + x + 1

    print("6. GF(2^3): Binary polynomials modulo x^3 + x + 1")
    print("f is irreducible:", modulus.is_irreducible())
    print("a:", a)
    print("b:", b)
    print("Modulus:", modulus)

    print("a is irreducible:", GF2Poly(a.bits).is_irreducible())

    print("a + b =", field.add(a, b))
    print("a * b =", field.multiply(a, b))  # already reduced mod f

    inverse = field.inverse(a)
    print("Inverse of a mod f:", inverse)
    print("a * its inverse =", field.multiply(a, inverse))


if __name__ == "__main__":
    ex1()
    print()
    ex2()
    print()
    ex3()
    print()
    ex4()
    print()
    ex5()
    print()
    ex6()
    print()
