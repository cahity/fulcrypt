"""fulcrypt: cryptography primitives implemented from scratch for learning.

Not for protecting real data: the code is not constant-time and has not been audited.

Modules:
- elements: GF(2)[x] polynomials, integers mod n, GF(2^n) fields
- algorithms: generic extended Euclid, modular inverse, fast exponentiation
- algebra: brute-force group, ring and field axiom checks
- aes: the AES block cipher (FIPS-197), 128/192/256-bit keys
- modes: PKCS#7 padding, ECB and CBC modes (NIST SP 800-38A)
"""
