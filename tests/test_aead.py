import pytest

from fulcrypt import aead, kdf, mac, modes

# =============================================================================
# Step 12: Encrypt-then-MAC  (slides 4c p13-14)
# =============================================================================
# Layout: nonce (16) || AES-128-CTR ciphertext || HMAC-SHA256 tag (32) over nonce || ciphertext.
# The nonce is random, so there are no fixed vectors: the tests take the output apart and
# rebuild each piece from the primitives of the earlier steps.

KEY = bytes(range(32))
NONCE_SIZE = 16
TAG_SIZE = 32


def split(data):
    return data[:NONCE_SIZE], data[NONCE_SIZE:-TAG_SIZE], data[-TAG_SIZE:]


def flip(data, i):
    out = bytearray(data)
    out[i] ^= 0x01
    return bytes(out)


class TestEtmDeriveKeys:
    def test_matches_hkdf(self):
        k_enc, k_mac = aead.etm_derive_keys(KEY)
        assert k_enc == kdf.hkdf(KEY, b"", b"encryption", 16)
        assert k_mac == kdf.hkdf(KEY, b"", b"authentication", 32)

    def test_sizes(self):
        k_enc, k_mac = aead.etm_derive_keys(KEY)
        assert type(k_enc) is bytes and type(k_mac) is bytes
        assert len(k_enc) == 16 and len(k_mac) == 32

    def test_keys_are_independent(self):
        # Never use the same key for encryption and authentication.
        k_enc, k_mac = aead.etm_derive_keys(KEY)
        assert k_enc != k_mac[:16]

    def test_different_master_keys(self):
        assert aead.etm_derive_keys(b"key one") != aead.etm_derive_keys(b"key two")


class TestEtmEncrypt:
    def test_layout_length(self):
        for length in (0, 1, 16, 33):
            assert len(aead.etm_encrypt(KEY, b"m" * length)) == NONCE_SIZE + length + TAG_SIZE

    def test_ciphertext_is_ctr_under_nonce(self):
        msg = b"attack at dawn, bring snacks"
        nonce, ct, _ = split(aead.etm_encrypt(KEY, msg))
        k_enc, _ = aead.etm_derive_keys(KEY)
        assert ct == modes.ctr_encrypt(msg, k_enc, nonce)

    def test_tag_is_hmac_of_nonce_and_ciphertext(self):
        # Encrypt-then-MAC: the tag covers what's sent (nonce || ciphertext), not the plaintext.
        nonce, ct, tag = split(aead.etm_encrypt(KEY, b"attack at dawn"))
        _, k_mac = aead.etm_derive_keys(KEY)
        assert tag == mac.hmac_sha256(k_mac, nonce + ct)

    def test_random_nonce(self):
        # Same key, same message: a fresh nonce makes every ciphertext different.
        outputs = {aead.etm_encrypt(KEY, b"same message") for _ in range(5)}
        assert len(outputs) == 5
        assert len({split(o)[0] for o in outputs}) == 5

    def test_ciphertext_hides_plaintext(self):
        msg = b"secret!!" * 4
        _, ct, _ = split(aead.etm_encrypt(KEY, msg))
        assert ct != msg

    def test_returns_bytes(self):
        assert type(aead.etm_encrypt(KEY, b"abc")) is bytes

    def test_accepts_bytearray(self):
        data = aead.etm_encrypt(bytearray(KEY), bytearray(b"abc"))
        assert aead.etm_decrypt(KEY, data) == b"abc"


class TestEtmDecrypt:
    @pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 100, 1000])
    def test_round_trip(self, length):
        msg = bytes((7 * i + 1) % 256 for i in range(length))
        assert aead.etm_decrypt(KEY, aead.etm_encrypt(KEY, msg)) == msg

    @pytest.mark.parametrize("key", [b"short", bytes(16), bytes(range(64))])
    def test_any_master_key_length(self, key):
        # The master key only feeds HKDF, so any length works.
        assert aead.etm_decrypt(key, aead.etm_encrypt(key, b"hello")) == b"hello"

    def test_returns_bytes(self):
        assert type(aead.etm_decrypt(KEY, aead.etm_encrypt(KEY, b"abc"))) is bytes

    def test_accepts_bytearray(self):
        data = aead.etm_encrypt(KEY, b"abc")
        assert aead.etm_decrypt(bytearray(KEY), bytearray(data)) == b"abc"


class TestEtmRejectsTampering:
    MSG = b"amount=100&to=bob"

    def data(self):
        return aead.etm_encrypt(KEY, self.MSG)

    @pytest.mark.parametrize("where", ["nonce", "ciphertext", "tag"])
    @pytest.mark.parametrize("position", ["first", "last"])
    def test_any_flipped_bit(self, where, position):
        data = self.data()
        start = {"nonce": 0, "ciphertext": NONCE_SIZE, "tag": len(data) - TAG_SIZE}[where]
        end = {"nonce": NONCE_SIZE, "ciphertext": len(data) - TAG_SIZE, "tag": len(data)}[where]
        i = start if position == "first" else end - 1
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, flip(data, i))

    def test_ctr_malleability_is_caught(self):
        # CTR alone lets an attacker turn "100" into "900" by XORing the ciphertext.
        # Here the change is detected because the tag no longer matches.
        data = bytearray(self.data())
        i = NONCE_SIZE + self.MSG.index(b"1")
        data[i] ^= ord("1") ^ ord("9")
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, bytes(data))

    def test_wrong_key(self):
        with pytest.raises(ValueError):
            aead.etm_decrypt(b"wrong key", self.data())

    def test_tag_from_another_message(self):
        nonce, ct, _ = split(self.data())
        _, _, other_tag = split(aead.etm_encrypt(KEY, b"amount=999&to=eve"))
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, nonce + ct + other_tag)

    def test_swapped_nonce(self):
        # Re-pairing a ciphertext with another nonce changes the decryption: the tag forbids it.
        _, ct, tag = split(self.data())
        other_nonce, _, _ = split(aead.etm_encrypt(KEY, self.MSG))
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, other_nonce + ct + tag)

    @pytest.mark.parametrize("cut", [1, 16, 32])
    def test_truncated(self, cut):
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, self.data()[:-cut])

    def test_extended(self):
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, self.data() + b"\x00")

    def test_dropped_ciphertext_byte(self):
        data = self.data()
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, data[:NONCE_SIZE] + data[NONCE_SIZE + 1 :])

    @pytest.mark.parametrize("length", [0, 1, 47])
    def test_too_short(self, length):
        # Anything shorter than nonce + tag (48 bytes) can't be valid.
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, bytes(length))

    def test_empty_message_tag_still_checked(self):
        data = aead.etm_encrypt(KEY, b"")
        assert len(data) == 48
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, flip(data, 47))
