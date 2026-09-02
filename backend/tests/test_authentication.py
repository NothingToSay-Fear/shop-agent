from app.services.authentication import hash_password, token_digest, verify_password


def test_password_hash_uses_a_random_salt_and_can_be_verified() -> None:
    first_hash = hash_password("correct-horse-battery-staple")
    second_hash = hash_password("correct-horse-battery-staple")

    assert first_hash != second_hash
    assert verify_password("correct-horse-battery-staple", first_hash) is True
    assert verify_password("wrong-password", first_hash) is False


def test_invalid_or_legacy_password_hash_cannot_be_verified() -> None:
    assert verify_password("any-password", "disabled") is False


def test_token_digest_is_stable_but_does_not_return_the_raw_token() -> None:
    token = "a-local-browser-token"

    assert token_digest(token) == token_digest(token)
    assert token_digest(token) != token
