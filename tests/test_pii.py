from guardrails.checks.pii import PIIScanner, _luhn_valid, _redact_text

pii = PIIScanner()


def test_email_redacted():
    text, counts = _redact_text("mail me at john@gmail.com")
    assert counts == {"email": 1}
    assert "[REDACTED_EMAIL]" in text
    assert "john@gmail.com" not in text


def test_phone_with_91():
    text, counts = _redact_text("call +91 9876543210 now")
    assert counts == {"phone": 1}
    assert "[REDACTED_PHONE]" in text


def test_valid_card_luhn():
    # 4111111111111111 is a classic Luhn-valid test number
    text, counts = _redact_text("card: 4111 1111 1111 1111")
    assert counts == {"card": 1}
    assert "[REDACTED_CARD]" in text


def test_non_luhn_digits_not_card():
    # 15 random digits, Luhn fail -> card NAHI banna chahiye
    text, counts = _redact_text("order id 123456789012345 confirmed")
    assert counts.get("card", 0) == 0


def test_pan_pattern():
    text, counts = _redact_text("PAN is ABCDE1234F")
    assert counts == {"pan": 1}
    assert "[REDACTED_ID]" in text


def test_overlap_card_wins_over_phone():
    # Luhn-valid 16 digits, phone/aadhaar ko andar match nahi karna chahiye
    text, counts = _redact_text("pay via 4111111111111111 today")
    assert counts == {"card": 1}
    assert counts.get("phone", 0) == 0


def test_clean_text_untouched():
    text, counts = _redact_text("totally normal sentence, amount 500")
    assert counts == {}
    assert text == "totally normal sentence, amount 500"


def test_parsed_redaction_keeps_json_valid():
    import json
    r = pii.check("", context={"parsed": {"customer": "mail: a@b.com", "amount": 5}})
    out = json.loads(r.transformed_output)  # ye crash nahi hona chahiye
    assert out["amount"] == 5
    assert "[REDACTED_EMAIL]" in out["customer"]


def test_no_pii_values_in_metadata():
    r = pii.check("", context={"parsed": {"c": "9876543210 and x@y.com"}})
    dumped = str(r.metadata)
    assert "9876543210" not in dumped
    assert "x@y.com" not in dumped