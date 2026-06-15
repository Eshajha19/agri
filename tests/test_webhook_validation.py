"""
tests/test_webhook_validation.py

Unit tests for the Celery task input validation layer (issue #2360).

Coverage:
  - Happy-path for text, image, and button message types
  - Malformed / missing required fields → WebhookValidationError
  - Oversized fields → WebhookValidationError
  - Invalid sender numbers → WebhookValidationError
  - Celery task discards bad payloads without calling any service function
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from webhook_validator import (
    MAX_BODY_BYTES,
    MAX_CAPTION_LEN,
    MAX_MESSAGE_TEXT_LEN,
    MAX_SENDER_NUMBER_LEN,
    NormalizedMessage,
    WebhookValidationError,
    validate_and_parse,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_SENDER = "+919876543210"


def _make_payload(
    msg_type: str = "text",
    body: str = "Hello farmer!",
    media_id: str = "mid_001",
    caption: str = "crop photo",
    button_payload: str = "YES",
    button_text: str = "Yes",
    msg_id: str = "wamid.abc123",
) -> dict:
    """Build a minimal valid WhatsApp Cloud API envelope."""
    if msg_type == "text":
        message = {"id": msg_id, "type": "text", "text": {"body": body}}
    elif msg_type in ("image", "audio", "video", "document", "sticker"):
        message = {
            "id": msg_id,
            "type": msg_type,
            msg_type: {"id": media_id, "caption": caption},
        }
    elif msg_type == "button":
        message = {
            "id": msg_id,
            "type": "button",
            "button": {"payload": button_payload, "text": button_text},
        }
    else:
        message = {"id": msg_id, "type": msg_type}

    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [message],
                        }
                    }
                ]
            }
        ],
    }


def _raw(payload: dict) -> str:
    return json.dumps(payload)


# ── Happy-path tests ──────────────────────────────────────────────────────────

class TestHappyPath:
    def test_text_message(self):
        result = validate_and_parse(_raw(_make_payload("text", body="Hi")), VALID_SENDER)
        assert isinstance(result, NormalizedMessage)
        assert result.message_type == "text"
        assert result.text == "Hi"
        assert result.sender_number == VALID_SENDER
        assert result.media_id is None

    def test_image_message(self):
        result = validate_and_parse(
            _raw(_make_payload("image", media_id="img_99", caption="wheat crop")),
            VALID_SENDER,
        )
        assert result.message_type == "image"
        assert result.media_id == "img_99"
        assert result.caption == "wheat crop"
        assert result.text is None

    def test_button_message(self):
        result = validate_and_parse(
            _raw(_make_payload("button", button_payload="BUY_NOW", button_text="Buy")),
            VALID_SENDER,
        )
        assert result.message_type == "button"
        assert result.button_payload == "BUY_NOW"
        assert result.text == "Buy"

    def test_sender_with_plus_prefix(self):
        result = validate_and_parse(_raw(_make_payload()), "+12025550123")
        assert result.sender_number == "+12025550123"

    def test_sender_without_plus_prefix(self):
        result = validate_and_parse(_raw(_make_payload()), "919876543210")
        assert result.sender_number == "919876543210"

    def test_whitespace_trimmed_from_text(self):
        payload = _make_payload("text", body="  hello  ")
        result = validate_and_parse(_raw(payload), VALID_SENDER)
        assert result.text == "hello"


# ── Malformed payload tests ───────────────────────────────────────────────────

class TestMalformedPayloads:
    def test_not_a_string(self):
        with pytest.raises(WebhookValidationError, match="must be a str"):
            validate_and_parse(12345, VALID_SENDER)  # type: ignore[arg-type]

    def test_invalid_json(self):
        with pytest.raises(WebhookValidationError, match="Invalid JSON"):
            validate_and_parse("{not valid json}", VALID_SENDER)

    def test_json_array_not_object(self):
        with pytest.raises(WebhookValidationError, match="root must be a JSON object"):
            validate_and_parse("[]", VALID_SENDER)

    def test_wrong_object_type(self):
        payload = _make_payload()
        payload["object"] = "something_else"
        with pytest.raises(WebhookValidationError, match="Unexpected object type"):
            validate_and_parse(_raw(payload), VALID_SENDER)

    def test_missing_entry(self):
        payload = _make_payload()
        del payload["entry"]
        with pytest.raises(WebhookValidationError, match="'entry'"):
            validate_and_parse(_raw(payload), VALID_SENDER)

    def test_empty_entry_list(self):
        payload = _make_payload()
        payload["entry"] = []
        with pytest.raises(WebhookValidationError, match="'entry' must be a non-empty list"):
            validate_and_parse(_raw(payload), VALID_SENDER)

    def test_empty_messages_list(self):
        payload = _make_payload()
        payload["entry"][0]["changes"][0]["value"]["messages"] = []
        with pytest.raises(WebhookValidationError, match="No messages found"):
            validate_and_parse(_raw(payload), VALID_SENDER)

    def test_message_missing_type(self):
        payload = _make_payload("text")
        del payload["entry"][0]["changes"][0]["value"]["messages"][0]["type"]
        with pytest.raises(WebhookValidationError):
            validate_and_parse(_raw(payload), VALID_SENDER)

    def test_text_message_missing_body(self):
        payload = _make_payload("text")
        # Replace text object with one missing 'body'
        payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"] = {}
        result = validate_and_parse(_raw(payload), VALID_SENDER)
        # Empty body is technically allowed (treated as empty string after strip)
        assert result.text == ""

    def test_text_field_not_object(self):
        payload = _make_payload("text")
        payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"] = "bad"
        with pytest.raises(WebhookValidationError, match="'text' field must be an object"):
            validate_and_parse(_raw(payload), VALID_SENDER)

    def test_image_field_not_object(self):
        payload = _make_payload("image")
        payload["entry"][0]["changes"][0]["value"]["messages"][0]["image"] = "bad"
        with pytest.raises(WebhookValidationError, match="'image' field must be an object"):
            validate_and_parse(_raw(payload), VALID_SENDER)


# ── Size-limit tests ──────────────────────────────────────────────────────────

class TestSizeLimits:
    def test_body_too_large(self):
        oversized = "x" * (MAX_BODY_BYTES + 1)
        with pytest.raises(WebhookValidationError, match="Payload too large"):
            validate_and_parse(oversized, VALID_SENDER)

    def test_text_body_too_long(self):
        long_text = "a" * (MAX_MESSAGE_TEXT_LEN + 1)
        payload = _make_payload("text", body=long_text)
        with pytest.raises(WebhookValidationError, match="too long"):
            validate_and_parse(_raw(payload), VALID_SENDER)

    def test_caption_too_long(self):
        long_caption = "c" * (MAX_CAPTION_LEN + 1)
        payload = _make_payload("image", caption=long_caption)
        with pytest.raises(WebhookValidationError, match="too long"):
            validate_and_parse(_raw(payload), VALID_SENDER)

    def test_sender_number_too_long(self):
        with pytest.raises(WebhookValidationError, match="too long"):
            validate_and_parse(_raw(_make_payload()), "+" + "9" * (MAX_SENDER_NUMBER_LEN + 5))


# ── Sender number validation ──────────────────────────────────────────────────

class TestSenderValidation:
    @pytest.mark.parametrize("bad_number", [
        "",                 # empty
        "abc",              # letters
        "123",              # too short (< 7 digits)
        "+0123456789",      # leading zero after +
        "++919876543210",   # double +
        "  ",               # whitespace only
    ])
    def test_invalid_sender_numbers(self, bad_number):
        with pytest.raises(WebhookValidationError):
            validate_and_parse(_raw(_make_payload()), bad_number)

    def test_non_string_sender(self):
        with pytest.raises(WebhookValidationError, match="must be a str"):
            validate_and_parse(_raw(_make_payload()), 919876543210)  # type: ignore[arg-type]


# ── Celery task integration tests ─────────────────────────────────────────────

class TestCeleryTaskValidation:
    """
    Verify that process_whatsapp_message:
      - Does NOT call any service function when payload is invalid.
      - DOES call the correct service function for valid payloads.
    """

    def _run_task(self, body, sender):
        """Execute the task synchronously (ALWAYS_EAGER style)."""
        from celery_worker import process_whatsapp_message

        # Call the underlying function directly to avoid broker dependency
        process_whatsapp_message.__wrapped__(
            process_whatsapp_message,  # self (bind=True)
            body,
            sender,
        )

    @patch("celery_worker._dispatch_to_service")
    def test_invalid_json_does_not_dispatch(self, mock_dispatch):
        self._run_task("{bad json}", VALID_SENDER)
        mock_dispatch.assert_not_called()

    @patch("celery_worker._dispatch_to_service")
    def test_oversized_body_does_not_dispatch(self, mock_dispatch):
        self._run_task("x" * (MAX_BODY_BYTES + 1), VALID_SENDER)
        mock_dispatch.assert_not_called()

    @patch("celery_worker._dispatch_to_service")
    def test_invalid_sender_does_not_dispatch(self, mock_dispatch):
        self._run_task(_raw(_make_payload()), "not-a-phone")
        mock_dispatch.assert_not_called()

    @patch("celery_worker._dispatch_to_service")
    def test_valid_payload_dispatches(self, mock_dispatch):
        self._run_task(_raw(_make_payload("text", body="hello")), VALID_SENDER)
        mock_dispatch.assert_called_once()
        called_with: NormalizedMessage = mock_dispatch.call_args[0][0]
        assert called_with.text == "hello"
        assert called_with.sender_number == VALID_SENDER

    @patch("celery_worker._dispatch_to_service")
    def test_empty_messages_list_does_not_dispatch(self, mock_dispatch):
        payload = _make_payload()
        payload["entry"][0]["changes"][0]["value"]["messages"] = []
        self._run_task(_raw(payload), VALID_SENDER)
        mock_dispatch.assert_not_called()