from unittest.mock import Mock

import pytest

from src.processing.classifier import classify_single_video


def test_valid_response():
    client = Mock()
    client.models.generate_content.return_value.text = '{"background": "sky"}'

    assert classify_single_video(client, []) == {"background": "sky"}


def test_request_error_is_preserved():
    client = Mock()
    error = TimeoutError("request timed out")
    client.models.generate_content.side_effect = error

    with pytest.raises(TimeoutError) as caught:
        classify_single_video(client, [])

    assert caught.value is error


@pytest.mark.parametrize(("text", "message"), [
    (None, "no response text"),
    ("  ", "no response text"),
    ("{", "invalid JSON"),
    ("[]", "non-empty JSON object"),
    ("{}", "non-empty JSON object"),
])
def test_invalid_response(text, message):
    client = Mock()
    client.models.generate_content.return_value.text = text

    with pytest.raises(ValueError, match=message):
        classify_single_video(client, [])