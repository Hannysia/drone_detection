import pytest
import cv2
from unittest.mock import MagicMock, patch
from src.processing.video import get_frame_count, get_video_metadata

@patch("src.processing.video.cv2.VideoCapture")
def test_get_frame_count_success(mock_video_capture):
    # Mocking successful video opening
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 150.0  # Simulated CAP_PROP_FRAME_COUNT
    mock_video_capture.return_value = mock_cap

    result = get_frame_count("mock_video.mp4")

    assert result == 150
    mock_cap.release.assert_called_once()


@patch("src.processing.video.cv2.VideoCapture")
def test_get_frame_count_failed_to_open(mock_video_capture):
    # Mocking video opening failure
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap

    result = get_frame_count("mock_video.mp4")

    assert result == 0
    mock_cap.release.assert_not_called()

@patch("src.processing.video.cv2.VideoCapture")
def test_get_video_metadata_success(mock_video_capture):
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 1920,
        cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_COUNT: 300
    }.get(prop, 0.0)
    mock_video_capture.return_value = mock_cap

    width, height, fps, length = get_video_metadata("mock_video.mp4")

    assert width == 1920
    assert height == 1080
    assert fps == pytest.approx(30.0)
    assert length == 300
    mock_cap.release.assert_called_once()