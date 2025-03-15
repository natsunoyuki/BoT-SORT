from pathlib import Path
import numpy as np
import cv2
import yaml
import pytest


BASE_DIR = Path(__file__).resolve().parents[1] / "BoT-SORT"
TEST_DATA_DIR = BASE_DIR.parent / "tests" / "data"


@pytest.fixture
def human_video_sequence(
    sequence_dir="two_people_crossing", yml_file_name="detections.yml",
):
    """Returns a list of dictionaries each containing:
    - A video frame
    - Bounding boxes

    Yielding bounding box allows us to test dabble.tracking without having to
    attach a object detector before it.

    Yielding a list of frames instead of a video file allows for better control
    of test data and frame specific manipulations to trigger certain code
    branches.
    """
    with open(TEST_DATA_DIR / sequence_dir / yml_file_name) as infile:
        detections = yaml.safe_load(infile.read())

    yield [
        {
            "img": cv2.imread(str(TEST_DATA_DIR / sequence_dir / f"{key}.jpg")),
            "bboxes": np.array(val["bboxes"]),
            "scores": np.ones(len(val["bboxes"])),
            "labels": np.ones(len(val["bboxes"])),
        }
        for key, val in detections.items()
    ]
