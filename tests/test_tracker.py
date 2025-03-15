import pytest

from bot_sort import BoTSORT


@pytest.fixture
def tracker():
    return BoTSORT()


class TestBoTSORT:
    def test_tracking_ids_should_be_consistent_across_frames(
        self, tracker, human_video_sequence,
    ):
        detections = human_video_sequence
        prev_tags = []
        for i, dets in enumerate(detections):
            tracked_objects = tracker.update(
                dets["bboxes"], dets["labels"], dets["scores"],
            )
            ids = [o.track_id for o in tracked_objects]

            assert len(ids) == len(dets["bboxes"])

            if i > 0:
                assert ids == prev_tags

            prev_tags = ids

    # TODO reID, camera motion estimation.
