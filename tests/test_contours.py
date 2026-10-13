import numpy as np
import pytest

from visionguard.processing import ImageValidationError, extract_candidate_regions


def test_extracts_bounding_boxes_areas_and_centroids_in_stable_order():
    mask = np.zeros((18, 26), dtype=np.uint8)
    mask[2:6, 3:8] = 255
    mask[10:14, 15:21] = 255

    candidates = extract_candidate_regions(mask)

    assert [item.bounding_box for item in candidates] == [(3, 2, 5, 4), (15, 10, 6, 4)]
    assert [item.contour_area for item in candidates] == [12.0, 15.0]
    assert candidates[0].centroid == (5.0, 3.5)
    assert candidates[1].centroid == (17.5, 11.5)
    assert candidates[0].contour.ndim == 3


def test_area_limits_filter_small_regions():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    mask[5:10, 5:10] = 255

    candidates = extract_candidate_regions(mask, min_area=5, max_area=20)

    assert len(candidates) == 1
    assert candidates[0].bounding_box == (5, 5, 5, 5)


def test_holes_do_not_become_separate_external_candidates():
    mask = np.zeros((14, 14), dtype=np.uint8)
    mask[2:12, 2:12] = 255
    mask[5:9, 5:9] = 0

    candidates = extract_candidate_regions(mask)

    assert len(candidates) == 1
    assert candidates[0].bounding_box == (2, 2, 10, 10)


def test_single_pixel_candidate_uses_bounding_box_center_for_centroid():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[3, 2] = 255

    candidate = extract_candidate_regions(mask)[0]

    assert candidate.bounding_box == (2, 3, 1, 1)
    assert candidate.contour_area == 0.0
    assert candidate.centroid == (2.0, 3.0)


@pytest.mark.parametrize(
    "mask, kwargs",
    [
        (np.zeros((4, 4, 3), dtype=np.uint8), {}),
        (np.full((4, 4), 128, dtype=np.uint8), {}),
        (np.zeros((4, 4), dtype=np.uint8), {"min_area": -1}),
        (np.zeros((4, 4), dtype=np.uint8), {"max_area": float("nan")}),
        (np.zeros((4, 4), dtype=np.uint8), {"min_area": 10, "max_area": 5}),
    ],
)
def test_candidate_extraction_rejects_invalid_masks_or_area_limits(mask, kwargs):
    with pytest.raises(ImageValidationError):
        extract_candidate_regions(mask, **kwargs)
