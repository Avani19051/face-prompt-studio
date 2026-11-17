"""Checks for the alignment maths (no GPU or model download needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_prep import (  # noqa: E402
    ARCFACE_TEMPLATE_112,
    alignment_matrix,
    landmarks68_to_5,
    make_template,
)


def _apply(m, pts):
    pts = np.hstack([pts, np.ones((len(pts), 1))])
    return (m @ pts.T).T[:, :2]


def test_68_to_5_picks_eye_centres_nose_and_mouth_corners():
    lmk = np.zeros((68, 2), np.float32)
    lmk[36:42] = [10, 20]
    lmk[42:48] = [30, 20]
    lmk[30] = [20, 30]
    lmk[48] = [12, 40]
    lmk[54] = [28, 40]
    np.testing.assert_allclose(landmarks68_to_5(lmk), [[10, 20], [30, 20], [20, 30], [12, 40], [28, 40]])


def test_68_to_5_rejects_wrong_shape():
    with pytest.raises(ValueError):
        landmarks68_to_5(np.zeros((5, 2)))


def test_template_scaling_and_margin():
    np.testing.assert_allclose(make_template(112), ARCFACE_TEMPLATE_112, rtol=1e-6)
    t = make_template(512, margin=0.5)
    # with a 50% margin the face's bounding box is centred and half as wide
    assert np.isclose(t[:, 0].mean(), 512 / 112 * 0.5 * ARCFACE_TEMPLATE_112[:, 0].mean() + 128)
    with pytest.raises(ValueError):
        make_template(512, margin=1.0)


def test_alignment_undoes_rotation_scale_and_shift():
    template = make_template(512, 0.4)
    theta = np.deg2rad(25)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    photo_pts = (template @ rot.T) * 0.6 + np.array([300.0, 150.0])  # a tilted, smaller, shifted face
    m = alignment_matrix(photo_pts, template)
    np.testing.assert_allclose(_apply(m, photo_pts), template, atol=1e-3)


def test_warp_moves_marked_pixels_onto_template():
    cv2 = pytest.importorskip("cv2")
    from face_prep import warp

    size = 256
    template = make_template(size, 0.4)
    # draw 5 bright dots at "landmark" positions in a 400x400 photo, rotated and shifted
    theta = np.deg2rad(-15)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    photo_pts = (template @ rot.T) * 1.2 + np.array([40.0, 60.0])
    photo = np.zeros((400, 400, 3), np.uint8)
    for x, y in photo_pts:
        cv2.circle(photo, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)
    out = warp(photo, alignment_matrix(photo_pts, template), size)
    for x, y in template:  # every dot should now sit on its template position
        assert out[int(round(y)), int(round(x))].max() > 128


def test_identity_summary():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from evaluate_identity import summarize

    s = summarize([0.62, 0.41, None, 0.55])
    assert s["images"] == 4 and s["faces_found"] == 3
    assert np.isclose(s["mean_similarity"], (0.62 + 0.41 + 0.55) / 3)
    assert np.isclose(s["share_above_0.5"], 2 / 3)
