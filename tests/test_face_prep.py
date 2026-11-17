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
