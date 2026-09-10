"""World-frame FMM gradient conversion (overflow / direction)."""

import numpy as np

from seqseg.modules.centerline import world_frame_gradient


def test_world_frame_gradient_identity_spacing():
    # f(x,y,z) = 2x + 3y + 4z on a unit grid -> grad = (2, 3, 4)
    x = np.arange(5)[:, None, None]
    y = np.arange(5)[None, :, None]
    z = np.arange(5)[None, None, :]
    field = 2.0 * x + 3.0 * y + 4.0 * z
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    grad = world_frame_gradient(field, (1.0, 1.0, 1.0), identity)
    interior = grad[:, 1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(interior[0], 2.0, atol=1e-12)
    np.testing.assert_allclose(interior[1], 3.0, atol=1e-12)
    np.testing.assert_allclose(interior[2], 4.0, atol=1e-12)


def test_world_frame_gradient_scales_by_inv_spacing():
    x = np.arange(5, dtype=np.float64)[:, None, None]
    field = np.broadcast_to(x, (5, 5, 5)).copy()
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # Two voxels of +1 in x, spacing 2 mm -> 0.5 / mm
    grad = world_frame_gradient(field, (2.0, 2.0, 2.0), identity)
    np.testing.assert_allclose(grad[0, 1:-1, :, :], 0.5, atol=1e-12)


def test_world_frame_gradient_float32_largevalue_does_not_overflow():
    """ITK FastMarching unreached voxels are ~1e38 in float32."""
    field = np.zeros((6, 6, 6), dtype=np.float32)
    field[2:4, 2:4, 2:4] = 10.0
    field[0, 0, 0] = np.float32(1.7e38)
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    with np.errstate(over="raise"):
        grad = world_frame_gradient(field, (0.65, 0.65, 1.0), identity)
    assert np.isfinite(grad).all()


def test_world_frame_gradient_applies_direction_flip():
    x = np.arange(5, dtype=np.float64)[:, None, None]
    field = np.broadcast_to(x, (5, 5, 5)).copy()
    flip_x = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    grad = world_frame_gradient(field, (1.0, 1.0, 1.0), flip_x)
    np.testing.assert_allclose(grad[0, 1:-1, :, :], -1.0, atol=1e-12)
    np.testing.assert_allclose(grad[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(grad[2], 0.0, atol=1e-12)
