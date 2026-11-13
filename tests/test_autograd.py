"""Vérifie chaque gradient analytique contre une différence finie (en float64)."""

import numpy as np
import pytest

from miradorsi.core import functional as F
from miradorsi.core.autograd import Tensor, no_grad, set_default_dtype


@pytest.fixture(autouse=True)
def float64():
    set_default_dtype(np.float64)
    yield
    set_default_dtype(np.float32)


def numeric_grad(fn, x: np.ndarray, eps=1e-6) -> np.ndarray:
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        plus = fn()
        x[i] = old - eps
        minus = fn()
        x[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def check(build, *shapes, seed=0, tol=1e-5):
    """build(*tensors) -> Tensor ; compare grad analytique et numérique pour chaque entrée."""
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(s) for s in shapes]
    tensors = [Tensor(a, requires_grad=True) for a in arrays]
    projection = None

    def scalar(out_data):
        nonlocal projection
        if projection is None:
            projection = rng.standard_normal(out_data.shape)
        return float((out_data * projection).sum())

    out = build(*tensors)
    scalar(out.data)
    (out * Tensor(projection)).sum().backward()
    for t, a in zip(tensors, arrays):
        def f():
            with no_grad():
                return scalar(build(*[Tensor(x) for x in arrays]).data)
        expected = numeric_grad(f, a)
        np.testing.assert_allclose(t.grad, expected, rtol=tol, atol=tol)


def test_add_mul_broadcast():
    check(lambda a, b: a * b + a, (3, 4), (1, 4))


def test_sub_div_pow():
    check(lambda a, b: (a - b) / (b * b + 2.0) + a ** 2, (2, 3), (2, 3))


def test_matmul_batched():
    check(lambda a, b: a @ b, (2, 3, 4), (4, 5))
    check(lambda a, b: a @ b, (2, 2, 3, 4), (2, 2, 4, 3))


def test_reshape_transpose_getitem():
    check(lambda a: a.reshape(4, 6).transpose(1, 0)[1:4], (2, 3, 4))
    check(lambda a: a[np.array([0, 2, 2])], (3, 4))


def test_sum_mean_exp_log_tanh():
    check(lambda a: a.sum(axis=1, keepdims=True) * a.mean(axis=0), (3, 4))
    check(lambda a: (a * a + 1.0).log() + a.tanh() + a.exp(), (3, 3))


def test_softmax_gelu():
    check(lambda a: F.softmax(a, axis=-1), (2, 5))
    check(lambda a: F.gelu(a), (3, 4))


def test_rms_norm():
    check(lambda x, w: F.rms_norm(x, w), (2, 3, 6), (6,))


def test_rope():
    cos, sin = F.rope_tables(5, 4, dtype=np.float64)
    check(lambda x: F.rope(x, cos, sin), (2, 2, 5, 4))


def test_concat_and_mask():
    mask = np.triu(np.full((3, 3), -1e3), k=1)
    check(lambda a, b: F.add_constant(F.concat([a, b], axis=-1)[..., :3], mask), (3, 2), (3, 2))


def test_embedding():
    idx = np.array([[0, 2, 2], [1, 0, 3]])
    check(lambda w: F.embedding(w, idx), (4, 3))


def test_cross_entropy_with_ignore():
    targets = np.array([[1, 3, -1], [0, 2, 4]])
    check(lambda logits: F.cross_entropy(logits, targets), (2, 3, 5))


def test_gradient_accumulates_through_reuse():
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    (a * a + a).sum().backward()
    np.testing.assert_allclose(a.grad, [5.0, 7.0])


def test_no_grad_builds_no_graph():
    a = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        out = a * 2
    assert not out.requires_grad


def test_sigmoid_silu():
    check(lambda a: F.sigmoid(a) * F.silu(a), (3, 4))


def test_causal_conv():
    check(lambda x, w, b: F.causal_conv(x, w, b), (2, 6, 3), (4, 3), (3,))


def test_causal_conv_is_causal():
    x = np.random.default_rng(0).standard_normal((1, 6, 2))
    w, b = Tensor(np.ones((3, 2))), Tensor(np.zeros(2))
    y1 = F.causal_conv(Tensor(x), w, b).data
    x[0, 4:] = 99.0
    y2 = F.causal_conv(Tensor(x), w, b).data
    np.testing.assert_allclose(y1[0, :4], y2[0, :4])


def test_linear_scan():
    check(lambda a, b: F.linear_scan(F.sigmoid(a), b), (2, 7, 3), (2, 7, 3))


def test_linear_scan_matches_loop():
    rng = np.random.default_rng(1)
    a, b = rng.uniform(0, 1, (1, 5, 2)), rng.standard_normal((1, 5, 2))
    h, expected = np.zeros(2), []
    for t in range(5):
        h = a[0, t] * h + b[0, t]
        expected.append(h.copy())
    np.testing.assert_allclose(F.linear_scan(Tensor(a), Tensor(b)).data[0], np.array(expected))


def test_basic_slicing_gradient():
    check(lambda a: a[:, 1:3] * a[:, :2], (3, 4))
