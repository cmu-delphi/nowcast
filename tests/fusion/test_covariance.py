"""Unit tests for covariance.py."""

# standard library
import unittest
from unittest.mock import MagicMock

# third party
import numpy as np

# py3tester coverage target
__test_target__ = 'delphi.nowcast.fusion.covariance'


def is_posdef(X):
  """Return whether the given matrix is positive definite."""
  return np.min(np.linalg.eigvals(X)) > 0


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_nancov(self):
    # no missing values
    X = np.random.randn(100, 3)
    cov1 = np.dot(X.T, X) / X.shape[0]
    cov2n, cov2d = nancov(X)
    cov2 = cov2n / cov2d
    # denominator as expected
    self.assertEqual(np.min(cov2d), 100)
    self.assertEqual(np.max(cov2d), 100)
    # equals expected covariance
    self.assertTrue(np.allclose(cov1, cov2))

    # many missing values
    X[:50, 0] = X[50:, 1] = X[25:75, 2] = np.nan
    cov2n, cov2d = nancov(X)
    # denominator as expected
    self.assertEqual(np.min(cov2d), 0)
    self.assertEqual(np.max(cov2d), 50)
    # numerator and denominator are symmetric
    self.assertTrue(np.allclose(cov2n, cov2n.T))
    self.assertTrue(np.allclose(cov2d, cov2d.T))

  def test_log_likelihood(self):
    cov = np.eye(3)
    data = np.random.randn(100, 3)
    ll = log_likelihood(cov, data)
    self.assertTrue(-np.inf < ll < 0)

  def test_posdef_max_likelihood_objective(self):
    X = np.zeros((2, 2)) * np.nan

    # posdef covariance
    shrinkage = MagicMock()
    shrinkage.get_cov = MagicMock(return_value=np.eye(2))
    objective = posdef_max_likelihood_objective(X, shrinkage)
    value = objective(None)
    self.assertEqual(shrinkage.get_cov.call_count, 1)
    self.assertTrue(-np.inf < value < 0)

    # non-posdef covariance
    shrinkage.get_cov = MagicMock(return_value=np.array([[1, 0], [0, -1]]))
    objective = posdef_max_likelihood_objective(X, shrinkage)
    value = objective(None)
    self.assertEqual(shrinkage.get_cov.call_count, 1)
    self.assertEqual(value, -np.inf)

  def test_mle_cov(self):
    X = np.random.randn(100, 3)
    shrinkage = MagicMock()
    shrinkage.get_alpha_bounds = MagicMock(return_value=(0, 1))
    shrinkage.get_cov = MagicMock(return_value=np.eye(3))
    cov = mle_cov(X, lambda *args: shrinkage)
    self.assertTrue(is_posdef(cov))
    self.assertTrue(-np.inf < log_likelihood(cov, X) < 0)
    self.assertTrue(shrinkage.get_alpha_bounds.called)
    self.assertTrue(shrinkage.get_cov.called)

  def test_shrinkage_methods(self):
    num, den, obs = np.eye(2), np.ones((2, 2)), 10
    for class_ in (BlendDiagonal0, BlendDiagonal1, BlendDiagonal2):
      with self.subTest(class_=class_):
        instance = class_(num, den, obs)
        a, b = instance.get_alpha_bounds()
        self.assertTrue(np.isfinite(a))
        self.assertTrue(np.isfinite(b))
        self.assertTrue(a < b)
        cov0 = instance.get_cov(a)
        cov1 = instance.get_cov((a + b) / 2)
        cov2 = instance.get_cov(b)
        self.assertTrue(np.allclose(cov0, cov0.T))
        self.assertTrue(np.allclose(cov1, cov1.T))
        self.assertTrue(np.allclose(cov2, cov2.T))
        self.assertTrue(is_posdef(cov0))
        self.assertTrue(is_posdef(cov1))
        self.assertTrue(is_posdef(cov2))
