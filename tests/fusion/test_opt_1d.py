"""Unit tests for opt_1d.py."""

# standard library
import math
import unittest

# py3tester coverage target
__test_target__ = 'delphi.nowcast.fusion.opt_1d'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  stop = lambda num, size, value: size <= 1e-6

  def assertApprox(self, a, b):
    self.assertEqual(round(a - b, 5), 0)

  def test_line(self):
    """maximize `y = x` over [0, 1]"""
    x, y = maximize(0, 1, lambda x: x, UnitTests.stop)
    self.assertApprox(x, 1)
    self.assertApprox(y, 1)

  def test_parabola(self):
    """maximize `y = -x^2` over [-1, 1]"""
    x, y = maximize(-1, 1, lambda x: -x * x, UnitTests.stop)
    self.assertApprox(x, 0)
    self.assertApprox(y, 0)

  def test_cosine(self):
    """maximize `y = cos(x)` over [0, pi]"""
    x, y = maximize(0, math.pi, lambda x: math.cos(x), UnitTests.stop)
    self.assertApprox(x, 0)
    self.assertApprox(y, 1)

  def test_polynomial(self):
    """maximize `y = x + x^2 - x^4` over [-2, 2]"""
    x, y = maximize(0, math.pi, lambda x: x + x ** 2 - x ** 4, UnitTests.stop)
    self.assertApprox(x, 0.88465)
    self.assertApprox(y, 1.05478)
