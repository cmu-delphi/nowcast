"""Unit tests for fusion.py."""

# standard library
import unittest

# third party
import numpy as np

# py3tester coverage target
__test_target__ = 'delphi.nowcast.fusion.fusion'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_fuse(self):
    num_states = 5
    num_inputs = 10
    z = np.ones((1, num_inputs))
    R = np.eye(num_inputs)
    H1 = np.eye(num_states)
    H2 = np.ones((num_inputs - num_states, num_states)) / num_states
    H = np.vstack((H1, H2))
    x, P = fuse(z, R, H)
    self.assertTrue(np.allclose(x, np.ones((1, num_states))))
    self.assertTrue(np.allclose(P, np.linalg.inv(np.dot(H.T, H))))

  def test_extract(self):
    num_states = 5
    num_outputs = 10
    x = np.ones((1, num_states))
    P = np.eye(num_states)
    W = np.ones((num_outputs, num_states)) / num_states
    y, S = extract(x, P, W)
    S_expected = np.ones((num_outputs, num_outputs)) / num_states
    self.assertTrue(np.allclose(y, np.ones((1, num_outputs))))
    self.assertTrue(np.allclose(S, S_expected))

  def test_eliminate(self):
    fractions = lambda X: np.array([[Fraction(x) for x in row] for row in X])

    X = eliminate(fractions(-np.eye(3)))
    self.assertTrue(np.allclose(X.astype(np.float), np.eye(3)))

    X = fractions([
      [6, 7, 8],
      [3, 5, 7],
      [11, 23, 31],
    ])
    eliminate(X)
    self.assertTrue(np.allclose(X.astype(np.float), np.eye(3)))

    X = fractions([
      [0, 1, -3, 4, 1],
      [2, -2, 1, 0, -1],
      [2, -1, -2, 4, 0],
      [-6, 4, 3, -8, 1],
    ])
    eliminate(X)
    X = X.astype(np.float)
    Y = np.array([
      [1, 0, -2.5, 4, 0.5],
      [0, 1, -3, 4, 1],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
    ])
    self.assertTrue(np.allclose(X, Y))

  def test_matmul(self):
    fractions = lambda X: np.array([[Fraction(x) for x in row] for row in X])
    rand = lambda a, b: (np.random.randn(a, b) * 3).astype(np.int)

    X, Y, Z = rand(4, 5), rand(5, 6), rand(6, 3)
    self.assertIs(X, matmul(X))

    XYZ1 = np.dot(np.dot(X, Y), Z)
    XYZ2 = matmul(*list(map(fractions, (X, Y, Z)))).astype(np.int)
    self.assertTrue(np.allclose(XYZ1, XYZ2))

    X, Y = fractions(rand(1, 2)), fractions(rand(3, 4))
    with self.assertRaises(Exception):
      matmul(X, Y)

  def test_determine_statespace(self):
    # sample data from email "improvements to nowcasting"
    states = ('a', 'b', 'c', 'd', 'e', 'f')
    populations = (1, 2, 3, 4, 5, 6)
    regions = (
      'nat', 'h1', 'h2', 'h3', 'v1', 'v2', 'v3', 'a', 'b', 'c', 'd', 'e', 'f'
    )
    rows = (
      (1, 1, 1, 1, 1, 1),
      (1, 0, 0, 1, 0, 0),
      (0, 1, 1, 0, 0, 0),
      (0, 0, 0, 0, 1, 1),
      (1, 0, 0, 1, 0, 0),
      (0, 1, 0, 0, 1, 0),
      (0, 0, 1, 0, 0, 1),
      (1, 0, 0, 0, 0, 0),
      (0, 1, 0, 0, 0, 0),
      (0, 0, 1, 0, 0, 0),
      (0, 0, 0, 1, 0, 0),
      (0, 0, 0, 0, 1, 0),
      (0, 0, 0, 0, 0, 1),
    )
    makeup = dict((loc, row) for (loc, row) in zip(regions, rows))

    def get_pop(location):
      return sum(m * p for (m, p) in zip(makeup[location], populations))

    def get_row(location):
      total = get_pop(location)
      row = []
      for (s, m) in zip(states, makeup[location]):
        row.append(Fraction(m * get_pop(s), total))
      return row

    def get_matrix(locations):
      return np.array([get_row(loc) for loc in locations])

    def assertRowsSumToOne(matrix):
      self.assertTrue(np.allclose(np.sum(matrix.astype(np.float), axis=1), 1))

    def assertStatespace(sensors, expected_num_states, expected_outputs):
      num_inputs = len(sensors)
      num_states = len(states)
      num_outputs = len(regions)
      H0 = get_matrix(sensors)
      W0 = get_matrix(regions)
      self.assertEqual(H0.shape, (num_inputs, num_states))
      self.assertEqual(W0.shape, (num_outputs, num_states))
      assertRowsSumToOne(H0)
      assertRowsSumToOne(W0)
      H, W, actual_rows = determine_statespace(H0, W0)
      num_latent_states = H.shape[1]
      self.assertEqual(num_latent_states, expected_num_states)
      self.assertEqual(H.shape, (num_inputs, num_latent_states))
      self.assertEqual(W.shape, (len(actual_rows), num_latent_states))
      outputs = tuple(regions[i] for i in actual_rows)
      self.assertEqual(outputs, expected_outputs)

    sensors = (
      'nat', 'nat', 'nat', 'h1', 'h2', 'h3', 'v1', 'v2', 'v3', 'b', 'b', 'b'
    )
    expected_num_states = 5
    expected_outputs = (
      'nat', 'h1', 'h2', 'h3', 'v1', 'v2', 'v3', 'b', 'c', 'e', 'f'
    )
    with self.subTest(sensors=sensors):
      assertStatespace(sensors, expected_num_states, expected_outputs)

    sensors = ('h1', 'h2', 'h3')
    expected_num_states = 3
    expected_outputs = ('nat', 'h1', 'h2', 'h3', 'v1')
    with self.subTest(sensors=sensors):
      assertStatespace(sensors, expected_num_states, expected_outputs)

    sensors = states
    expected_num_states = 6
    expected_outputs = regions
    with self.subTest(sensors=sensors):
      assertStatespace(sensors, expected_num_states, expected_outputs)
