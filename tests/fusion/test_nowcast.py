"""Unit tests for nowcast.py."""

# standard library
import unittest

# third party
import numpy as np

# first party
from delphi.nowcast.fusion.covariance import BlendDiagonal2

# py3tester coverage target
__test_target__ = 'delphi.nowcast.fusion.nowcast'


def get_data_source(truth, sensors, exclude_locations):

  class TestDataSource(DataSource):

    def get_locations(self):
      first_week = self.get_weeks()[0]
      return sorted(truth[first_week])

    def get_missing_locations(self, epiweek):
      return exclude_locations

    def get_sensors(self):
      return sorted(sensors)

    def get_weeks(self):
      return sorted(truth)

    def get_truth_value(self, epiweek, location):
      return truth[epiweek][location]

    def get_sensor_value(self, epiweek, location, name):
      return sensors[name][epiweek][location]

  return TestDataSource()


def get_scenario():
  N = None
  truth = {
    202020: {'jfk': 1, 'nj': 2, 'ny': 3},
    202021: {'jfk': 4, 'nj': 5, 'ny': 6},
    202022: {'jfk': N, 'nj': N, 'ny': N},
    202023: {'jfk': 7, 'nj': N, 'ny': 8},
  }
  sensors = {
    'a': {
      202020: {'jfk': 11, 'nj': 21, 'ny': 31},
      202021: {'jfk': 12, 'nj': 22, 'ny': N},
      202022: {'jfk': 13, 'nj': 23, 'ny': 33},
      202023: {'jfk': 14, 'nj': 24, 'ny': 34},
      202024: {'jfk': 15, 'nj': 25, 'ny': 35},
    },
    'b': {
      202020: {'jfk': N, 'nj': 41, 'ny': 51},
      202021: {'jfk': N, 'nj': 42, 'ny': 52},
      202022: {'jfk': N, 'nj': 43, 'ny': 53},
      202023: {'jfk': N, 'nj': 44, 'ny': 54},
      202024: {'jfk': N, 'nj': 45, 'ny': N},
    },
  }
  # assume HHS2 is only NY + NJ (i.e. pre-2012)
  exclude_locations = ('vi', 'pr')
  data_source = get_data_source(truth, sensors, exclude_locations)
  nowcaster = Nowcast(data_source, min_observations=2)
  test_weeks = [202022, 202023, 202024]
  return nowcaster, test_weeks


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def assertNowcast(self, nc, loc, value, stdev):
    """Assert that the given nowcast matches the expected values."""
    self.assertEqual(nc[0], loc)
    self.assertTrue(np.isclose(nc[1], value))
    self.assertTrue(np.isclose(nc[2], stdev))

  def test_get_sensor_data_for_all_weeks(self):
    nowcaster, test_weeks = get_scenario()
    inputs, noise, readings = nowcaster.get_sensor_data_for_all_weeks(
        test_weeks)

    # ('b', 'jfk') is missing because 'jfk' is never provided by sensor 'b'
    self.assertEqual(inputs, [
      ('a', 'jfk'), ('a', 'nj'), ('a', 'ny'), ('b', 'nj'), ('b', 'ny'),
    ])

    # expected values for this test scenario
    N = np.nan
    self.assertTrue(np.allclose(noise, [
      [10, 19, 28, 39, 48],
      [8, 17, N, 37, 46],
      [N, N, N, N, N],
      [7, N, 26, N, 46],
    ], equal_nan=True))
    self.assertTrue(np.allclose(readings, [
      [13, 23, 33, 43, 53],
      [14, 24, 34, 44, 54],
      [15, 25, 35, 45, N],
    ], equal_nan=True))

  def test_get_sensor_data_for_week(self):
    nowcaster, test_weeks = get_scenario()
    inputs, noise, readings = nowcaster.get_sensor_data_for_all_weeks(
        test_weeks)

    # various slices of the data, based on what's available where and when
    expected_locations = {
      202022: ('jfk', 'nj', 'nj', 'ny'),
      202023: ('jfk', 'nj', 'nj', 'ny'),
      202024: ('jfk', 'nj', 'ny', 'nj'),
    }
    N = np.nan
    expected_noise = {
      202022: [
        [10, 19, 39, 48],
        [8, 17, 37, 46],
      ],
      202023: [
        [10, 19, 39, 48],
        [8, 17, 37, 46],
      ],
      202024: [
        [10, 19, 28, 39],
        [8, 17, N, 37],
        [7, N, 26, N],
      ],
    }
    expected_readings = {
      202022: [13, 23, 43, 53],
      202023: [14, 24, 44, 54],
      202024: [15, 25, 35, 45],
    }

    for week, reading in zip(test_weeks, readings):
      with self.subTest(week=week):
        l, n, r = nowcaster.get_sensor_data_for_week(
            inputs, noise, week, reading)
        self.assertEqual(l, expected_locations[week])
        self.assertTrue(np.allclose(n, expected_noise[week], equal_nan=True))
        self.assertTrue(
            np.allclose(r, expected_readings[week], equal_nan=True))

  def test_compute_nowcast_independent(self):
    input_locations = ('hhs2', 'hhs3')
    A, B, C, D = 11, 13, 17, 19
    noise = np.array([
      [A, -B],
      [-A, B],
    ])
    reading = np.array([C, D])
    nc = Nowcast.compute_nowcast(
        input_locations, noise, reading, BlendDiagonal2)

    self.assertEqual(len(nc), 2)
    self.assertNowcast(nc[0], 'hhs2', C, A)
    self.assertNowcast(nc[1], 'hhs3', D, B)

  def test_compute_nowcast_excluded(self):
    input_locations = ('ar', 'la')
    A, B, C, D = 11, 13, 17, 19
    noise = np.array([
      [A, -B],
      [-A, B],
    ])
    reading = np.array([C, D])
    exclude_locations = ('ok', 'tx')
    nc = Nowcast.compute_nowcast(
        input_locations,
        noise,
        reading,
        BlendDiagonal2,
        exclude_locations=exclude_locations)

    self.assertEqual(len(nc), 3)
    self.assertNowcast(nc[1], 'ar', C, A)
    self.assertNowcast(nc[2], 'la', D, B)
    self.assertEqual(nc[0][0], 'cen7')
    self.assertTrue(min(C, D) < nc[0][1] < max(C, D))
    self.assertTrue(nc[0][2] < max(A, B))

  def test_compute_nowcast_inference(self):
    input_locations = ('jfk', 'ny')
    A, B, C, D = 11, 13, 17, 19
    noise = np.array([
      [A, -B],
      [-A, B],
    ])
    reading = np.array([C, D])
    nc = Nowcast.compute_nowcast(
        input_locations, noise, reading, BlendDiagonal2)

    self.assertEqual(len(nc), 3)
    self.assertNowcast(nc[1], 'ny', D, B)
    self.assertNowcast(nc[2], 'jfk', C, A)
    self.assertEqual(nc[0][0], 'ny_state')
    self.assertTrue(min(C, D) < nc[0][1] < max(C, D))
    self.assertTrue(nc[0][2] < max(A, B))

  def test_compute_nowcast_redundant(self):
    input_locations = ('cen9', 'cen9')
    A, B, C, D = 11, 13, 17, 19
    noise = np.array([
      [A, -B],
      [-A, B],
    ])
    reading = np.array([C, D])
    nc = Nowcast.compute_nowcast(
        input_locations, noise, reading, BlendDiagonal2)

    self.assertEqual(len(nc), 1)
    self.assertEqual(nc[0][0], 'cen9')
    self.assertTrue(min(C, D) < nc[0][1] < max(C, D))
    self.assertTrue(nc[0][2] < min(A, B))

  def test_batch_nowcast(self):
    nowcaster, test_weeks = get_scenario()
    ncs = nowcaster.batch_nowcast(test_weeks)

    self.assertEqual(len(ncs), len(test_weeks))

    expected_locations = ['hhs2', 'ny_state', 'nj', 'ny', 'jfk']
    for week, nc in zip(test_weeks, ncs):
      with self.subTest(week=week):
        self.assertEqual([l for l, v, s in nc], expected_locations)
        hhs2, ny_state, nj, ny, jfk = [n[1] for n in nc]
        # ny state is bounded by ny upstate and new york city
        self.assertTrue(min(ny, jfk) < ny_state < max(ny, jfk))
        # hhs2 is bounded by new jersey and new york (pr and vi are excluded)
        self.assertTrue(min(nj, ny_state) < hhs2 < max(nj, ny_state))

  def test_get_season_early(self):
    self.assertEqual(Nowcast.get_season(201740), 2017)

  def test_get_season_late(self):
    self.assertEqual(Nowcast.get_season(201839), 2017)
