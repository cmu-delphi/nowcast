"""Unit tests for flu_data_source.py."""

# standard library
import unittest
from unittest.mock import MagicMock

# first party
from delphi.utils.epiweek import range_epiweeks
from delphi.utils.geo.locations import Locations

# py3tester coverage target
__test_target__ = 'delphi.nowcast.util.flu_data_source'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_get_most_recent_issue(self):
    epidata = MagicMock()
    epidata.check.return_value = [{'issue': i} for i in [201802, 201801]]
    data_source = FluDataSource(epidata, None)
    issue = data_source.get_most_recent_issue()
    self.assertEqual(issue, 201802)

  def test_prefetch(self):
    # setup
    epidata = MagicMock()
    epidata.check.return_value = [{
      'epiweek': 201813,
      'num_providers': 1,
      'wili': 1,
      'value': 2
    }]
    data_source = FluDataSource(epidata, ['epic', 'sar3'])
    data_source.get_locations = lambda *a: ['nat', 'vi']

    # populate the cache
    data_source.prefetch(201813)

    self.assertEqual(epidata.fluview.call_count, 2)
    self.assertEqual(epidata.sensors.call_count, 4)
    self.assertEqual(epidata.check.call_count, 6)

    expected_names = set(['ilinet', 'epic', 'sar3'])
    actual_names = set(data_source.cache)
    self.assertEquals(expected_names, actual_names)

    expected_locations = set(['nat', 'vi'])
    actual_locations = set(data_source.cache['ilinet'])
    self.assertEquals(expected_locations, actual_locations)

    # cache hit (from prefetch)
    epidata.fluview.reset_mock()
    value = data_source.get_truth_value(201813, 'nat')
    self.assertEqual(value, 1)
    self.assertEqual(epidata.fluview.call_count, 0)

    # cache miss
    epidata.fluview.reset_mock()
    value = data_source.get_truth_value(201813, 'nm')
    self.assertEqual(value, None)
    self.assertEqual(epidata.fluview.call_count, 1)

    # cache hit (from miss)
    epidata.fluview.reset_mock()
    value = data_source.get_truth_value(201813, 'nm')
    self.assertEqual(value, None)
    self.assertEqual(epidata.fluview.call_count, 0)

  def test_implemented_methods(self):
    # sample data
    locations = ['ar', 'tx']
    sensors = ['epic', 'sar3']
    epiweek = 201812

    # helper that mimics an Epidata API response
    def fake_api(value=1, result=1, num_providers=1):
      return {
        'result': result,
        'epidata': [{
          'value': value,
          'wili': value,
          'num_providers': num_providers
        }]
      }

    # fake implementation of epidata.fluview
    def get_fluview(loc, week):
      if loc == 'X':
        return fake_api(num_providers=0)
      if loc in locations:
        return fake_api()
      return fake_api(result=-2)

    # fake implementation of epidata.sensors
    def get_sensors(auth, name, loc, week):
      if name in sensors:
        return fake_api()
      return fake_api(result=-2)

    # create data source
    epidata = MagicMock(fluview=get_fluview, sensors=get_sensors)
    data_source = FluDataSource(epidata, sensors)
    data_source.get_most_recent_issue = lambda: epiweek

    # expected values
    expected_locations = set(Locations.region_list)
    expected_missing = set(Locations.region_list) - set(locations)
    expected_sensors = set(sensors)
    expected_weeks = set(
        range_epiweeks(
            FluDataSource.FIRST_DATA_EPIWEEK, epiweek, inclusive=True))

    # actual values
    actual_locations = set(data_source.get_locations())
    actual_missing = set(data_source.get_missing_locations(None))
    actual_sensors = set(data_source.get_sensors())
    actual_weeks = set(data_source.get_weeks())

    # compare values
    self.assertEqual(actual_locations, expected_locations)
    self.assertEqual(actual_missing, expected_missing)
    self.assertEqual(actual_sensors, expected_sensors)
    self.assertEqual(actual_weeks, expected_weeks)

    # don't have data
    self.assertIsNone(data_source.get_truth_value(None, None))
    self.assertIsNone(data_source.get_sensor_value(None, None, None))

    # have data, but location had no reporting providers
    self.assertIsNone(data_source.get_truth_value(None, 'X'))

    # have data
    self.assertIsNotNone(data_source.get_truth_value(None, 'tx'))
    self.assertIsNotNone(data_source.get_sensor_value(None, None, 'epic'))

  def test_prefetch_missing_values(self):
    # setup
    epidata = MagicMock()
    epidata.fluview.return_value = {'result': -2}
    data_source = FluDataSource(epidata, ['wiki'])
    data_source.get_locations = lambda *a: ['nat']

    # populate the cache
    data_source.prefetch(201453)

    self.assertEqual(epidata.fluview.call_count, 1)
    self.assertEqual(epidata.sensors.call_count, 1)

  def test_missing_locations_all_reporting(self):
    """Expect no missing locations when all locations are reporting."""
    data_source = FluDataSource(None, None)
    data_source.get_truth_value = lambda *a: 1
    missing = data_source.get_missing_locations(None)
    self.assertEqual(missing, ())

  def test_missing_locations_some_reporting(self):
    """Expect some missing locations when some locations are reporting."""
    data_source = FluDataSource(None, None)
    data_source.get_truth_value = lambda e, l: {'nat': 1}.get(l, None)
    missing = data_source.get_missing_locations(None)
    self.assertTrue(len(missing) > 0)
    self.assertNotIn('nat', missing)

  def test_missing_locations_none_reporting(self):
    """Expect no missing locations when no locations are reporting."""
    data_source = FluDataSource(None, None)
    data_source.get_truth_value = lambda *a: None
    missing = data_source.get_missing_locations(None)
    self.assertEqual(missing, ())
