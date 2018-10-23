"""Unit tests for sensor_update.py."""

# standard library
import argparse
import unittest
from unittest.mock import MagicMock

# first party
from delphi.utils.geo.locations import Locations

# py3tester coverage target
__test_target__ = 'delphi.nowcast.sensors.sensor_update'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_get_argument_parser(self):
    """An ArgumentParser should be returned."""
    self.assertIsInstance(get_argument_parser(), argparse.ArgumentParser)

  def test_validate_args(self):
    """Arguments should be validated."""

    def get_args(
        names='arch-nat',
        first=None,
        last=None,
        epiweek=None,
        test=False,
        valid=False):
      return MagicMock(
          names=names,
          first=first,
          last=last,
          epiweek=epiweek,
          test=test,
          valid=valid)

    with self.subTest(name='first after last'):
      with self.assertRaises(ValueError):
        validate_args(get_args(first=201801, last=201701))

    with self.subTest(name='first and epiweek'):
      with self.assertRaises(ValueError):
        validate_args(get_args(first=201801, epiweek=201901))

    with self.subTest(name='last and epiweek'):
      with self.assertRaises(ValueError):
        validate_args(get_args(last=201701, epiweek=201901))

    with self.subTest(name='only epiweek'):
      names, first, last, valid, test = validate_args(get_args(epiweek=201901))
      self.assertEqual(first, 201901)
      self.assertEqual(last, 201901)

    with self.subTest(name='invalid names'):
      with self.assertRaises(ValueError):
        validate_args(get_args(names='hello world'))

    with self.subTest(name='valid names'):
      args = get_args(names='abc-def,foo-bar,123-321')
      names, first, last, valid, test = validate_args(args)
      self.assertEqual(names, 'abc-def,foo-bar,123-321')

  def test_new_instance(self):
    """Create a SensorUpdate instance with default parameters."""
    self.assertIsInstance(SensorUpdate.new_instance(True, True), SensorUpdate)

  def test_parse_sensor_location_pairs(self):
    """Parse sensor name/location pairs."""
    names = parse_sensor_location_pairs('abc-def,foo-bar,123-321')
    self.assertEqual(names, [['abc', 'def'], ['foo', 'bar'], ['123', '321']])

  def test_get_most_recent_issue(self):
    """Fetch the most recent issue."""

    epidata = MagicMock()
    epidata.check.return_value = [{'issue': 201820}, {'issue': 201721}]

    self.assertEqual(get_most_recent_issue(epidata), 201820)
    self.assertEqual(epidata.fluview.call_count, 1)
    self.assertEqual(epidata.range.call_count, 1)

    args, kwargs = epidata.fluview.call_args
    self.assertEqual('nat', args[0])

    args, kwargs = epidata.range.call_args
    ew1, ew2 = args
    self.assertIsInstance(ew1, int)
    self.assertIsInstance(ew2, int)
    self.assertTrue(ew1 < ew2)

  def test_get_sensor_implementations(self):
    """Get a map of sensor implementations."""
    impls = SensorGetter.get_sensor_implementations()
    self.assertIsInstance(impls, dict)
    self.assertIn('sar3', impls)
    self.assertTrue(callable(impls['sar3']))

  def test_update_single(self):
    """Update a single sensor reading."""

    database = MagicMock()
    valid, test_week, name, location, value = True, 201820, 'name', 'loc', 3.14
    impl = MagicMock(return_value=value)
    implementations = {name: impl}
    sensor_update = SensorUpdate(valid, None, implementations, None)

    sensor_update.update_single(database, test_week, name, location)

    self.assertEqual(impl.call_count, 1)
    args, kwargs = impl.call_args
    self.assertEqual(args, (location, test_week - 1, valid))

    self.assertEqual(database.insert.call_count, 1)
    args, kwargs = database.insert.call_args
    self.assertEqual(args, (name, location, test_week, value))

  def test_update_single_tolerates_sensor_failure(self):
    """Suppress failure to read a sensor."""

    database = MagicMock()
    valid, test_week, name, location, value = True, 201820, 'name', 'loc', 3.14
    impl = MagicMock(side_effect=Exception)
    implementations = {name: impl}
    sensor_update = SensorUpdate(valid, None, implementations, None)

    sensor_update.update_single(database, test_week, name, location)

    self.assertEqual(impl.call_count, 1)
    args, kwargs = impl.call_args
    self.assertEqual(args, (location, test_week - 1, valid))

  def test_update_single_does_not_toerate_database_failure(self):
    """Propagate failure to write to database."""

    class ExpectedException(Exception):
      """Expect the database to raise this exception."""

    database = MagicMock()
    database.insert = MagicMock(side_effect=ExpectedException)
    valid, test_week, name, location, value = True, 201820, 'name', 'loc', 3.14
    impl = MagicMock(return_value=value)
    implementations = {name: impl}
    sensor_update = SensorUpdate(valid, None, implementations, None)

    with self.assertRaises(ExpectedException):
      sensor_update.update_single(database, test_week, name, location)

    self.assertEqual(impl.call_count, 1)
    args, kwargs = impl.call_args
    self.assertEqual(args, (location, test_week - 1, valid))

    self.assertEqual(database.insert.call_count, 1)
    args, kwargs = database.insert.call_args
    self.assertEqual(args, (name, location, test_week, value))

  def test_get_location_list(self):
    """Get list of locations in a named group."""

    for loc in ['fl', 'dc', 'ny_minus_jfk', 'hhs2', 'cen4', 'nat']:
      with self.subTest(name=loc):
        self.assertEqual(get_location_list(loc), [loc])

    with self.subTest(name='all'):
      self.assertEqual(get_location_list('all'), Locations.region_list)

    with self.subTest(name='hhs'):
      self.assertEqual(get_location_list('hhs'), Locations.hhs_list)

    with self.subTest(name='cen'):
      self.assertEqual(get_location_list('cen'), Locations.cen_list)

    with self.subTest(name='sta'):
      with self.assertRaises(UnknownLocationException):
        get_location_list('sta')

  def test_update(self):
    """Bulk update sensor readings."""

    database = MagicMock()
    database.__enter__.return_value = database
    database.get_most_recent_epiweek.return_value = 201820

    epidata = MagicMock()
    epidata.check.return_value = [{'issue': 201820}]

    impl1 = MagicMock(return_value=1)
    impl2 = MagicMock(return_value=2)
    implementations = {'s1': impl1, 's2': impl2}

    sensors = [('s1', 'ar'), ('s2', 'or')]

    sensor_update = SensorUpdate(True, database, implementations, epidata)
    sensor_update.update(sensors, None, None)

    self.assertEqual(database.get_most_recent_epiweek.call_count, 2)
    args = [a for a, k in database.get_most_recent_epiweek.call_args_list]
    self.assertEqual(args[0], ('s1', 'ar'))
    self.assertEqual(args[1], ('s2', 'or'))

    self.assertEqual(impl1.call_count, 2)
    for i, (args, kwargs) in enumerate(impl1.call_args_list):
      self.assertEqual(args, ('ar', 201819 + i, True))

    self.assertEqual(impl2.call_count, 2)
    for i, (args, kwargs) in enumerate(impl2.call_args_list):
      self.assertEqual(args, ('or', 201819 + i, True))

    self.assertEqual(database.insert.call_count, 4)
    args = [a for a, k in database.insert.call_args_list]
    self.assertEqual(args[0], ('s1', 'ar', 201820, 1))
    self.assertEqual(args[1], ('s1', 'ar', 201821, 1))
    self.assertEqual(args[2], ('s2', 'or', 201820, 2))
    self.assertEqual(args[3], ('s2', 'or', 201821, 2))

  def test_update_with_unknown_epiweek(self):
    """Bulk update sensor readings for a new sensor."""

    database = MagicMock()
    database.__enter__.return_value = database
    database.get_most_recent_epiweek.return_value = None

    impl1 = MagicMock(return_value=1)
    impl2 = MagicMock(return_value=2)
    implementations = {'s1': impl1, 's2': impl2}

    sensors = [('s1', 'ar'), ('s2', 'or')]

    sensor_update = SensorUpdate(True, database, implementations, None)
    sensor_update.update(sensors, None, 201040)

    args = [a for a, k in database.get_most_recent_epiweek.call_args_list]
    self.assertEqual(args[0], ('s1', 'ar'))
    self.assertEqual(args[1], ('s2', 'or'))

    self.assertEqual(impl1.call_count, 1)
    args, kwargs = impl1.call_args
    self.assertEqual(args, ('ar', 201039, True))

    self.assertEqual(impl2.call_count, 1)
    args, kwargs = impl2.call_args
    self.assertEqual(args, ('or', 201039, True))

    self.assertEqual(database.insert.call_count, 2)
    args = [a for a, k in database.insert.call_args_list]
    self.assertEqual(args[0], ('s1', 'ar', 201040, 1))
    self.assertEqual(args[1], ('s2', 'or', 201040, 2))

  def test_update_with_failing_sensors(self):
    """Bulk update sensor readings with some missing values."""

    database = MagicMock()
    database.__enter__.return_value = database

    def impl(location, epiweek, valid):
      # missing all but these
      return {'ak': 1, 'ar': 2, 'az': 3}[location]
    implementations = {'s': impl}

    sensors = [('s', 'all')]

    sensor_update = SensorUpdate(True, database, implementations, None)
    sensor_update.update(sensors, 201820, 201820)

    self.assertEqual(database.insert.call_count, 3)
    args = [a for a, k in database.insert.call_args_list]
    self.assertEqual(args[0], ('s', 'ak', 201820, 1))
    self.assertEqual(args[1], ('s', 'ar', 201820, 2))
    self.assertEqual(args[2], ('s', 'az', 201820, 3))

  # TODO: more tests
