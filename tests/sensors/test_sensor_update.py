"""Unit tests for sensor_update.py."""

# standard library
import argparse
import unittest
from unittest.mock import MagicMock

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
