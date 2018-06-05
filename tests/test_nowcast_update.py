"""Unit tests for nowcast_update.py."""

# standard library
import argparse
import random
import unittest
from unittest.mock import MagicMock

# first party
from delphi.utils.epiweek import range_epiweeks

# py3tester coverage target
__test_target__ = 'delphi.nowcast.nowcast_update'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def setUp(self):
    random.seed(12345)

  def test_get_argument_parser(self):
    self.assertIsInstance(get_argument_parser(), argparse.ArgumentParser)

  def test_validate_args(self):
    with self.subTest(name='first only'):
      args = MagicMock(first=2, last=None, test=False)
      with self.assertRaises(Exception):
        validate_args(args)

    with self.subTest(name='last only'):
      args = MagicMock(first=None, last=1, test=False)
      with self.assertRaises(Exception):
        validate_args(args)

    with self.subTest(name='first > last'):
      args = MagicMock(first=2, last=1, test=False)
      with self.assertRaises(Exception):
        validate_args(args)

    with self.subTest(name='first < last'):
      args = MagicMock(first=1, last=2, test=False)
      self.assertEqual(validate_args(args), (1, 2, False))

    with self.subTest(name='test mode'):
      args = MagicMock(first=None, last=None, test=True)
      self.assertEqual(validate_args(args), (None, None, True))

  def test_new_instance(self):
    self.assertIsInstance(NowcastUpdate.new_instance(True), NowcastUpdate)

  def test_update(self):
    connector = MagicMock()
    database = NowcastDatabase(connector, False)
    data_source = MagicMock(
        get_truth_locations=lambda *a: ['nat', 'vi'],
        get_sensor_locations=lambda *a: ['nat', 'vi'],
        get_missing_locations=lambda *a: (),
        get_sensors=lambda *a: ['epic', 'sar3'],
        get_most_recent_issue=lambda *a: 201813,
        get_weeks=lambda *a: list(range_epiweeks(201713, 201813)),
        get_truth_value=lambda *a: random.random(),
        get_sensor_value=lambda *a: random.random(),
        prefetch=lambda *a: None)

    NowcastUpdate(database, data_source).update(201812, 201813)

    self.assertEqual(connector.connect.call_count, 1)
    cnx = connector.connect()
    self.assertEqual(cnx.cursor.call_count, 1)
    cur = cnx.cursor()
    self.assertEqual(cur.execute.call_count, 5)
    self.assertEqual(cur.close.call_count, 1)
    self.assertEqual(cnx.commit.call_count, 1)
    self.assertEqual(cnx.close.call_count, 1)

  def test_get_update_range(self):
    data_source = MagicMock(get_most_recent_issue=lambda *a: 201701)
    updater = NowcastUpdate(None, data_source)

    first_week, last_week = updater.get_update_range(None, None)

    self.assertEqual(first_week, 201701)
    self.assertEqual(last_week, 201702)

    first_week, last_week = updater.get_update_range(200101, 200201)

    self.assertEqual(first_week, 200101)
    self.assertEqual(last_week, 200201)
