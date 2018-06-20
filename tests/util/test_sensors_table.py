"""Unit tests for sensors_table.py."""

# standard library
import unittest
from unittest.mock import MagicMock

# py3tester coverage target
__test_target__ = 'delphi.nowcast.util.sensors_table'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_insert(self):
    """Insert a sensor reading."""

    database = MagicMock()
    name, location, epiweek, value = 'wiki', 'vi', 201820, 3.14
    SensorsTable(database=database).insert(name, location, epiweek, value)

    self.assertEqual(database.execute.call_count, 1)
    args, kwargs = database.execute.call_args
    sql, args = args
    self.assertEqual(sql, SensorsTable.SQL_INSERT)

  def test_get_most_recent_epiweek(self):
    """Get the epiweek of the most recent sensor reading."""

    database = MagicMock()
    name, location, epiweek = 'twtr', 'dc', 201820
    database.execute.return_value = [(epiweek,)]
    table = SensorsTable(database=database)
    returned_epiweek = table.get_most_recent_epiweek(name, location)

    self.assertEqual(database.execute.call_count, 1)
    args, kwargs = database.execute.call_args
    sql, args = args
    self.assertEqual(sql, SensorsTable.SQL_SELECT)
    self.assertEqual(args, (name, location))
    self.assertEqual(returned_epiweek, epiweek)

  def test_get_connection_info(self):
    """Return connection info."""

    username, password, database = SensorsTable()._get_connection_info()
    self.assertIsInstance(username, str)
    self.assertIsInstance(password, str)
    self.assertIsInstance(database, str)
