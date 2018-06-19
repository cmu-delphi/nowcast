"""Unit tests for database.py."""

# standard library
import unittest
from unittest.mock import MagicMock

# py3tester coverage target
__test_target__ = 'delphi.nowcast.util.nowcasts_table'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_insert(self):
    """Insert a nowcast."""

    database = MagicMock()
    epiweek, location, value, stdev = 1, 'a', 2, 3
    NowcastsTable(database=database).insert(epiweek, location, value, stdev)

    self.assertEqual(database.execute.call_count, 1)
    args, kwargs = database.execute.call_args
    sql, args = args
    self.assertEqual(sql, NowcastsTable.SQL_INSERT)
    self.assertEqual(args, (epiweek, location, value, stdev, value, stdev))

  def test_set_last_update_time(self):
    """Set the timestamp of last nowcast update."""

    database = MagicMock()
    NowcastsTable(database=database).set_last_update_time()

    self.assertEqual(database.execute.call_count, 1)
    args, kwargs = database.execute.call_args
    sql, args = args
    self.assertEqual(sql, NowcastsTable.SQL_INSERT)
    self.assertEqual(args[0], 0)
    self.assertEqual(args[1], 'updated')
    self.assertIsInstance(args[2], int)
    self.assertIsInstance(args[3], int)
    self.assertEqual(args[2], args[4])
    self.assertEqual(args[3], args[5])

  def test_get_connection_info(self):
    """Return connection info."""

    username, password, database = NowcastsTable()._get_connection_info()
    self.assertIsInstance(username, str)
    self.assertIsInstance(password, str)
    self.assertIsInstance(database, str)
