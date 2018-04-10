"""Unit tests for database.py."""

# standard library
import unittest
from unittest.mock import MagicMock

# py3tester coverage target
__test_target__ = 'delphi.nowcast.util.database'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_new_instance(self):
    self.assertIsInstance(NowcastDatabase.new_instance(True), NowcastDatabase)

  def test_test_mode(self):
    connector = MagicMock()
    with NowcastDatabase(connector, True) as db:
      pass

    self.assertEqual(connector.connect.call_count, 1)
    cnx = connector.connect()
    self.assertEqual(cnx.cursor.call_count, 1)
    cur = cnx.cursor()
    self.assertEqual(cur.execute.call_count, 0)
    self.assertEqual(cur.close.call_count, 1)
    self.assertEqual(cnx.commit.call_count, 0)
    self.assertEqual(cnx.close.call_count, 1)

  def test_real_mode(self):
    connector = MagicMock()
    with NowcastDatabase(connector, False) as db:
      pass

    self.assertEqual(connector.connect.call_count, 1)
    cnx = connector.connect()
    self.assertEqual(cnx.cursor.call_count, 1)
    cur = cnx.cursor()
    self.assertEqual(cur.execute.call_count, 0)
    self.assertEqual(cur.close.call_count, 1)
    self.assertEqual(cnx.commit.call_count, 1)
    self.assertEqual(cnx.close.call_count, 1)

  def test_insert(self):
    connector = MagicMock()
    epiweek, location, value, stdev = 1, 'a', 2, 3
    with NowcastDatabase(connector, False) as db:
      db.insert(epiweek, location, value, stdev)

    self.assertEqual(connector.connect.call_count, 1)
    cnx = connector.connect()
    self.assertEqual(cnx.cursor.call_count, 1)
    cur = cnx.cursor()
    self.assertEqual(cur.execute.call_count, 1)
    self.assertEqual(cur.close.call_count, 1)
    self.assertEqual(cnx.commit.call_count, 1)
    self.assertEqual(cnx.close.call_count, 1)

  def test_set_last_update_time(self):
    connector = MagicMock()
    with NowcastDatabase(connector, False) as db:
      db.set_last_update_time()

    self.assertEqual(connector.connect.call_count, 1)
    cnx = connector.connect()
    self.assertEqual(cnx.cursor.call_count, 1)
    cur = cnx.cursor()
    self.assertEqual(cur.execute.call_count, 1)
    self.assertEqual(cur.close.call_count, 1)
    self.assertEqual(cnx.commit.call_count, 1)
    self.assertEqual(cnx.close.call_count, 1)
