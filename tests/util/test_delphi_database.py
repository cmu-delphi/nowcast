"""Unit tests for database.py."""

# standard library
import unittest
from unittest.mock import MagicMock

# py3tester coverage target
__test_target__ = 'delphi.nowcast.util.delphi_database'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_test_mode(self):
    """Close without commit."""

    connector = MagicMock()
    with DelphiDatabase(connector, True, 'u', 'p', 'd') as db:
      pass

    self.assertEqual(connector.connect.call_count, 1)
    args, kwargs = connector.connect.call_args
    self.assertEqual(kwargs, {'user': 'u', 'password': 'p', 'database': 'd'})
    cnx = connector.connect()
    self.assertEqual(cnx.cursor.call_count, 1)
    cur = cnx.cursor()
    self.assertEqual(cur.execute.call_count, 0)
    self.assertEqual(cur.close.call_count, 1)
    self.assertEqual(cnx.commit.call_count, 0)
    self.assertEqual(cnx.close.call_count, 1)

  def test_real_mode(self):
    """Close with commit."""

    connector = MagicMock()
    with DelphiDatabase(connector, False, 'u', 'p', 'd') as db:
      pass

    self.assertEqual(connector.connect.call_count, 1)
    args, kwargs = connector.connect.call_args
    self.assertEqual(kwargs, {'user': 'u', 'password': 'p', 'database': 'd'})
    cnx = connector.connect()
    self.assertEqual(cnx.cursor.call_count, 1)
    cur = cnx.cursor()
    self.assertEqual(cur.execute.call_count, 0)
    self.assertEqual(cur.close.call_count, 1)
    self.assertEqual(cnx.commit.call_count, 1)
    self.assertEqual(cnx.close.call_count, 1)

  def test_execute(self):
    """Execute a SQL statement with arguments."""

    connector = MagicMock()
    with DelphiDatabase(connector, False, 'u', 'p', 'd') as db:
      db.execute('sql', 'args')

    cur = connector.connect().cursor()
    self.assertEqual(cur.execute.call_count, 1)
    args, kwargs = cur.execute.call_args
    sql, args = args
    self.assertEqual(sql, 'sql')
    self.assertEqual(args, 'args')

  def test_table_context_manager(self):
    """Create and use a table's context manager."""

    class FakeTable(DelphiDatabase.Table):

      def do_something(self):
        self._database.execute('sql', 'args')

      def _get_connection_info(self):
        return 'u', 'p', 'd'

    connector = MagicMock()
    with FakeTable(connector=connector, test_mode=False) as table:
      table.do_something()

    self.assertEqual(connector.connect.call_count, 1)
    args, kwargs = connector.connect.call_args
    self.assertEqual(kwargs, {'user': 'u', 'password': 'p', 'database': 'd'})
    cnx = connector.connect()
    self.assertEqual(cnx.cursor.call_count, 1)
    cur = cnx.cursor()
    self.assertEqual(cur.execute.call_count, 1)
    args, kwargs = cur.execute.call_args
    sql, args = args
    self.assertEqual(sql, 'sql')
    self.assertEqual(args, 'args')
    self.assertEqual(cur.close.call_count, 1)
    self.assertEqual(cnx.commit.call_count, 1)
    self.assertEqual(cnx.close.call_count, 1)
