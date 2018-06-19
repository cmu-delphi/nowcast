"""
===============
=== Purpose ===
===============

A database wrapper for arbitrary Delphi tables.

This is intended to be subclassed on a per-table basis. Instances can be used
most conveniently in the `with` statement. Here's an example for the `nowcasts`
table (see nowcasts_table.py):

  with NowcastsTable() as table:
    table.insert(epiweek, location, value, stdev)

The actual database connection will be opened and closed automatically. The
transaction will be committed prior to closing unless `test_mode` (passed to
the constructor) is True.

For more complex database operations, like interacting with several tables at
once, a generic database connection can be made with the `with` statement. For
example:

  with DelphiDatabase(*params) as db:
    db.execute(sql, args)
"""

# first party
import abc

# third party
import mysql.connector


class DelphiDatabase:
  """A context manager for database connections."""

  class Table(abc.ABC):
    """A wrapper for individual Delphi tables."""

    def __init__(
        self, database=None, test_mode=False, connector=mysql.connector):
      self._database = database
      self.__test_mode = test_mode
      self.__connector = connector

    def __enter__(self):
      username, password, database = self._get_connection_info()
      self._database = DelphiDatabase(
          self.__connector, self.__test_mode, username, password, database)
      self._database.connect()
      return self

    @abc.abstractmethod
    def _get_connection_info(self):
      """Return username, password, and database name."""

    def __exit__(self, *error):
      self._database.disconnect()

  def __init__(self, connector, test_mode, username, password, database):
    self.__connector = connector
    self.__test_mode = test_mode
    self.__username = username
    self.__password = password
    self.__database = database

  def __enter__(self):
    self.connect()
    return self

  def __exit__(self, *error):
    self.disconnect()

  def connect(self):
    """Open a connection to the database."""
    self.__cnx = self.__connector.connect(
        user=self.__username,
        password=self.__password,
        database=self.__database)
    self.__cur = self.__cnx.cursor()

  def disconnect(self):
    """
    Close the connection to the database. Unless test mode is enabled,
    outstanding changes will be committed at this point.
    """
    self.__cur.close()
    if self.__test_mode:
      print('test mode: transaction not committed')
    else:
      self.__cnx.commit()
    self.__cnx.close()

  def execute(self, sql, args):
    """The database cursor."""
    return self.__cur.execute(sql, args)
