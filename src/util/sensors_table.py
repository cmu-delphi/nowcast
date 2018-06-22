"""
===============
=== Purpose ===
===============

A simple wrapper for the `sensors` table in the Delphi database.


=======================
=== Data Dictionary ===
=======================

`sensors` is the table where the data is stored.
+----------+-------------+------+-----+---------+----------------+
| Field    | Type        | Null | Key | Default | Extra          |
+----------+-------------+------+-----+---------+----------------+
| id       | int(11)     | NO   | PRI | NULL    | auto_increment |
| name     | varchar(8)  | NO   | MUL | NULL    |                |
| epiweek  | int(11)     | NO   | MUL | NULL    |                |
| location | varchar(12) | YES  | MUL | NULL    |                |
| value    | float       | NO   |     | NULL    |                |
+----------+-------------+------+-----+---------+----------------+
id: unique identifier for each record
name: the name of the signal (ex: 'wiki')
epiweek: the epiweek during which the data was collected
location: where the data was collected (see below)
value: the estimated final value of (w)ILI on this epiweek

Locations vary by data source, but include (at least) all of:
  - 'nat' (U.S. National): 1
  - 'hhs[1-10]' (HHS regions): 10
  - 'cen[1-9]' (Census regions): 9
  - '[two-letter state]' (U.S. states and DC): 51
"""

# first party
from delphi.nowcast.util.delphi_database import DelphiDatabase
from delphi.operations import secrets


class SensorsTable(DelphiDatabase.Table):
  """A database wrapper for the `sensors` table."""

  SQL_SELECT = '''
    SELECT
      max(`epiweek`)
    FROM
      `sensors`
    WHERE
      `name` = %s AND `location` = %s
  '''

  SQL_INSERT = '''
    INSERT INTO
      `sensors` (`name`, `location`, `epiweek`, `value`)
    VALUES
      (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      `value` = %s
  '''

  def insert(self, name, location, epiweek, value):
    """
    Add a new sensor reading to the database, or update an existing record with
    the same key.
    """
    args = (name, location, epiweek, value, value)
    self._database.execute(SensorsTable.SQL_INSERT, args)

  def get_most_recent_epiweek(self, name, location):
    """
    Return the epiweek of the most recent reading of a particular sensor and
    location. Returns None if no reading is found.
    """
    args = (name, location)
    cursor = self._database.execute(SensorsTable.SQL_SELECT, args)
    for (epiweek,) in cursor:
      return epiweek

  def _get_connection_info(self):
    """Return username, password, and database name."""
    return secrets.db.epi + ('epidata',)
