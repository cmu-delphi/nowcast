"""
===============
=== Purpose ===
===============

A simple wrapper for the `nowcasts` table in the Delphi database.


=======================
=== Data Dictionary ===
=======================

Nowcasts (value and standard deviation) are stored in the `nowcasts` table.
+----------+------------+------+-----+---------+----------------+
| Field    | Type       | Null | Key | Default | Extra          |
+----------+------------+------+-----+---------+----------------+
| id       | int(11)    | NO   | PRI | NULL    | auto_increment |
| epiweek  | int(11)    | NO   | MUL | NULL    |                |
| location | varchar(8) | NO   | MUL | NULL    |                |
| value    | float      | NO   |     | NULL    |                |
| std      | float      | NO   |     | NULL    |                |
+----------+------------+------+-----+---------+----------------+
id: unique identifier for each record
epiweek: the epiweek for which (w)ILI is being predicted
location: where the data was collected (nat, hhs, cen, and states)
value: nowcast point prediction
std: nowcast standard deviation
"""

# standard library
import time

# first party
from delphi.nowcast.util.delphi_database import DelphiDatabase
from delphi.operations import secrets


class NowcastsTable(DelphiDatabase.Table):
  """A database wrapper for the `nowcasts` table."""

  SQL_INSERT = '''
    INSERT INTO `nowcasts`
      (`epiweek`, `location`, `value`, `std`)
    VALUES
      (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      value = %s, std = %s
  '''

  def insert(self, epiweek, location, value, stdev):
    """
    Add a new nowcast record to the database, or update an existing record with
    the same key.
    """
    args = (epiweek, location, value, stdev, value, stdev)
    self._database.execute(NowcastsTable.SQL_INSERT, args)

  def set_last_update_time(self):
    """
    Store the timestamp of the most recent nowcast update.

    This hack was copied from the old nowcast.py, which has this to say:
    > Store the unix timestamp in a meta row representing the last update time.
    > The key to this row is `epiweek`=0, `location`='updated'. The timestamp
    > is stored across the `value` and `std` fields. These are 32-bit floats,
    > so precision is limited (hence, using both fields).
    """
    t = round(time.time())
    a, b = t // 100000, t % 100000
    self.insert(0, 'updated', a, b)

  def _get_connection_info(self):
    """Return username, password, and database name."""
    return secrets.db.epi + ('epidata',)
