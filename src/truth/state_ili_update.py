"""
===============
=== Purpose ===
===============

Stores imputed state ILI in the database.

See also: state_info.py


=======================
=== Data Dictionary ===
=======================

`state_ili_imputed` is the table where the data is stored.
+---------+---------+------+-----+---------+----------------+
| Field   | Type    | Null | Key | Default | Extra          |
+---------+---------+------+-----+---------+----------------+
| id      | int(11) | NO   | PRI | NULL    | auto_increment |
| epiweek | int(11) | NO   | MUL | NULL    |                |
| state   | char(2) | NO   | MUL | NULL    |                |
| ili     | float   | NO   |     | NULL    |                |
+---------+---------+------+-----+---------+----------------+
id: unique identifier for each record
epiweek: the epiweek during which the data was collected
state: the state abbreviation (51 total, including DC)
ili: imputed ILI


=================
=== Changelog ===
=================

2016-12-13
  + use secrets
2016-04-06
  + initial version
"""

# standard library
import argparse

# third party
import mysql.connector

# first party
from delphi.epidata.client.delphi_epidata import Epidata
import delphi.operations.secrets as secrets
from delphi.utils.epidate import EpiDate
import delphi.utils.epiweek as flu
from delphi.utils.state_info import StateInfo


def get_most_recent_issue():
  # search for FluView issues within the last 10 weeks
  ew2 = EpiDate.today().get_ew()
  ew1 = flu.add_epiweeks(ew2, -9)
  rows = Epidata.check(Epidata.fluview('nat', Epidata.range(ew1, ew2)))
  return max([row['issue'] for row in rows])


def update(ew1, ew2, test_mode=False):
  # init
  si = StateInfo()
  sql = '''
    INSERT INTO
      `state_ili_imputed` (`epiweek`, `state`, `ili`)
    VALUES
      (%s, %s, %s)
    ON DUPLICATE KEY UPDATE
      `ili` = %s
  '''

  # connect
  u, p = secrets.db.epi
  cnx = mysql.connector.connect(user=u, password=p, database='epidata')
  cur = cnx.cursor()

  # get state ILI on each week
  for ew in flu.range_epiweeks(ew1, ew2, inclusive=True):
    print('epiweek:', ew)
    result = si.get_ili(ew)
    for state in si.sta:
      ili = result[state]
      if not (0 <= ili < 25):
        raise Exception('ILI for %s is %+.3f?' % (state, ili))
      print(' %s %.3f' % (state, ili))
      # upload
      if not test_mode:
        args = (ew, state, ili, ili)
        cur.execute(sql, args)

  # disconnect
  cur.close()
  cnx.commit()
  cnx.close()


if __name__ == '__main__':
  # args and usage
  parser = argparse.ArgumentParser()
  parser.add_argument('--first', '-f', default=None, type=int, help='first epiweek override')
  parser.add_argument('--last', '-l', default=None, type=int, help='last epiweek override')
  parser.add_argument('--test', '-t', default=False, action='store_true', help='dry run only')
  args = parser.parse_args()

  # epiweeks and timing
  first, last = None, None
  if args.first is not None:
    first = args.first
  if args.last is not None:
    last = args.last
  if last is None:
    last = get_most_recent_issue()
  if first is None:
    first = flu.add_epiweeks(last, -52)
  if last < first:
    raise Exception('epiweeks in the wrong order')
  flu.check_epiweek(first, last)
  print('Updating epiweeks from %d to %d.' % (first, last))

  # make it happen
  update(first, last, args.test)
