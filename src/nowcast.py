"""
===============
=== Purpose ===
===============

Nowcasting! Currently doing sensor fusion using eight digital surveillance
sources:
  - gft  (62) | gft-nat,gft-hhs,gft-state
  - ght  (52) | ght-nat,ght-state
  - twtr (71) | twtr-nat,twtr-hhs,twtr-cen,twtr-state
  - wiki (1)  | wiki-nat
  - cdc  (1)  | cdc-nat
  - epic (11) | epic-nat,epic-hhs
  - sar3 (20) | sar3-nat,sar3-hhs,sar3-cen
  - arch (20) | arch-nat,arch-hhs,arch-cen

Note that regions hhs1 and cen1 contain the same states, so their signals
*should* be equal. For this reason, cen1 is omitted when nowcasting; whenever
cen1 is available, hhs1 is also available. The nowcast does however produce an
estimate (identically) for both cen1 and hhs1.

To update sensors, run:
  python3 sensor_update.py <sensor name-loc>
For example, update wiki nationally and twtr at all locations:
  python3 sensor_update.py wiki-nat,twtr-nat,twtr-hhs,twtr-cen,twtr-state

See also: state_info.py, fusion.py, sensor_update.py


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


=================
=== Changelog ===
=================

2016-12-13
  + use secrets
2016-11-28
  + store timestamp of last update (this undoubtedly takes the world record for
    ugliest hack ever written)
2016-06-03
  * updated precision matrix estimation `b` parameter
2016-04-27
  * assume that sensors are pre-centered (zero mean)
2016-04-15
  + experimental prefetch/cache for faster nowcasting
2016-04-14
  + gft state signal
2016-04-11
  + work in progess: use cached data for a significant speedup
  * more useful output and more documentation
  * fixed unstable wILI lookup (off by one)
2016-04-09
  + first "real" version (based on various PoC's)
  * stole the `nowcast` table from old/nowcast_20160409.py
"""

# built-in
import argparse
import sys
import time
# external
import mysql.connector
import numpy as np
# local
from delphi_epidata import Epidata
from epidate import EpiDate
import fluv_utils as flu
from fusion import Fusion
import secrets
from state_info import StateInfo


def get_most_recent_issue():
  # search for FluView issues within the last 10 weeks
  ew2 = EpiDate.today().get_ew()
  ew1 = flu.add_epiweeks(ew2, -9)
  rows = Epidata.check(Epidata.fluview('nat', Epidata.range(ew1, ew2)))
  return max([row['issue'] for row in rows])


def get_all_sensors():
  si = StateInfo()
  all_names = ['gft', 'ght', 'twtr', 'wiki', 'cdc', 'epic', 'sar3', 'arch']
  all_loc = si.nat + si.hhs + si.cen[1:] + si.sta
  return (all_names, all_loc)


def nowcast(epiweek, epidata_cache=None):
  si = StateInfo()
  # all sensors and locations
  all_names, all_loc = get_all_sensors()
  # get sensors available on the target week
  rows = Epidata.check(Epidata.sensors(secrets.api.sensors, all_names, all_loc, epiweek))
  present = {}
  for row in rows:
    name, loc, value = row['name'], row['location'], row['value']
    if name not in present:
      present[name] = {}
    if loc not in present[name]:
      present[name][loc] = value
  # get the history of each available sensor (6 sec)
  past = {}
  sensor_locs = set()
  missing = set()
  past_weeks = Epidata.range(200950, flu.add_epiweeks(epiweek, -1))
  all_epiweeks = [w for w in flu.range_epiweeks(past_weeks['from'], past_weeks['to'], inclusive=True)]
  num_obs = len(all_epiweeks)
  for name in present.keys():
    past[name] = {}
    for loc in present[name].keys():
      past[name][loc] = {}
      sensor_locs |= set([loc])
      #print(name, loc)
      try:
        if epidata_cache is not None:
          rows = epidata_cache.sensors(name, loc, past_weeks)
        else:
          rows = Epidata.check(Epidata.sensors(secrets.api.sensors, name, loc, past_weeks))
        if len(rows) < 2:
          raise Exception()
        for row in rows:
          past[name][loc][row['epiweek']] = row['value']
      except:
        missing |= set([(name, loc)])
  # remove sensors with zero past data
  for (n, l) in missing:
    del present[n][l]
    if len(present[n]) == 0:
      del present[n]
    del past[n][l]
    if len(past[n]) == 0:
      del past[n]
    #print(n, l, 'is missing')
  # inventory
  all_sensors = []
  for n in all_names:
    for l in si.nat + si.hhs + si.cen + si.sta:
      if n in past and l in past[n]:
        all_sensors.append((n, l))
  #print(all_sensors)
  num_sensors = len(all_sensors)
  # get historical ground truth for each sensor (4 sec)
  truth = {}
  for loc in sensor_locs:
    truth[loc] = {}
    if loc in si.sta:
      if epidata_cache is not None:
        rows = epidata_cache.stateili(loc, past_weeks)
      else:
        rows = Epidata.check(Epidata.stateili(secrets.api.stateili, loc, past_weeks))
      field = 'ili'
    else:
      if epidata_cache is not None:
        srows = epidata_cache.fluview(loc, past_weeks)
      else:
        srows = Epidata.check(Epidata.fluview(loc, past_weeks))
      sdata = dict([(r['epiweek'], r) for r in srows])
      udata = {}
      try:
        urows = Epidata.check(Epidata.fluview(loc, past_weeks, issues=past_weeks['to']))
        udata = dict([(r['epiweek'], r) for r in urows])
      except:
        pass
      rows = []
      for ew in all_epiweeks:
        if ew in udata:
          rows.append(udata[ew])
        else:
          rows.append(sdata[ew])
      field = 'wili'
    for row in rows:
      truth[loc][row['epiweek']] = row[field]
  # rows are epiweeks, cols are sensors
  X = np.zeros((num_obs, num_sensors)) * np.nan
  for (r, ew) in enumerate(all_epiweeks):
    for (c, (name, loc)) in enumerate(all_sensors):
      if name in past and loc in past[name] and ew in past[name][loc] and loc in truth and ew in truth[loc]:
        X[r, c] = past[name][loc][ew] - truth[loc][ew]
  # sparse precision matrix
  Ri = Fusion.precision(X, mean=np.zeros((1, num_sensors)), b=0.25)
  # prepare for sensor fusion
  inputs = all_sensors
  state = si.sta
  outputs = si.nat + si.hhs + si.cen + si.sta
  num_i, num_s, num_o = len(inputs), len(state), len(outputs)
  # input  (z): [ num_i  x    1   ]
  # state  (x): [ num_s  x    1   ]
  # output (y): [ num_o  x    1   ]
  # S->I   (H): [ num_i  x  num_s ]
  # S->O   (W): [ num_o  x  num_s ]
  z = np.array([present[n][l] for (n, l) in inputs]).reshape((num_i, 1))
  H = np.zeros((num_i, num_s))
  W = np.zeros((num_o, num_s))
  # populate H, given input signals
  for (row, (name, location)) in enumerate(inputs):
    for (col, loc) in enumerate(state):
      if loc in si.within[location]:
        H[row, col] = si.weight[location][loc]
  if np.linalg.matrix_rank(np.dot(H.T, H)) != num_s:
    raise Exception('H is singluar')
  if not np.allclose(np.sum(H, axis=1), 1):
    raise Exception('H rows do not sum to 1')
  # populate W, given output locations
  for (row, location) in enumerate(outputs):
    for (col, loc) in enumerate(state):
      if loc in si.within[location]:
        W[row, col] = si.weight[location][loc]
  if not np.allclose(np.sum(W, axis=1), 1):
    raise Exception('W rows do not sum to 1')
  # sensor fusion
  x, P = Fusion.fuse(z, Ri, H)
  y, S = Fusion.extract(x, P, W)
  print(num_obs, num_i, num_s, num_o)
  pt = [float(v) for v in y.flatten()]
  std = [float(v) for v in np.sqrt(S).flatten()]
  return (outputs, pt, std)


def update(ew1, ew2, test_mode=False, epidata_cache=None):
  # database setup
  u, p = secrets.db.epi
  cnx = mysql.connector.connect(user=u, password=p, database='epidata')
  cur = cnx.cursor()
  sql = """
    INSERT INTO `nowcasts`
      (`epiweek`, `location`, `value`, `std`)
    VALUES
      (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      value = %s, std = %s
  """

  for ew in flu.range_epiweeks(ew1, ew2, inclusive=True):
    try:
      print(ew)
      locations, values, stds = nowcast(ew, epidata_cache)
      print(' ', locations[0], values[0], stds[0])
      for (l, v, s) in zip(locations, values, stds):
        cur.execute(sql, (ew, l, v, s, v, s))
    except Exception as ex:
      print('failed: ', ew, ex)
      #raise ex
    sys.stdout.flush()

  # the Ugliest Hack Ever Written lies below. turn back now, cannot be unseen.
  # please fix me
  # store the unix timestamp in a meta row representing the last update time
  # the key to this row is `epiweek`=0, `location`='updated'
  # the timestamp is stored across the `value` and `std` fields
  # these are 32-bit floats, so precision is limited (hence, using both fields)
  t = round(time.time())
  a, b = t // 100000, t % 100000
  cur.execute(sql, (0, 'updated', a, b, a, b))
  # /hack

  # database cleanup
  cur.close()
  if test_mode:
    print('test mode - nowcasts not saved')
  else:
    cnx.commit()
  cnx.close()


class Cache:

  def __init__(self, ew2):
    ew1 = 200950
    print('prefetching %d--%d...' % (ew1, ew2))
    weeks = Epidata.range(ew1, ew2)
    si = StateInfo()
    all_names, all_loc = get_all_sensors()
    self._sensors = {}
    self._stateili = {}
    self._fluview = {}
    na, nb, nc = 0, 0, 0
    for loc in all_loc:
      for name in all_names:
        res = Epidata.sensors(secrets.api.sensors, name, loc, weeks)
        if res['result'] == 1:
          for row in res['epidata']:
            n, l = row['name'], row['location']
            if n not in self._sensors:
              self._sensors[n] = {}
            if l not in self._sensors[n]:
              self._sensors[n][l] = []
            self._sensors[n][l].append(row)
            na += 1
      if loc in si.sta:
        res = Epidata.stateili(secrets.api.stateili, loc, weeks)
        if res['result'] == 1:
          for row in res['epidata']:
            l = row['state']
            if l not in self._stateili:
              self._stateili[l] = []
            self._stateili[l].append(row)
            nb += 1
      else:
        res = Epidata.fluview(loc, weeks)
        if res['result'] == 1:
          for row in res['epidata']:
            l = row['region']
            if l not in self._fluview:
              self._fluview[l] = []
            self._fluview[l].append(row)
            nc += 1
    print('done (%d|%d|%d)' % (na, nb, nc))

  def sensors(self, name, loc, past_weeks):
    ew1, ew2 = past_weeks['from'], past_weeks['to']
    if name in self._sensors and loc in self._sensors[name]:
      return [r for r in self._sensors[name][loc] if ew1 <= r['epiweek'] <= ew2]
    else:
      return []

  def stateili(self, loc, past_weeks):
    ew1, ew2 = past_weeks['from'], past_weeks['to']
    if loc in self._stateili:
      return [r for r in self._stateili[loc] if ew1 <= r['epiweek'] <= ew2]
    else:
      return []

  def fluview(self, loc, past_weeks):
    ew1, ew2 = past_weeks['from'], past_weeks['to']
    if loc in self._fluview:
      return [r for r in self._fluview[loc] if ew1 <= r['epiweek'] <= ew2]
    else:
      return []


if __name__ == '__main__':
  # args and usage
  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--epiweek', default=None, type=int, help='epiweek override')
  parser.add_argument('-f', '--first', default=None, type=int, help='epiweek range first (requires --last)')
  parser.add_argument('-l', '--last', default=None, type=int, help='epiweek range last (requires --first)')
  parser.add_argument('-t', '--test', default=False, action='store_true', help='test mode, nothing is stored')
  args = parser.parse_args()

  # sanity checks
  if (args.first is None) ^ (args.last is None):
    raise Exception('--first and --last must be used together')
  if args.epiweek is not None and args.first is not None:
    raise Exception('epiweek and range both given')
  if args.first is not None and args.last is not None and args.first > args.last:
    raise Exception('first > last')

  # check input
  for ew in [args.epiweek, args.first, args.last]:
    if ew is not None:
      flu.check_epiweek(ew)

  # figure out which weeks to update
  epidata_cache = None
  if args.first is not None and args.last is not None:
    ew1, ew2 = args.first, args.last
    epidata_cache = Cache(ew2)
  else:
    epiweek = args.epiweek
    if epiweek is None:
      epiweek = flu.add_epiweeks(get_most_recent_issue(), +1)
      print('assuming nowcast for next week: %d' % epiweek)
    ew1, ew2 = epiweek, epiweek

  # make nowcast for each week
  update(ew1, ew2, args.test, epidata_cache)
