"""
===============
=== Purpose ===
===============

Produces a signal for each flu digital surveillance source, which is then used
as a 'sensor' in the context of nowcasting through sensor fusion.

Each signal is updated over the following inclusive range of epiweeks:
  - epiweek of most recently computed signal of this type
  - last epiweek
The idea is to recompute the last stored value (just in case there were
changes to the underlying data source), and to compute all weeks up to, but
not including, the current week (because the current week is, by definition,
still ongoing).

The following signals are available:
  - gft: Google Flu Trends
  - ght: Google Health Trends
  - twtr: HealthTweets
  - wiki: Wikipedia access
  - cdc: CDC Page Hits
  - epic: Epicast 1-week-ahead point prediction
  - quid: Flu lab test data
  - sar3: Seasonal Autoregression (order 3)
  - arch: Best-fit Archetype at 1-week-ahead

See also:
  - signal_update.py
  - sar3.py
  - arch.py


=======================
=== Data Dictionary ===
=======================

`sensors` is the table where the data is stored.
+----------+------------+------+-----+---------+----------------+
| Field    | Type       | Null | Key | Default | Extra          |
+----------+------------+------+-----+---------+----------------+
| id       | int(11)    | NO   | PRI | NULL    | auto_increment |
| name     | varchar(8) | NO   | MUL | NULL    |                |
| epiweek  | int(11)    | NO   | MUL | NULL    |                |
| location | varchar(8) | NO   | MUL | NULL    |                |
| value    | float      | NO   |     | NULL    |                |
+----------+------------+------+-----+---------+----------------+
id: unique identifier for each record
name: the name of the signal (ex: 'wiki')
epiweek: the epiweek during which the data was collected
location: where the data was collected (see below)
value: the estimated final value of (w)ILI on this epiweek

Locations vary by data source, but include all of:
  - 'nat' (U.S. National): 1
  - 'hhs[1-10]' (HHS regions): 10
  - 'cen[1-9]' (Census regions): 9
  - '[two-letter state]' (U.S. states and DC): 51


=================
=== Changelog ===
=================
2017-12-15
  + add `quid` data source
2017-01-31
  + updated `wiki` to use the same fitting procedure as everything else
2016-12-13
  + use secrets
2016-04-22
  + include periodic bias when signal is observed at least a year
  * fit pre/post 2013w40 `gft` separately
  * replace `cdc` sqrt-transform with log-transform
2016-04-20
  * reworked the weight function to (hopefully) reduce bias
2016-04-18
  * don't use `num1` and `num3` for `cdc`
  * sqrt-transform `cdc` before fitting
2016-04-16
  + added --epiweek argument
  * from single to multiple regression (for `cdc`)
2016-04-14
  * work with missing state ili
  * ght at state level
2016-04-11
  + switch for valid/invalid mode (whether to force use of unstable wILI)
  + flush output for use with `tee`
  * aesthetic edits
2016-04-09
  * fixed Exception message formatting
  * impute missing twtr values as 0%
2016-04-08
  + finished implementing all sensors
2016-04-07
  + initial version (based heavily on signal_update.py)
"""

# standard library
import argparse
import sys
import subprocess

# third party
import mysql.connector
import numpy as np

# first party
from .arch import ARCH
from .sar3 import SAR3
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


def get_last_update(cur, name, location):
  # find the last epiweek for which this signal is available
  sql = '''
    SELECT
      max(`epiweek`)
    FROM
      `sensors`
    WHERE
      `name` = %s AND `location` = %s
  '''
  args = (name, location)
  cur.execute(sql, args)
  epiweek = None
  for (epiweek,) in cur:
    pass
  if epiweek is None:
    raise Exception('%s-%s does not exist yet' % (name, location))
  return epiweek


def store_value(cur, name, location, epiweek, value):
  # find the last epiweek for which this signal is available
  sql = '''
    INSERT INTO
      `sensors` (`name`, `location`, `epiweek`, `value`)
    VALUES
      (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      `value` = %s
  '''
  args = (name, location, epiweek, value, value)
  cur.execute(sql, args)


def dot(*Ms):
  N = Ms[0]
  for M in Ms[1:]:
    N = np.dot(N, M)
  return N


def get_weight(ew1, ew2):
  # I want something that:
  #   - drops sharply over the most recent ~3 weeks
  #   - falls off exponentially with time
  #   - puts extra emphasis on the past weeks at the same time of year
  #   - gives no week a weight of zero
  dw = flu.delta_epiweeks(ew1, ew2)
  yr = 52.2
  hl1, hl2, bw = yr, 1, 4
  a = 0.05
  #b = (np.cos(2 * np.pi * (dw / yr)) + 1) / 2
  b = np.exp(-((min(dw % yr, yr - dw % yr) / bw) ** 2))
  c = 2 ** -(dw / hl1)
  d = 1 - 2 ** -(dw / hl2)
  return (a + (1 - a) * b) * c * d


def get_periodic_bias(epiweek):
  weeks_per_year = 52.2
  offset = flu.delta_epiweeks(200001, epiweek) % weeks_per_year
  angle = np.pi * 2 * offset / weeks_per_year
  return [np.sin(angle), np.cos(angle)]


def get_model(ew2, epiweeks, X, Y):
  ne, nx1, nx2, ny = len(epiweeks), len(X), len(X[0]), len(Y)
  if ne != nx1 or nx1 != ny:
    raise Exception('length mismatch e=%d X=%d Y=%d' % (ne, nx1, ny))
  weights = np.diag([get_weight(ew1, ew2) for ew1 in epiweeks])
  X = np.array(X).reshape((nx1, nx2))
  Y = np.array(Y).reshape((ny, 1))
  bias0 = np.ones(Y.shape)
  if ne >= 26 and flu.delta_epiweeks(epiweeks[0], epiweeks[-1]) >= 52:
    # constant and periodic bias
    bias1 = np.array([get_periodic_bias(ew) for ew in epiweeks])
    X = np.hstack((X, bias0, bias1))
  else:
    # constant bias only
    X = np.hstack((X, bias0))
  XtXi = np.linalg.inv(dot(X.T, weights, X))
  XtY = dot(X.T, weights, Y)
  return np.dot(XtXi, XtY)


def apply_model(epiweek, beta, values):
  bias0 = [1.]
  if beta.shape[0] > len(values) + 1:
    # constant and periodic bias
    bias1 = get_periodic_bias(epiweek)
    obs = np.array([values + bias0 + bias1])
  else:
    # constant bias only
    obs = np.array([values + bias0])
  return float(dot(obs, beta))


def extract(rows, fields):
  data = {}
  for row in rows:
    data[row['epiweek']] = [float(row[f]) for f in fields]
  return data


def get_weeks(epiweek):
  ew1 = 200330
  ew2 = epiweek
  ew3 = flu.add_epiweeks(epiweek, 1)
  weeks0 = Epidata.range(ew1, ew2)
  weeks1 = Epidata.range(ew1, ew3)
  return (ew1, ew2, ew3, weeks0, weeks1)


def get_training_set_data(data):
  epiweeks = sorted(list(data.keys()))
  X = [data[ew]['x'] for ew in epiweeks]
  Y = [data[ew]['y'] for ew in epiweeks]
  return (epiweeks, X, Y)


def get_training_set_ili(location, epiweek, signal, valid):
  #if valid:
  #  raise Exception('state ili is stable')
  ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
  rows = Epidata.check(Epidata.stateili(secrets.api.stateili, location, weeks0))
  stable = extract(rows, ['ili'])
  data = {}
  for ew in signal.keys():
    if ew == ew3:
      continue
    if ew not in stable:
      #raise Exception('missing state ili on %d' % ew)
      continue
    sig = signal[ew]
    ili = stable[ew]
    data[ew] = {'x': sig, 'y': ili}
  return get_training_set_data(data)


def get_training_set_wili(location, epiweek, signal, valid):
  ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
  try:
    rows = Epidata.check(Epidata.fluview(location, weeks0, issues=ew2))
    unstable = extract(rows, ['wili'])
  except:
    unstable = {}
  rows = Epidata.check(Epidata.fluview(location, weeks0))
  stable = extract(rows, ['wili'])
  data = {}
  for ew in signal.keys():
    if ew == ew3:
      continue
    sig = signal[ew]
    if ew not in unstable:
      if valid and flu.delta_epiweeks(ew, ew3) <= 5:
        raise Exception('unstable wILI is not available on %d' % ew)
      if ew not in stable:
        raise Exception('wILI (any) is not available on %d' % ew)
      wili = stable[ew]
    else:
      wili = unstable[ew]
    data[ew] = {'x': sig, 'y': wili}
  return get_training_set_data(data)


def get_training_set(location, epiweek, signal, valid):
  if len(location) == 2:
    # state
    return get_training_set_ili(location, epiweek, signal, valid)
  else:
    # national or regional
    return get_training_set_wili(location, epiweek, signal, valid)


def get_prediction(location, epiweek, name, fields, fetch, valid):
  if type(fields) == str:
    fields = [fields]
  ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
  rows = Epidata.check(fetch(weeks1))
  signal = extract(rows, fields)
  min_rows = 3 + len(fields)
  if ew3 not in signal:
    raise Exception('%s unavailable on %d' % (name, ew3))
  if len(signal) < min_rows:
    raise Exception('%s available less than %d weeks' % (name, min_rows))
  epiweeks, X, Y = get_training_set(location, epiweek, signal, valid)
  min_rows = min_rows - 1
  if len(Y) < min_rows:
    raise Exception('(w)ILI available less than %d weeks' % (min_rows))
  model = get_model(ew3, epiweeks, X, Y)
  value = apply_model(ew3, model, signal[ew3])
  return value


def get_gft(location, epiweek, valid):
  def fetch(weeks):
    # The GFT model update of 2013 significantly improved the GFT signal, so
    # much so that training on the old data will severely hurt the predictive
    # power of the new data. To overcome this, I basically pretend that GFT
    # versions before and after mid-2013 are different signals.
    if weeks['to'] >= 201340:
      # this is the new GFT model, so throw out data from the old model
      weeks = Epidata.range(max(weeks['from'], 201331), weeks['to'])
    return Epidata.gft(location, weeks)
  return get_prediction(location, epiweek, 'gft', 'num', fetch, valid)


def get_ght(location, epiweek, valid):
  loc = 'US' if location == 'nat' else location
  fetch = lambda weeks: Epidata.ght(secrets.api.ght, loc, weeks, '/m/0cycc')
  return get_prediction(location, epiweek, 'ght', 'value', fetch, valid)


def get_ghtj(location, epiweek, valid):
  loc = 'US' if location == 'nat' else location
  def justinfun(location, epiweek):
    main_driver = 'ghtj.R'   ### Need to set an absolute path
    subprocess.check_call(['Rscript', main_driver, location, epiweek], shell=False)
    outputdir = '/home/justin/repos/ghtModel/output/' ### Need to set an absolute path
    prefix = 'ghtpred-'
    predfilename = outputdir + prefix + '-'+ location +'-' + epiweek + '.txt'
    file = open(outputdir+prefix+epiweek+'.txt', 'r')
    mypred = file.read()
    return mypred

  # Making the single prediction now:
  mypred = justinfun(location, epiweek)
  mypred = 1
  return mypred



def get_twtr(location, epiweek, valid):
  def fetch(weeks):
    # Impute missing weeks with 0%
    # This is actually correct because twitter does not store rows with `num` =
    # 0. So weeks with 0 `num` (and `percent`) are missing from the response.
    res = Epidata.twitter(secrets.api.twitter, location, epiweeks=weeks)
    if 'epidata' in res:
      epiweeks = set([r['epiweek'] for r in res['epidata']])
      first, last = 201149, weeks['to']
      for ew in flu.range_epiweeks(first, last, inclusive=True):
        if ew not in epiweeks:
          res['epidata'].append({'epiweek': ew, 'percent': 0.})
    return res
  return get_prediction(location, epiweek, 'twtr', 'percent', fetch, valid)


def get_wiki(location, epiweek, valid):
  if location != 'nat':
    raise Exception('wiki is only available for nat')
  articles = ['human_flu', 'influenza', 'influenza_a_virus', 'influenzavirus_a', 'influenzavirus_c', 'oseltamivir', 'zanamivir']
  hours = [17, 18, 21]
  # There are 21 time series (7 articles, 3 hours) of N epiweeks. Each time
  # series needs to be fetched, and then the whole dataset needs to be pivoted
  # so that there are N rows, each with 21 values.
  fields = ['f%d' % i for i in range(len(articles) * len(hours))]
  def fetch(weeks):
    # a map from epiweeks to a map of field-value pairs (for each article/hour)
    data = {}
    # field name index
    idx = 0
    # download each time series individually
    for article in articles:
      for hour in hours:
        # fetch the data from the API
        res = Epidata.wiki(article, epiweeks=weeks, hours=hour)
        epidata = Epidata.check(res)
        field_name = fields[idx]
        idx += 1
        # loop over rows of the response, ordered by epiweek
        for row in epidata:
          ew = row['epiweek']
          if ew not in data:
            # make a new entry for this epiweek
            data[ew] = {'epiweek': ew}
          # save the value of this field
          data[ew][field_name] = row['value']
    # convert the map to a list matching the API epidata list
    rows = []
    for ew in sorted(list(data.keys())):
      rows.append(data[ew])
    # spoof the API response
    return {
      'result': 1,
      'message': None,
      'epidata': rows,
    }
  return get_prediction(location, epiweek, 'wiki', fields, fetch, valid)


def get_cdc(location, epiweek, valid):
  fields = ['num2', 'num4', 'num5', 'num6', 'num7', 'num8']
  def fetch(weeks):
    # It appears that log-transformed counts provide a much better fit.
    res = Epidata.cdc(secrets.api.cdc, weeks, location)
    if 'epidata' in res:
      for row in res['epidata']:
        for col in fields:
          row[col] = np.log(1. + row[col])
    return res
  return get_prediction(location, epiweek, 'cdc', fields, fetch, valid)


def get_epic(location, epiweek, valid):
  fc = Epidata.check(Epidata.delphi('ec', epiweek))[0]
  return fc['forecast']['data'][location]['x1']['point']


def get_quid(location, epiweek, valid):
  fields = ['value']
  def fetch(weeks):
    res = Epidata.quidel(secrets.api.quidel, weeks, location)
    return res
  return get_prediction(location, epiweek, 'quid', fields, fetch, valid)


def get_sar3(location, epiweek, valid):
  return SAR3(location).predict(epiweek, valid=valid)


def get_arch(location, epiweek, valid):
  return ARCH(location).predict(epiweek, valid=valid)


def update(sensors, first_week=None, last_week=None, valid=False, test_mode=False):
  # most recent issue
  last_issue = get_most_recent_issue()

  # location information
  loc_info = StateInfo()

  # connect
  u, p = secrets.db.epi
  cnx = mysql.connector.connect(user=u, password=p, database='epidata')
  cur = cnx.cursor()

  # update each sensor
  for (name, loc) in sensors:
    if loc == 'hhs':
      locations = loc_info.hhs
    elif loc == 'cen':
      locations = loc_info.cen
    elif loc == 'state' or loc == 'sta':
      locations = loc_info.sta
    else:
      locations = [loc]
    # update each location
    print(locations)
    for location in locations:
      # timing
      ew1, ew2 = first_week, last_week
      if ew1 is None:
        ew1 = get_last_update(cur, name, location)
      if ew2 is None:
        ew2 = flu.add_epiweeks(last_issue, +1)
      print('Updating %s-%s from %d to %d.' % (name, location, ew1, ew2))
      for test_week in flu.range_epiweeks(ew1, ew2, inclusive=True):
        train_week = flu.add_epiweeks(test_week, -1)
        try:
          value = {
            'gft': get_gft,
            'ght': get_ght,
            'ghtj': get_ghtj,
            'twtr': get_twtr,
            'wiki': get_wiki,
            'cdc': get_cdc,
            'epic': get_epic,
            'sar3': get_sar3,
            'arch': get_arch,
            'quid': get_quid,
          }[name](location, train_week, valid)
          print(' %4s %5s %d -> %.3f' % (name, location, test_week, value))
          # upload
          store_value(cur, name, location, test_week, value)
        except Exception as ex:
          print(' failed: %4s %5s %d' % (name, location, test_week), ex)
          #raise ex
        sys.stdout.flush()

  # disconnect
  cur.close()
  if not test_mode:
    cnx.commit()
  cnx.close()


if __name__ == '__main__':
  # args and usage
  parser = argparse.ArgumentParser()
  parser.add_argument('names', type=str, help='list of name-location pairs (location can be nat/hhs/cen/state or specific location labels)')
  parser.add_argument('--first', '-f', default=None, type=int, help='first epiweek override')
  parser.add_argument('--last', '-l', default=None, type=int, help='last epiweek override')
  parser.add_argument('--epiweek', '-w', default=None, type=int, help='epiweek override')
  parser.add_argument('--test', '-t', default=False, action='store_true', help='dry run only')
  parser.add_argument('--valid', '-v', default=False, action='store_true', help='do not fall back to stable wILI; require unstable wILI')
  args = parser.parse_args()

  # sanity check
  first, last, week = args.first, args.last, args.epiweek
  for ew in [first, last, week]:
    if ew is not None:
      flu.check_epiweek(ew)
  if first is not None and last is not None and first > last:
    raise Exception('epiweeks in the wrong order')
  if week is not None:
    first = last = week

  # extract name-location pairs
  sensors = [pair.split('-') for pair in args.names.split(',')]

  # update the requested sensors
  print(sensors, first, last, args.valid, args.test, sep=" ")
  update(sensors, first, last, args.valid, args.test)
