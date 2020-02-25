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
"""
# standard library
import argparse
import re
import subprocess
import sys

# third party
import numpy as np

# first party
from delphi.epidata.client.delphi_epidata import Epidata
from delphi.nowcast.fusion.nowcast import Nowcast
from delphi.nowcast.sensors.arch import ARCH
from delphi.nowcast.sensors.sar3 import SAR3
from delphi.nowcast.util.flu_data_source import FluDataSource
from delphi.nowcast.util.sensors_table import SensorsTable
import delphi.operations.secrets as secrets
from delphi.utils.epidate import EpiDate
import delphi.utils.epiweek as flu
from delphi.utils.geo.locations import Locations

def get_most_recent_issue(epidata):
  # search for FluView issues within the last 10 weeks
  ew2 = EpiDate.today().get_ew()
  ew1 = flu.add_epiweeks(ew2, -9)
  rows = epidata.check(epidata.fluview('nat', epidata.range(ew1, ew2)))
  return max([row['issue'] for row in rows])


def get_location_list(loc):
  """Return the list of locations described by the given string."""
  if loc == 'all':
    return Locations.region_list
  elif loc == 'hhs':
    return Locations.hhs_list
  elif loc == 'cen':
    return Locations.cen_list
  elif loc in Locations.region_list:
    return [loc]
  else:
    raise UnknownLocationException('unknown location: %s' % str(loc))


class UnknownLocationException(Exception):
  """An Exception indicating that the given location is not known."""


class SignalGetter:
  """Class with static methods that implement the fetching of
  different data signals. Each function returns a function that
  only takes a single argument:
  - weeks: an Epiweek range of weeks to fetch data for.
  """
  def __init__(self):
    pass

  @staticmethod
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
    return fetch

  @staticmethod
  def get_ght(location, epiweek, valid):
    loc = 'US' if location == 'nat' else location
    fetch = lambda weeks: Epidata.ght(secrets.api.ght, loc, weeks, '/m/0cycc')
    return fetch

  @staticmethod
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
    return fetch

  @staticmethod
  def get_wiki(location, epiweek, valid):
    if location != 'nat':
      raise Exception('wiki is only available for nat')
    articles = [
      'human_flu',
      'influenza',
      'influenza_a_virus',
      'influenzavirus_a',
      'influenzavirus_c',
      'oseltamivir',
      'zanamivir',
    ]
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

    return fetch, fields

  @staticmethod
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

    return fetch, fields

  @staticmethod
  def get_quid(location, epiweek, valid):
    fields = ['value']

    def fetch(weeks):
      res = Epidata.quidel(secrets.api.quidel, weeks, location)
      return res

    return fetch, fields


class SensorFitting:
  def __init__(self):
    pass

  @staticmethod
  def fit_twicing(location, epiweek, name, fields, fetch, valid):

    # Helper functions
    def get_weeks(epiweek):
      ew1 = 200330
      ew2 = epiweek
      ew3 = flu.add_epiweeks(epiweek, 1)
      weeks0 = Epidata.range(ew1, ew2)
      weeks1 = Epidata.range(ew1, ew3)
      return (ew1, ew2, ew3, weeks0, weeks1)

    def extract(rows, fields):
      data = {}
      for row in rows:
        data[row['epiweek']] = [float(row[f]) for f in fields]
      return data

    def get_training_set_data(data):
      epiweeks = sorted(list(data.keys()))
      X = [data[ew]['x'] for ew in epiweeks]
      Y = [data[ew]['y'] for ew in epiweeks]
      return (epiweeks, X, Y)

    def get_training_set(location, epiweek, signal, valid):
      ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
      auth = secrets.api.fluview
      try:
        result = Epidata.fluview(location, weeks0, issues=ew2, auth=auth)
        rows = Epidata.check(result)
        unstable = extract(rows, ['wili'])
      except Exception:
        unstable = {}
      rows = Epidata.check(Epidata.fluview(location, weeks0, auth=auth))
      stable = extract(rows, ['wili'])
      data = {}
      num_dropped = 0
      for ew in signal.keys():
        if ew == ew3:
          continue
        sig = signal[ew]
        if ew not in unstable:
          if valid and flu.delta_epiweeks(ew, ew3) <= 5:
            raise Exception('unstable wILI is not available on %d' % ew)
          if ew not in stable:
            num_dropped += 1
            continue
          wili = stable[ew]
        else:
          wili = unstable[ew]
        data[ew] = {'x': sig, 'y': wili}
      if num_dropped:
        msg = 'warning: dropped %d/%d signal weeks because (w)ILI was unavailable'
        print(msg % (num_dropped, len(signal)))
      return get_training_set_data(data)

    def apply_model(epiweek, beta, values):
      bias0 = [1.]
      # constant bias only
      obs = np.array([values + bias0])
      return float(obs @ beta)

    def get_model(ew2, epiweeks, X, Y):
      ne, nx1, nx2, ny = len(epiweeks), len(X), len(X[0]), len(Y)
      if ne != nx1 or nx1 != ny:
        raise Exception('length mismatch e=%d X=%d Y=%d' % (ne, nx1, ny))
      X = np.array(X).reshape((nx1, nx2))
      Y = np.array(Y).reshape((ny, 1))
      bias0 = np.ones(Y.shape)
      X = np.hstack((X, bias0)) # constant bias only
      XtXi = np.linalg.inv(X.T @ X)
      XtY = X.T @ Y
      return XtXi @ XtY

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

    # historical nowcasts
    rows = Epidata.check(Epidata.nowcast(location, weeks1))
    hist_nc = extract(rows, ['value'])
    nc_signal = []
    for ew in epiweeks:
      nc_signal.append(hist_nc[ew][0])

    # current nowcast
    maybe_curr_nc = Epidata.nowcast(location, ew3)
    curr_nc = None
    if 'epidata' in maybe_curr_nc:
      # use stored intermediate nowcast for this week
      curr_nc = [Epidata.check(maybe_curr_nc)[0]['value']]
    else:
      # produce intermediate nowcast for this week
      data_source = FluDataSource.new_instance()
      data_source.prefetch(ew3)
      nowcaster = Nowcast(data_source)
      curr_nc_all_loc = nowcaster.batch_nowcast([ew3])[0]
      for loc, val, std in curr_nc_all_loc:
        if loc == location:
          curr_nc = [val]

    assert curr_nc
    X = np.array(X)
    X = X - np.repeat(nc_signal, X.shape[1]).reshape(-1, X.shape[1])

    min_rows = min_rows - 1
    if len(Y) < min_rows:
      raise Exception('(w)ILI available less than %d weeks' % (min_rows))

    model = get_model(ew3, epiweeks, X, Y)
    value = apply_model(ew3, model, np.subtract(signal[ew3], curr_nc).tolist())
    return value

  @staticmethod
  def fit_loch_ness(location, epiweek, name, fields, fetch, valid):

    # Helper functions
    def get_weeks(epiweek):
      ew1 = 200330
      ew2 = epiweek
      ew3 = flu.add_epiweeks(epiweek, 1)
      weeks0 = Epidata.range(ew1, ew2)
      weeks1 = Epidata.range(ew1, ew3)
      return (ew1, ew2, ew3, weeks0, weeks1)

    def extract(rows, fields):
      data = {}
      for row in rows:
        data[row['epiweek']] = [float(row[f]) for f in fields]
      return data

    def get_training_set_data(data):
      epiweeks = sorted(list(data.keys()))
      X = [data[ew]['x'] for ew in epiweeks]
      Y = [data[ew]['y'] for ew in epiweeks]
      return (epiweeks, X, Y)

    def get_training_set(location, epiweek, signal, valid):
      ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
      auth = secrets.api.fluview
      try:
        result = Epidata.fluview(location, weeks0, issues=ew2, auth=auth)
        rows = Epidata.check(result)
        unstable = extract(rows, ['wili'])
      except Exception:
        unstable = {}
      rows = Epidata.check(Epidata.fluview(location, weeks0, auth=auth))
      stable = extract(rows, ['wili'])
      data = {}
      num_dropped = 0
      for ew in signal.keys():
        if ew == ew3:
          continue
        sig = signal[ew]
        if ew not in unstable:
          if valid and flu.delta_epiweeks(ew, ew3) <= 5:
            raise Exception('unstable wILI is not available on %d' % ew)
          if ew not in stable:
            num_dropped += 1
            continue
          wili = stable[ew]
        else:
          wili = unstable[ew]
        data[ew] = {'x': sig, 'y': wili}
      if num_dropped:
        msg = 'warning: dropped %d/%d signal weeks because (w)ILI was unavailable'
        print(msg % (num_dropped, len(signal)))
      return get_training_set_data(data)

    def dot(*Ms):
      """ Simple function to compute the dot product
      for any number of arguments.
      """
      N = Ms[0]
      for M in Ms[1:]:
        N = np.dot(N, M)
      return N

    def get_weight(ew1, ew2):
      """ This function gives the weight between two given
      epiweeks based on a function that:
        - drops sharply over the most recent ~3 weeks
        - falls off exponentially with time
        - puts extra emphasis on the past weeks at the
          same time of year (seasonality)
        - gives no week a weight of zero
      """
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


class SensorGetter:
  """Class that implements different sensors. Some sensors
  may take in a signal to do the fitting on, others do not.
  """
  def __init__(self):
    pass

  @staticmethod
  def get_sensor_implementations():
    """Return a map from sensor names to sensor implementations."""
    return {
      'cdc': SensorGetter.get_cdc,
      'gft': SensorGetter.get_gft,
      'ght': SensorGetter.get_ght,
      'ghtj': SensorGetter.get_ghtj,
      'twtr': SensorGetter.get_twtr,
      'wiki': SensorGetter.get_wiki,
      'epic': SensorGetter.get_epic,
      'sar3': SensorGetter.get_sar3,
      'arch': SensorGetter.get_arch,
      'quid': SensorGetter.get_quid,
    }

  @staticmethod
  def get_epic(location, epiweek, valid):
    fc = Epidata.check(Epidata.delphi('ec', epiweek))[0]
    return fc['forecast']['data'][location]['x1']['point']

  @staticmethod
  def get_sar3(location, epiweek, valid):
    return SAR3(location).predict(epiweek, valid=valid)

  @staticmethod
  def get_arch(location, epiweek, valid):
    return ARCH(location).predict(epiweek, valid=valid)

  @staticmethod
  def get_ghtj(location, epiweek, valid):
    loc = 'US' if location == 'nat' else location

    def justinfun(location, epiweek):
      # Need to set an absolute path
      main_driver = '/home/automation/ghtj/ghtj.R'
      args = ['Rscript', main_driver, location, str(epiweek)]
      subprocess.check_call(args, shell=False)
      # Need to set an absolute path
      outputdir = '/home/automation/ghtj/output'
      prefix = 'ghtpred'
      predfilename = '%s/%s-%s-%d.txt' % (outputdir, prefix, loc, epiweek)
      with open(predfilename, 'r') as f:
        mypred = float(f.read())
      print(mypred)
      return mypred

    # Making the single prediction now:
    mypred = justinfun(location, epiweek)
    return mypred

  # sensors using the loch ness fitting

  @staticmethod
  def get_gft(location, epiweek, valid):
    fetch = SignalGetter.get_gft(location, epiweek, valid)
    return SensorFitting.fit_loch_ness(location, epiweek, 'gft', 'num', fetch, valid)

  @staticmethod
  def get_ght(location, epiweek, valid):
    fetch = SignalGetter.get_ght(location, epiweek, valid)
    return SensorFitting.fit_loch_ness(location, epiweek, 'ght', 'value', fetch, valid)

  @staticmethod
  def get_twtr(location, epiweek, valid):
    fetch = SignalGetter.get_twtr(location, epiweek, valid)
    return SensorFitting.fit_loch_ness(location, epiweek, 'twtr', 'percent', fetch, valid)

  @staticmethod
  def get_wiki(location, epiweek, valid):
    fetch, fields = SignalGetter.get_wiki(location, epiweek, valid)
    return SensorFitting.fit_loch_ness(location, epiweek, 'wiki', fields, fetch, valid)

  @staticmethod
  def get_cdc(location, epiweek, valid):
    fetch, fields = SignalGetter.get_cdc(location, epiweek, valid)
    return SensorFitting.fit_loch_ness(location, epiweek, 'cdc', fields, fetch, valid)

  @staticmethod
  def get_quid(location, epiweek, valid):
    fetch, fields = SignalGetter.get_quid(location, epiweek, valid)
    return SensorFitting.fit_loch_ness(location, epiweek, 'quid', fields, fetch, valid)


class SensorUpdate:
  """
  Produces both real-time and retrospective sensor readings for ILI in the US.
  Readings (predictions of ILI made using raw inputs) are stored in the Delphi
  database and are accessible via the Epidata API.
  """

  @staticmethod
  def new_instance(valid, test_mode):
    """
    Return a new instance under the default configuration.

    If `test_mode` is True, database changes will not be committed.

    If `valid` is True, be punctilious about hiding values that were not known
    at the time (e.g. run the model with preliminary ILI only). Otherwise, be
    more lenient (e.g. fall back to final ILI when preliminary ILI isn't
    available).
    """
    database = SensorsTable(test_mode=test_mode)
    implementations = SensorGetter.get_sensor_implementations()
    return SensorUpdate(valid, database, implementations, Epidata)

  def __init__(self, valid, database, implementations, epidata):
    self.valid = valid
    self.database = database
    self.implementations = implementations
    self.epidata = epidata

  def update(self, sensors, first_week, last_week):
    """
    Compute sensor readings and store them in the database.
    """

    # most recent issue
    if last_week is None:
      last_issue = get_most_recent_issue(self.epidata)
      last_week = flu.add_epiweeks(last_issue, +1)

    # connect
    with self.database as database:

      # update each sensor
      for (name, loc) in sensors:

        # update each location
        for location in get_location_list(loc):

          # timing
          ew1 = first_week
          if ew1 is None:
            ew1 = database.get_most_recent_epiweek(name, location)
            if ew1 is None:
              # If an existing sensor reading wasn't found in the database and
              # no start week was given, just assume that readings should start
              # at 2010w40.
              ew1 = 201040
              print('%s-%s not found, starting at %d' % (name, location, ew1))

          args = (name, location, ew1, last_week)
          print('Updating %s-%s from %d to %d.' % args)
          for test_week in flu.range_epiweeks(ew1, last_week, inclusive=True):
            self.update_single(database, test_week, name, location)

  def update_single(self, database, test_week, name, location):
    train_week = flu.add_epiweeks(test_week, -1)
    impl = self.implementations[name]
    try:
      value = impl(location, train_week, self.valid)
      print(' %4s %5s %d -> %.3f' % (name, location, test_week, value))
    except Exception as ex:
      value = None
      print(' failed: %4s %5s %d' % (name, location, test_week), ex)
    if value is not None:
      database.insert(name, location, test_week, value)
    sys.stdout.flush()


def get_argument_parser():
  """Define command line arguments and usage."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'names',
      help=(
        'list of name-location pairs '
        '(location can be nat/hhs/cen/state or specific location labels)'))
  parser.add_argument(
      '--first',
      '-f',
      type=int,
      help='first epiweek override')
  parser.add_argument(
      '--last',
      '-l',
      type=int,
      help='last epiweek override')
  parser.add_argument(
      '--epiweek',
      '-w',
      type=int,
      help='epiweek override')
  parser.add_argument(
      '--test',
      '-t',
      default=False,
      action='store_true',
      help='dry run only')
  parser.add_argument(
      '--valid',
      '-v',
      default=False,
      action='store_true',
      help='do not fall back to stable wILI; require unstable wILI')
  return parser


def validate_args(args):
  """Validate and return command line arguments."""

  # check epiweek specification
  first, last, week = args.first, args.last, args.epiweek
  for ew in [first, last, week]:
    if ew is not None:
      flu.check_epiweek(ew)
  if week is not None:
    if first is not None or last is not None:
      raise ValueError('`week` overrides `first` and `last`')
    first = last = week
  if first is not None and last is not None and first > last:
    raise ValueError('`first` must not be greater than `last`')

  # validate and extract name-location pairs
  pair_regex = '[^-,]+-[^-,]+'
  names_regex = '%s(,%s)*' % (pair_regex, pair_regex)
  if not re.match(names_regex, args.names):
    raise ValueError('invalid sensor specification')

  return args.names, first, last, args.valid, args.test


def parse_sensor_location_pairs(names):
  return [pair.split('-') for pair in names.split(',')]


def main(names, first, last, valid, test):
  """Run this script from the command line."""
  sensors = parse_sensor_location_pairs(names)
  SensorUpdate.new_instance(valid, test).update(sensors, first, last)


if __name__ == '__main__':
  main(*validate_args(get_argument_parser().parse_args()))
