import argparse
import csv
import functools
import glob
import os

import numpy as np
import matplotlib.pylab as plt

from delphi.nowcast.util.flu_data_source import FluDataSource
from delphi.utils.geo.locations import Locations
from delphi.utils import epiweek as Epiweek

from ugly_plot import UglyPlot


class Analysis:

  @staticmethod
  def load_everything():
    data = {}
    for name in glob.glob('data/*.csv'):
      key = os.path.split(name)[1][:-4]
      data[key] = []
      with open(name) as f:
        reader = csv.reader(f)
        for row in reader:
          row[0] = int(row[0])
          row[-1] = float(row[-1])
          try:
            row[-2] = float(row[-2])
          except ValueError:
            pass
          data[key].append(row)
    return data

  def __init__(self):
    self.data = Analysis.load_everything()

  def get_truth(self, loc):
    result = {}
    for week, place, value in self.data['fluview']:
      if place == loc:
        result[week] = value
    return result

  def get_preliminary_truth(self, lag):
    result = {}
    for week, place, value in self.data['fluview_prelim']:
      if place == 'nat_%d' % lag:
        result[week] = value
    return result

  def get_sensor(self, name, loc):
    result = {}
    for week, sensor, place, value in self.data['sensors']:
      if sensor == name and place == loc:
        result[week] = value
    return result

  def get_experiment(self, name, loc):
    result = {}
    for week, place, value, std in self.data['nc_%s' % name]:
      if place == loc:
        if (loc == 'vi' and week < 201327) or (loc == 'pr' and week < 201453):
          # The nowcast for this location on this week was made *without* any
          # sensors for this location on this week. It's only available by
          # inference indirectly from sensors in other locations in the same
          # region. That these nowcasts were possible at all is a testament to
          # the flexibility of sensor fusion, but they are not interesting from
          # the perspective of comparing nowcast accuracy to sensor accuracy --
          # there are no sensors with which to compare. Full disclosure, these
          # nowcasts are much less accurate than nowcasts for which direct
          # sensors are available, and this is reflected by their associated
          # standard deviations on these weeks.
          continue
        result[week] = value
    return result

  def get_nowcast(self, loc):
    return self.get_experiment('vanilla', loc)

  def get_naive_nowcast(self, loc):
    # Because final wILI is not known for multiple months, it's not possible to
    # implement a *real-time* random walk naive nowcaster. There are (at least)
    # two ways to define a substitute naive nowcaster:
    #   - Naive Oracle: assume final wILI is known at runtime (it's not)
    #     and define the nowcast as final wILI on the previous week.
    #   - Seasonal Naive: define the nowcast as final wILI on the same week
    #     one year in the past.
    # Naive Oracle has the disadvantage that it's not realistic (because of
    # backfill), and therefore it is unfairly advantaged. Seasonal Naive has
    # the disadvantage that wILI 52 weeks ago is only very loosely correlated
    # with wILI at runtime, and therefore it is unfairly disadvantaged.
    # (Ideally we would define the naive nowcast as preliminary wILI on the
    # previous week, but that data isn't generally available, except for
    # certain locations and seasons.)
    # It's not immediately clear which definition of "naive" is better in this
    # situation. The variable below controls which definition is used
    # throughout this analysis; 1 corresponds to Naive Oracle, and 52
    # corresponds to Seasonal Naive.
    delta = 1

    nowcast = {}
    truth = self.get_truth(loc)
    for ew1 in truth:
      ew0 = Epiweek.add_epiweeks(ew1, -delta)
      if ew0 in truth:
        nowcast[ew1] = truth[ew0]
    return nowcast

  def get_alternative_nowcast(self, loc):
    # For each week, report the median value of all sensor readings in this
    # location, ignoring sensor readings in all other locations. This is a
    # basic "wisdom of the crowd" approach to nowcasting.
    sensors = {}
    for sensor in FluDataSource.SENSORS:
      sensors[sensor] = self.get_sensor(sensor, loc)
    nowcast = {}
    for ew in self.get_truth(loc):
      values = []
      for sensor in FluDataSource.SENSORS:
        if ew in sensors[sensor]:
          values.append(sensors[sensor][ew])
      if values:
        nowcast[ew] = np.median(values)
    return nowcast

  def get_metrics(self, truth, nowcast, naive=None):
    common_weeks = sorted(truth.keys() & nowcast.keys())
    a = np.array([truth[ew] for ew in common_weeks])
    b = np.array([nowcast[ew] for ew in common_weeks])
    result = {
      'n': len(common_weeks),
      'rmse': np.sqrt(np.mean(np.square(a - b))),
      'mae': np.mean(np.abs(a - b)),
    }
    if naive is not None:
      common_weeks = sorted(nowcast.keys() & naive.keys())
      trimmed_naive = dict([(ew, naive[ew]) for ew in common_weeks])
      result_naive = self.get_metrics(truth, trimmed_naive)
      result['mase'] = result['mae'] / result_naive['mae']
      result['rmsse'] = result['rmse'] / result_naive['rmse']
    return result

  def get_heatmap_data(self):
    w0s, w1s = [], []
    for sensor in FluDataSource.SENSORS:
      x = self.get_sensor(sensor, 'nat')
      w0s.append(min(x))
      w1s.append(max(x))

    w0, w1 = min(w0s), max(w1s)
    weeks = list(Epiweek.range_epiweeks(w0, w1, inclusive=True))
    data = np.ones((len(FluDataSource.SENSORS), len(weeks))) * -1
    for i, sensor in enumerate(FluDataSource.SENSORS):
      x = self.get_sensor(sensor, 'nat')
      for j, ew in enumerate(weeks):
        if ew in x:
          data[i, j] = x[ew]

    return data, FluDataSource.SENSORS, weeks

  @functools.lru_cache()
  def get_accuracy_table(self):
    result = {}
    for loc in Locations.region_list:
      t = self.get_truth(loc)
      n_actual = self.get_nowcast(loc)
      n_alt = self.get_alternative_nowcast(loc)
      n_naive = self.get_naive_nowcast(loc)
      st_actual = self.get_metrics(t, n_actual, naive=n_naive)
      st_alt = self.get_metrics(t, n_alt, naive=n_naive)
      result[loc] = {
        'actual_mae': st_actual['mae'],
        'actual_rmse': st_actual['rmse'],
        'actual_mase': st_actual['mase'],
        'alt_mae': st_alt['mae'],
        'alt_rmse': st_alt['rmse'],
      }
    return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'name',
      help='name of the analysis to run')
  args = parser.parse_args()
  name = args.name

  analysis = Analysis()

  if name in ('all', 'nowcast_info'):
    n = analysis.get_nowcast('nat')
    print('num national nowcasts: %d' % len(n))
    print('first week: %d' % min(n))
    print('last week: %d' % max(n))
    t = 0
    for l in Locations.region_list:
      t += len(analysis.get_nowcast(l))
    print('total num nowcasts: %d' % t)
    print('num locations: %d' % len(Locations.region_list))

  if name in ('all', 'sensor_info'):
    grand_total = 0
    for s in FluDataSource.SENSORS:
      print('%s:' % s)
      n = analysis.get_sensor(s, 'nat')
      print('  num national: %d' % len(n))
      print('  first week: %d' % min(n))
      print('  last week: %d' % max(n))
      num_loc = 0
      for l in Locations.region_list:
        n = len(analysis.get_sensor(s, l))
        if n:
          num_loc += 1
        grand_total += n
      print('  num locations: %d' % num_loc)
    print('total num readings: %d' % grand_total)

  if name in ('all', 'plot'):
    plotter = UglyPlot(analysis)
    plotter.plot_sensor_heatmap()
    plotter.plot_all_nowcasts()
    plotter.plot_accuracy_vs_sensors()
    plotter.plot_accuracy_vs_ablation()
    plotter.plot_accuracy_vs_abscission()
    plotter.plot_all_mase()

  if name in ('all', 'table'):
    filename = 'data/files/table.csv'
    print('saving %s' % filename)
    with open(filename, 'w') as f:
      writer = csv.writer(f)
      writer.writerow([
        'Location',
        'MAE Actual',
        'MAE Baseline',
        'RMSE Actual',
        'RMSE Baseline',
        'MASE Actual',
      ])
      table = analysis.get_accuracy_table()
      for loc in Locations.region_list:
        writer.writerow([
          loc,
          '%.3f' % table[loc]['actual_mae'],
          '%.3f' % table[loc]['alt_mae'],
          '%.3f' % table[loc]['actual_rmse'],
          '%.3f' % table[loc]['alt_rmse'],
          '%.3f' % table[loc]['actual_mase'],
        ])
    print('saved %s' % filename)


if __name__ == '__main__':
  main()
