import argparse
import csv
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

  def merge(self, a, b):
    return dict((ew, (a[ew], b[ew])) for ew in a.keys() & b.keys())

  def get_stats(self, merged):
    a = np.array([merged[ew][0] for ew in sorted(merged)])
    b = np.array([merged[ew][1] for ew in sorted(merged)])
    mae = np.mean(np.abs(a - b))

    # normalize MAE by MAE of naive nowcaster
    naive = []
    for ew1 in merged:
      ew0 = Epiweek.add_epiweeks(ew1, -52)
      if ew0 in merged:
        naive.append(np.abs(merged[ew1][0] - merged[ew0][0]))
    mae_naive = np.mean(naive)

    return {
      'n': len(merged),
      'rmse': np.sqrt(np.mean(np.square(a - b))),
      'mae': mae,
      'mae_naive': mae_naive,
      'mase': mae / mae_naive,
    }

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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'name',
      help='name of the analysis to run')
  args = parser.parse_args()
  name = args.name

  analysis = Analysis()

  if name == 'nowcast_info':
    n = analysis.get_nowcast('nat')
    print('num national nowcasts: %d' % len(n))
    print('first week: %d' % min(n))
    print('last week: %d' % max(n))
    t = 0
    for l in Locations.region_list:
      t += len(analysis.get_nowcast(l))
    print('total num nowcasts: %d' % t)
    print('num locations: %d' % len(Locations.region_list))
  elif name == 'sensor_info':
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
  elif name == 'plot':
    plotter = UglyPlot(analysis)
    plotter.plot_sensor_heatmap()
    plotter.plot_all_nowcasts()
    plotter.plot_accuracy_vs_sensors()
    plotter.plot_accuracy_vs_ablation()
    plotter.plot_accuracy_vs_abscission()
  else:
    raise Exception('unknown analysis: %s' % name)


if __name__ == '__main__':
  main()
