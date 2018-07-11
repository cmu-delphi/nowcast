# quick and dirty plots for development

import numpy as np
import matplotlib.pylab as plt

from delphi.nowcast.util.flu_data_source import FluDataSource
from delphi.utils import epiweek as Epiweek
from delphi.utils.geo.locations import Locations


class UglyPlot:

  def __init__(self, analysis):
    self.analysis = analysis

  def _save(self, name):
    plt.savefig('data/figures/' + name + '.png', bbox_inches='tight')
    plt.close()

  def plot_sensor_heatmap(self):
    data, sensors, weeks = self.analysis.get_heatmap_data()

    fig, ax = plt.subplots(figsize=(6, 3))
    wk_idx = list(range(0, len(weeks), 100))
    plt.xticks(wk_idx)
    plt.yticks([0.5 + i for i in range(len(sensors))])
    ax.set_xticklabels(['%d' % weeks[i] for i in wk_idx])
    ax.set_yticklabels(sensors[::-1])
    plt.pcolor(data[::-1, :])
    self._save('sensor_heatmap_bottom')

    plt.figure(figsize=(6, 3))
    for i, sensor in enumerate(sensors):
      plt.plot(data[i, :], label=sensor)
    t = self.analysis.get_truth('nat')
    plt.plot([t[w] for w in weeks], label='Ground Truth', color='black')
    plt.ylim([0, 10])
    plt.legend()
    self._save('sensor_heatmap_top')

  def plot_all_nowcasts(self):

    def get_xy(data):
      weeks = sorted(data)
      y = [data[w] for w in weeks]
      x = [Epiweek.delta_epiweeks(201030, w) for w in weeks]
      return list(map(np.array, [x, y]))

    def shift(i, x, y):
      x = x / max(x)
      y = y / max(y)
      x = x * 0.8 + 0.1
      y = y * 0.8 + 0.1
      r = i // 5
      c = i % 5
      return x + c, y - r, r, c

    plt.figure(figsize=(6, 8))
    for i, loc in enumerate(Locations.region_list):
      x, y, r, c = shift(i, *get_xy(self.analysis.get_truth(loc)))
      s, t, r, c = shift(i, *get_xy(self.analysis.get_nowcast(loc)))
      x, y = x[-len(s):], y[-len(t):]
      plt.plot(x, y, color='#404040')
      plt.plot(s, t, color='#008060')
      plt.text(c, -r + 0.8, loc)
    self._save('all_nowcasts')

  def plot_accuracy_vs_sensors(self):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6, 8))

    loc = 'nat'
    for i, sensor in enumerate(FluDataSource.SENSORS):
      t = self.analysis.get_truth(loc)
      n = self.analysis.get_nowcast(loc)
      s = self.analysis.get_sensor(sensor, loc)
      ns = self.analysis.merge(n, s)
      t = dict((w, t[w]) for w in ns.keys())
      n = dict((w, n[w]) for w in ns.keys())
      s = dict((w, s[w]) for w in ns.keys())
      ts = self.analysis.merge(t, s)
      tn = self.analysis.merge(t, n)

      st_s = self.analysis.get_stats(ts)
      st_n = self.analysis.get_stats(tn)
      if st_s['n'] != st_n['n']:
        raise Exception()

      r = i // 2
      c = i % 2
      x = [3, 4, 0, 1]
      y = [st_s['mae'], st_n['mae'], st_s['rmse'], st_n['rmse']]
      axes[r, c].barh(x, y, tick_label=['S', 'N', 'S', 'N'])
      axes[r, c].set_xlim([0, 0.65])
      axes[r, c].text(.4, 3.5, '%s (%d)' % (sensor, st_n['n']))

    self._save('accuracy_vs_sensors')
