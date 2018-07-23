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
    print('>', name)

  def plot_sensor_heatmap(self):
    data, sensors, weeks = self.analysis.get_heatmap_data()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

    ax = axes[1]
    wk_idx = list(range(0, len(weeks), 100))
    ax.set_xticks(wk_idx)
    ax.set_yticks([0.5 + i for i in range(len(sensors))])
    ax.set_xticklabels(['%d' % weeks[i] for i in wk_idx])
    ax.set_yticklabels(sensors[::-1])
    ax.pcolor(data[::-1, :])

    ax = axes[0]
    for i, sensor in enumerate(sensors):
      ax.plot(data[i, :], label=sensor)
    t = self.analysis.get_truth('nat')
    ax.plot([t[w] for w in weeks], label='Ground Truth', color='black')
    ax.set_ylim([0, 10])
    ax.legend()

    self._save('sensor_heatmap')

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

  def plot_accuracy_vs_ablation(self):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6, 8))

    loc = 'nat'
    for i, sensor in enumerate(FluDataSource.SENSORS):
      t = self.analysis.get_truth(loc)
      n = self.analysis.get_nowcast(loc)
      s = self.analysis.get_experiment('ablate_%s' % sensor, loc)
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
      axes[r, c].barh(x, y, tick_label=['-', '+', '-', '+'])
      axes[r, c].set_xlim([0, 0.3])
      axes[r, c].text(.2, 3.5, '%s (%d)' % (sensor, st_n['n']))

    self._save('accuracy_vs_ablation')

  def plot_accuracy_vs_abscission(self):
    fig, ax = plt.subplots(figsize=(3, 3))

    loc = 'nat'
    tru = self.analysis.get_truth(loc)
    a_n = self.analysis.get_experiment('abscise2_national', loc)
    a_r = self.analysis.get_experiment('abscise2_regional', loc)
    a_s = self.analysis.get_experiment('abscise2_state', loc)

    tvn = self.analysis.merge(tru, a_n)
    tvr = self.analysis.merge(tru, a_r)
    tvs = self.analysis.merge(tru, a_s)

    st_n = self.analysis.get_stats(tvn)
    st_r = self.analysis.get_stats(tvr)
    st_s = self.analysis.get_stats(tvs)
    if st_n['n'] != st_r['n'] or st_n['n'] != st_s['n']:
      raise Exception()

    x = [6, 5, 4, 2, 1, 0]
    y = [
      st_n['mae'],
      st_r['mae'],
      st_s['mae'],
      st_n['rmse'],
      st_r['rmse'],
      st_s['rmse'],
    ]
    ax.barh(x, y, tick_label=['Nm', 'Rm', 'Sm', 'Nr', 'Rr', 'Sr'])
    ax.set_xlim([0, 0.3])
    ax.text(.25, 5, '(%d)' % (st_n['n']))

    self._save('accuracy_vs_abscission')

  def plot_all_mase(self):
    fig, ax = plt.subplots(figsize=(6, 3))
    nums = []
    colors = []
    for i, loc in enumerate(Locations.region_list):
      idx = i % 2
      idx += 2 if loc in Locations.hhs_list else 0
      idx += 4 if loc in Locations.cen_list else 0
      idx += 6 if loc in Locations.atom_list else 0
      colors.append([
        '#404040',
        '#808080',
        '#804040',
        '#e08080',
        '#408040',
        '#80e080',
        '#404080',
        '#8080e0'][idx])
      ft = self.analysis.get_truth(loc)
      nc = self.analysis.get_nowcast(loc)
      merged = self.analysis.merge(ft, nc)
      st = self.analysis.get_stats(merged)
      nums.append(st['mase'])
    idx = list(range(len(nums)))
    plt.bar(idx, nums, color=colors)
    ax.tick_params(axis='x', labelsize=5)
    plt.xticks(idx, Locations.region_list, rotation='vertical')
    self._save('all_mase')
