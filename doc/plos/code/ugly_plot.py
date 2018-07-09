# quick and dirty plots for development

import numpy as np
import matplotlib.pylab as plt


class UglyPlot:

  def __init__(self, analysis):
    self.analysis = analysis

  def _save(self, name):
    plt.savefig('data/figures/' + name + '.png', bbox_inches='tight')
    plt.close()

  def plot_sensor_heatmap(self):
    data, sensors, weeks = self.analysis.get_heatmap_data()
    fig, ax = plt.subplots()
    wk_idx = list(range(0, len(weeks), 100))
    plt.xticks(wk_idx)
    plt.yticks([0.5 + i for i in range(len(sensors))])
    ax.set_xticklabels(['%d' % weeks[i] for i in wk_idx])
    ax.set_yticklabels(sensors[::-1])
    plt.pcolor(data[::-1, :])
    self._save('sensor_heatmap_top')

    for i, sensor in enumerate(sensors):
      plt.plot(data[i, :], label=sensor)
    t = self.analysis.get_truth('nat')
    plt.plot([t[w] for w in weeks], label='Ground Truth', color='black')
    plt.ylim([0, 10])
    plt.legend()
    self._save('sensor_heatmap_bottom')
