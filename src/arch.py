"""
===============
=== Purpose ===
===============

Predict wILI using the empirical Archetype. This uses the `Archetype` class
from the Archefilter forecasting system, and `ARCH` is a modified version of
the Archefilter forecaster with the goal of predicting only 1-week ahead,
instead of season-wide forecasting.

The basic idea is this: the future will look like the past, up to some
modifications. Here, possible modifications are shift in time, and scale in
magnitude. This system is inspired by Empirical Bayes and Pinned Spline
forecasting systems.

When producing retrospective predictions, great care is taken to only use
'valid' data: values that would have actually been available at the time.
However, unstable wILI is only available for recent years and for only some of
the regions (i.e. not in census regions). During prediction, ARCH will raise an
Exception if unstable data is unavailable.

See also:
  - archetype.py: implements the archetype idea
  - fc_archefilter.py: a forecasting system based on the archetype
  - sar3.py: another system that generates 1-week-ahead predictions


=================
=== Changelog ===
=================

2016-04-11
  * allow predictions using invalid (stable) data
  - don't produce predictions during the off-season
2016-04-07
  + initial version (borrowing heavily from fc_archefilter.py)
"""

# built-in
import argparse
# external
import numpy as np
# local
from archetype import Archetype
from delphi_epidata import Epidata
import fluv_utils as EW
from neldermead import NelderMead


class ARCH:

  def __init__(self, region):
    self.region = region
    weeks = Epidata.range(200330, 202330)
    rows = Epidata.check(Epidata.fluview(self.region, weeks))
    self.seasons = {}
    for row in rows:
      ew, wili = row['epiweek'], row['wili']
      y, w = EW.split_epiweek(ew)
      if w < 30:
        y -= 1
      i = EW.delta_epiweeks(EW.join_epiweek(y, 30), ew)
      if y not in self.seasons:
        self.seasons[y] = {}
      if 0 <= i < 52:
        self.seasons[y][i] = wili
    years = sorted(list(self.seasons.keys()))
    for year in years:
      if len(self.seasons[year]) != 52:
        del self.seasons[year]
    if 2008 in self.seasons and 2009 in self.seasons:
      for i in range(40, 52):
        self.seasons[2008][i] = self.seasons[2009][i]
      del self.seasons[2009]
    curve = lambda y: [self.seasons[y][i] for i in range(52)]
    self.years = sorted(list(self.seasons.keys()))
    self.curves = dict([(y, curve(y)) for y in self.years])

  def _get_partial_trajectory(self, epiweek, valid=True):
    y, w = EW.split_epiweek(epiweek)
    if w < 30:
      y -= 1
    ew1 = EW.join_epiweek(y, 30)
    ew2 = epiweek
    limit = EW.add_epiweeks(ew2, -5)
    weeks = Epidata.range(ew1, ew2)
    stable = Epidata.check(Epidata.fluview(self.region, weeks))
    try:
      unstable = Epidata.check(Epidata.fluview(self.region, weeks, issues=ew2))
    except:
      unstable = []
    wili = {}
    for row in stable:
      ew, value = row['epiweek'], row['wili']
      if not valid or ew < limit:
        wili[ew] = value
    for row in unstable:
      ew, value = row['epiweek'], row['wili']
      wili[ew] = value
    curve = []
    for ew in EW.range_epiweeks(ew1, ew2, inclusive=True):
      if ew not in wili:
        if valid:
          t = 'unstable'
        else:
          t = 'any'
        raise Exception('wILI (%s) not available for week %d' % (t, ew))
      curve.append(wili[ew])
    n1 = EW.delta_epiweeks(ew1, ew2) + 1
    n2 = len(curve)
    if n1 != n2:
      raise Exception('missing data (expected %d, found %d)' % (n1, n2))
    return curve

  def _fit(self, curve):
    # parameters
    min_shift, max_shift, n_shift = -10, +10, 32
    min_scale, max_scale, n_scale = 1 / 4, 4, 32
    # calculate parameter bins
    shifts = np.linspace(min_shift, max_shift, n_shift)
    scales = np.linspace(min_scale, max_scale, n_scale)
    d_shift, d_scale = shifts[1] - shifts[0], scales[1] - scales[0]
    bins = [[(t, s) for s in scales] for t in shifts]
    samples = []
    # extend partial trajectory
    i = len(curve)
    test = np.array(curve + list(self.model.mean[i:]))
    weights = 1 / np.sqrt(self.model.var)
    weights[i - 5:i] *= 2
    # objective function for best fit
    def objective(params):
      shift, scale = params
      if not (-11 <= shift <= +11) or not (1 / 5 <= scale <= 5):
        return 1e6
      arch = self.model.instance(scale, shift, True)
      score = np.mean(np.square(weights * (test - arch)))
      return score
    # get score of curve in center of each bin
    grid = np.zeros((n_shift, n_scale))
    for (t, shift) in enumerate(shifts):
      for (s, scale) in enumerate(scales):
        grid[t, s] = objective((shift, scale))
    # convert scores to PMF
    grid = np.exp(-grid)
    grid /= np.sum(grid)
    # find best bin index
    best = np.unravel_index(np.argmax(grid), grid.shape)
    # optimize parameters
    guess = bins[best[0]][best[1]]
    solver = NelderMead(objective, limit_iterations=1024, limit_time=0.5, silent=True)
    simplex = solver.get_simplex(len(guess), guess, max([d_shift, d_scale]))
    best = solver.run(simplex)._location
    # if the best fit is worse (edge case), use the original guess
    obj0, obj1 = objective(guess), objective(best)
    if np.isclose(obj0, obj1) or obj0 < obj1:
      best = guess
    # return the best-fit archetype
    shift, scale = best
    return self.model.instance(scale, shift, True)

  def train(self, epiweek):
    curves = []
    for year in self.years:
      season_end = EW.join_epiweek(year + 1, 29)
      if epiweek >= season_end:
        curves.append(self.curves[year])
    self.model = Archetype(curves)
    self.training_week = epiweek
    return curves, self.model

  def predict(self, epiweek, train=True, valid=True):
    if train:
      self.train(epiweek)
    if self.training_week > epiweek:
      raise Exception('trained on future data')
    y, w = EW.split_epiweek(epiweek)
    #if 30 <= w < 40:
    #  return float(self.model.mean[w - 30])
    #if 20 < w < 30:
    #  return float(self.model.mean[-(30 - w)])
    if 20 <= w < 39:
      raise Exception('no prediction on weeks 21--39')
    curve = self._get_partial_trajectory(epiweek, valid=valid)
    arch = self._fit(curve)
    return float(arch[len(curve)])


if __name__ == '__main__':
  # args and usage
  parser = argparse.ArgumentParser()
  parser.add_argument('epiweek', type=int, help='most recently published epiweek (best 201030+)')
  parser.add_argument('region', type=str, help='region (nat, hhs, cen)')
  args = parser.parse_args()

  # options
  ew1, reg = args.epiweek, args.region
  ew2 = EW.add_epiweeks(ew1, 1)

  # train and predict
  print('Most recent issue: %d' % ew1)
  prediction = ARCH(reg).predict(ew1, True)
  print('Predicted wILI in %s on %d: %.3f' % (reg, ew2, prediction))
  res = Epidata.fluview(reg, ew2)
  if res['result'] == 1:
    row = res['epidata'][0]
    issue = row['issue']
    wili = row['wili']
    err = prediction - wili
    print('Actual wILI as of %d: %.3f (err=%+.3f)' % (issue, wili, err))
  else:
    print('Actual wILI: unknown')
