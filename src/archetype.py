"""
===============
=== Purpose ===
===============

An empirical flu model.


=================
=== Changelog ===
=================

2015-11-09
  * Fixed [add|remove]_holiday_week
2015-10-30
  + Add/remove holiday effect for individual weeks
  * Fixed curve rotation when using numpy floats
  - Unused plotting code
2015-10-28
  + Also store mean/variance of unaligned curves
  * Fixed negative curve rotation
2015-10-26
  + First version
"""

from math import floor, ceil
import numpy as np
import numpy.linalg as linalg
import scipy.stats as stats
from neldermead import NelderMead
#from trendfilter_r import trendfilter


class Archetype:

  def __init__(self, curves, week0=30, bandwidth=2, window=17, baseline=None):
    if window % 2 != 1:
      raise Exception('window length must be odd')
    self.curves = [np.array(c) for c in curves]
    self.week0 = week0
    self.bandwidth = bandwidth
    norm = stats.norm(window // 2, bandwidth)
    prob = lambda i: norm.cdf(i + 0.5) - norm.cdf(i - 0.5)
    self.kernel = np.array([prob(i) for i in range(window)])
    if not np.isclose(self.kernel[0], self.kernel[-1]):
      raise Exception('window must be symmetric')
    self.w2i = dict([(w, (52 + w - self.week0) % 52) for w in range(1, 53)])
    self.i2w = dict([(v, k) for (k, v) in self.w2i.items()])
    self.holiday = self.build_holiday_model()
    self.curves_nh = [c * self.holiday for c in self.curves]
    self.curves_nh_sm = [self.smooth(c) for c in self.curves_nh]
    alignment = [25 - self.w2i[self.peakweek(c)] for c in self.curves_nh_sm]
    self.curves_nh_al = [self.rotate(c, a) for (c, a) in zip(self.curves_nh, alignment)]
    self.curves_nh_sm_al = [self.rotate(c, a) for (c, a) in zip(self.curves_nh_sm, alignment)]
    self.smoothed_mean = np.mean(self.curves_nh_sm_al, axis=0)
    self.unsmoothed_mean = np.mean(self.curves_nh_al, axis=0)
    self.unaligned_unsmoothed_mean = np.mean(self.curves_nh, axis=0)
    self.unaligned_unsmoothed_var = np.var(self.curves_nh, axis=0, ddof=1)
    weights = np.concatenate((np.hanning(51), np.zeros(1)))
    self.mean = weights * self.unsmoothed_mean + (1 - weights) * self.smoothed_mean
    #self.mean = self.smoothed_mean
    self.var = np.var(self.curves_nh_sm_al, axis=0, ddof=1)
    if baseline is None:
      self.baseline = min(self.mean)
    else:
      self.baseline = baseline

  def peakweek(self, curve):
    return self.i2w[np.argmax(curve)]

  def peakheight(self, curve):
    return max(curve)

  def rotate(self, curve, n):
    if np.isclose(n, round(n)):
      return np.roll(curve, int(round(n)))
    else:
      n1, n2 = floor(n), ceil(n)
      w1, w2 = n2 - n, n - n1
      return w1 * np.roll(curve, n1) + w2 * np.roll(curve, n2)

  def build_holiday_model(self):
    week0, week1 = 49, 2
    idx0, idx1 = self.w2i[week0], self.w2i[week1]
    if idx0 >= idx1:
      raise Exception('holiday period must be contiguous')
    norm = lambda x: linalg.norm(x[idx0:idx1], 2)
    peaks0, derivatives0, scores0 = [], [], []
    for c0 in self.curves:
      peaks0.append(self.peakheight(c0))
      derivatives0.append(np.diff(c0, 2))
      scores0.append(norm(derivatives0[-1]))
    def objective(params):
      holiday = np.array(params)
      if max(holiday) > 1:
        return 1e9
      score = 0
      for (c0, p0, d0, s0) in zip(self.curves, peaks0, derivatives0, scores0):
        c1 = np.copy(c0)
        for i in range(4):
          c1[idx0 + 1 + i] *= holiday[i]
        p1 = self.peakheight(c1)
        d1 = np.diff(c1, 2)
        s1 = norm(d1)
        if not np.isclose(p0, p1):
          s1 = s0
        score += s1
      return score
    # optimize parameters
    guess = (1, 1, 1, 1)
    solver = NelderMead(objective, limit_iterations=100, limit_value=1e-3, limit_time=0.5, silent=True)
    simplex = solver.get_simplex(len(guess), guess, 0.1)
    best = solver.run(simplex)._location
    obj0, obj1 = objective(guess), objective(best)
    if np.isclose(obj0, obj1) or obj0 < obj1:
      best = guess
    best = self.rotate(list(best) + [1] * (48), idx0 + 1)
    return best

  def smooth(self, curve):
    extend = len(self.kernel) // 2
    temp = list(curve)
    temp = temp[-extend:] + temp + temp[:+extend]
    return np.convolve(temp, self.kernel, 'valid')
    #return trendfilter(curve)

  def scale(self, curve, s):
    return (curve - self.baseline) * s + self.baseline

  def instance(self, s, t, add_holiday):
    curve = self.scale(self.rotate(self.mean, t), s)
    if add_holiday:
      curve /= self.holiday
    return curve

  def add_holiday_week(self, ili, week):
    return ili / self.holiday[week]

  def remove_holiday_week(self, ili, week):
    return ili * self.holiday[week]

  def get_best_fit(self, curve):
    curve = np.array(curve)
    curve_nh = curve * self.holiday
    curve_nh_sm = self.smooth(curve_nh)
    def objective(params):
      t, s = params
      return linalg.norm(self.instance(s, t, False) - curve_nh, 2)
    # optimize parameters
    shift0 = self.peakweek(curve_nh_sm) - self.peakweek(self.mean)
    scale0 = (self.peakheight(curve_nh) - self.baseline) / (self.peakheight(self.mean) - self.baseline)
    #scale0 = self.peakheight(curve_nh) / self.peakheight(self.mean)
    guess = (shift0, scale0)
    solver = NelderMead(objective, limit_iterations=100, limit_value=1e-3, limit_time=0.5, silent=True)
    simplex = solver.get_simplex(len(guess), guess, 0.1)
    best = solver.run(simplex)._location
    obj0, obj1 = objective(guess), objective(best)
    if np.isclose(obj0, obj1) or obj0 < obj1:
      best = guess
    shift1, scale1 = best
    return self.instance(scale1, shift1, True)

  @staticmethod
  def RMS(a, b):
    return np.sqrt(np.mean(np.square(a - b)))

  @staticmethod
  def evaluate_model(target, fitted, variance=1, alpha=0.05, silent=False):
    residuals = (np.array(fitted) - np.array(target)) / np.sqrt(np.array(variance))
    # condition 1: residuals must be normally distributed
    k2, p1 = stats.normaltest(residuals)
    # condition 2: residuals must be idd (not autocorrelated)
    cc, p2 = stats.pearsonr(residuals[:-1], residuals[1:])
    # check both conditions
    is_normal = p1 > alpha
    is_correlated = p2 <= alpha
    good_enough = is_normal and not is_correlated
    # measure error
    rms = Archetype.RMS(fitted, target)
    # show results
    if not silent:
      if good_enough:
        print('RMS=%.3f | Normal and IID! (p_1=%.3f, p_2=%.3f)' % (rms, p1, p2))
      elif is_normal:
        print('RMS=%.3f | Normal but not IID. (p_1=%.3f, p_2=%.6f)' % (rms, p1, p2))
      else:
        print('RMS=%.3f | Neither normal nor IID. (p_1=%.6f, p_2=%.6f)' % (rms, p1, p2))
    return rms, good_enough, p1, p2
