"""
===============
=== Purpose ===
===============

Autoregression of order 3. It predicts wILI in some region on some
epiweek using ordinary regression. There are 7 covariates (8 if you count the
intercept term):
  - 3 most recent (unstable) wILI values
  - 4 indicator (0/1) variables for holiday weeks (50 or 51 through 01)

When producing retrospective predictions, great care is taken to only use
'valid' data: values that would have actually been available at the time.
However, unstable wILI is only available for recent years and for only some of
the regions (i.e. not in census regions). During training, AR3 will fall back
to stable data if unstable data is unavailable; however, during prediction,
AR3 will raise an Exception if unstable data is unavailable.

Note that the epiweek parameter represents the most recently published issue.
The returned value is a prediction for the following week.

"""

# standard library
import argparse

# third party
import numpy as np

# first party
from delphi.epidata.client.delphi_epidata import Epidata
import delphi.operations.secrets as secrets
import delphi.utils.epiweek as EW


class AR3:

  @staticmethod
  def dot(*Ms):
    N = Ms[0]
    for M in Ms[1:]:
      N = np.dot(N, M)
    return N

  def __init__(self, region):
    self.region = region
    weeks = Epidata.range(200330, 202330)
    auth = secrets.api.fluview
    r0 = Epidata.check(Epidata.fluview(self.region, weeks, lag=0, auth=auth))
    r1 = Epidata.check(Epidata.fluview(self.region, weeks, lag=1, auth=auth))
    r2 = Epidata.check(Epidata.fluview(self.region, weeks, lag=2, auth=auth))
    rx = Epidata.check(Epidata.fluview(self.region, weeks, auth=auth))
    self.data = {}
    self.valid = {}
    self.ew2i, self.i2ew = {}, {}
    for ew in EW.range_epiweeks(weeks['from'], weeks['to'], inclusive=True):
      if 200916 <= ew <= 201015:
        continue
      i = len(self.ew2i)
      self.ew2i[ew] = i
      self.i2ew[i] = ew
    for row in r0 + r1 + r2 + rx:
      ew, wili, lag = row['epiweek'], row['wili'], row['lag']
      if ew not in self.ew2i:
        continue
      i = self.ew2i[ew]
      if i not in self.data:
        self.data[i] = {}
        self.valid[i] = {0: False, 1: False, 2: False, 'stable': False}
      if not (0 <= lag <= 2):
        lag = 'stable'
      self.data[i][lag] = wili
      self.valid[i][lag] = True
    self.weeks = sorted(list(self.data.keys()))
    for i in self.weeks:
      if 'stable' not in self.data[i]:
        continue
      for lag in range(3):
        if lag not in self.data[i]:
          self.data[i][lag] = self.data[i]['stable']

  def _get_features(self, ew, valid=True):
    X = np.zeros((1, 8))
    i = self.ew2i[ew]
    X[0, 0] = 1
    for lag in range(3):
      if valid and not self.valid[i - lag][lag]:
        w = self.i2ew[i - lag]
        raise Exception('missing unstable wILI (ew=%d|lag=%d)' % (w, lag))
      X[0, 1 + lag] = self.data[i - lag][lag]
    for holiday in range(4):
      if EW.split_epiweek(EW.add_epiweeks(ew, holiday))[1] == 1:
        X[0, 4 + holiday] = 1
    # y, w = EW.split_epiweek(ew)
    # N = EW.get_num_weeks(y)
    # offset = np.pi * 2 * w / N
    # X[0, 8] = np.sin(offset)
    # X[0, 9] = np.cos(offset)
    return X

  def train(self, epiweek):
    if epiweek not in self.ew2i:
      raise Exception('not predicting during the pandemic')
    i1 = self.weeks[2]
    i2 = self.ew2i[epiweek] - 5
    ew1, ew2 = self.i2ew[i2], self.i2ew[i2]
    num_weeks = i2 - i1 + 1
    X, Y = np.zeros((num_weeks, 8)), np.zeros((num_weeks, 1))
    r = 0
    for i in range(i1, i2 + 1):
      X[r, :] = self._get_features(self.i2ew[i], valid=False)
      Y[r, 0] = self.data[i + 1]['stable']
      r += 1
    self.model = AR3.dot(np.linalg.inv(AR3.dot(X.T, X)), X.T, Y)
    self.training_week = epiweek
    return (X, Y, self.model)

  def predict(self, epiweek, train=True, valid=True):
    if train:
      self.train(epiweek)
    if self.training_week > epiweek:
      raise Exception('trained on future data')
    X = self._get_features(epiweek, valid=valid)
    return float(AR3.dot(X, self.model)[0, 0])


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
  prediction = AR3(reg).predict(ew1, True)
  print('Predicted wILI in %s on %d: %.3f' % (reg, ew2, prediction))
  res = Epidata.fluview(reg, ew2, auth=secrets.api.fluview)
  if res['result'] == 1:
    row = res['epidata'][0]
    issue = row['issue']
    wili = row['wili']
    err = prediction - wili
    print('Actual wILI as of %d: %.3f (err=%+.3f)' % (issue, wili, err))
  else:
    print('Actual wILI: unknown')
