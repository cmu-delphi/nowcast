"""
===============
=== Purpose ===
===============

A general-purpose implementation of sensor fusion.

See also: nowcast.py


=================
=== Changelog ===
=================

2016-04-27
  + `precision` takes optional `mean` vector
2016-04-11
  * much more efficient covariance calculation (~59s -> ~0.6s)
2016-04-09
  + first "real" version (based on various PoC's)
"""

# third party
import numpy as np


class Fusion:

  @staticmethod
  def dot(*Ms):
    """
    Convenience method to multiply several matrices.

    input:
      *Ms: a set of $n$ matrices

    output:
      dot product of all $M_0$ ... $M_n$
    """
    N = Ms[0]
    for M in Ms[1:]:
      N = np.dot(N, M)
    return N

  @staticmethod
  def mean(X):
    """
    Estimates the mean vector of partially observed data.

    input:
      X: data matrix (N x P) (N observations, P variables)

    output:
      empirical mean vector (1 x P)
    """
    n, p = X.shape
    res = np.zeros((p,))
    for i in range(p):
      Y = X[:, i]
      res[i] = np.mean(Y[np.logical_not(np.isnan(Y))])
    return res

  @staticmethod
  def cov(X, mean=None):
    """
    Estimates the covariance matrix of partially observed data.
    The resulting matrix is NOT guaranteed to be PSD or invertable.

    input:
      X: data matrix (N x P) (N observations, P variables)
      mean: mean vector

    output:
      empirical pairwise covariance matrix (P x P)
    """
    if mean is None:
      mean = Fusion.mean(X)
    Y = X - mean
    n, p = Y.shape
    res = np.zeros((p, p))
    for i in range(p):
      for j in range(i, p):
        num, total = 0, 0
        dot = Y[:, i] * Y[:, j]
        sel = np.logical_not(np.isnan(dot))
        num, total = np.sum(sel), np.sum(dot[sel])
        if num == 0:
          value = 0
        else:
          value = total / num
        res[i, j] = res[j, i] = value
    return res

  @staticmethod
  def precision(X, mean=None, b=0.1, tol=1e-2):
    """
    Estimates the precision matrix of partially observed data.
    The resulting matrix is guaranteed to be valid: PSD and invertable.

    input:
      X: data matrix (N x P) (N observations, P variables)
      b: blending weight between minimum valid alpha and one
      tol: tolerance, alpha will be valid to this precision

    output:
      estimated precision matrix (P x P)
    """
    cov0 = Fusion.cov(X, mean)
    cov1 = np.diag(np.diag(cov0))
    n = X.shape[1]
    ev, mr = np.linalg.eigvals, np.linalg.matrix_rank
    mix = lambda a: (1 - a) * cov0 + a * cov1
    check = lambda c: min(ev(c)) > 0 and mr(c) == n
    a0, a1 = 0, 1
    if check(mix(a0)):
      a1 = a0
    else:
      while a1 - a0 > tol:
        a = (a0 + a1) / 2
        if check(mix(a)):
          a1 = a
        else:
          a0 = a
    alpha = b + (1 - b) * a1
    return np.linalg.inv(mix(alpha))

  @staticmethod
  def fuse(z, Ri, H):
    """
    Implements the sensor fusion kernel (infers state from measurement):
      $$ P = (H^T R^{-1} H)^{-1} $$
      $$ x = P H^T R^{-1} z $$

    input:
      z: measurement vector (I x 1)
      Ri: measurement noise precision (I x I)
      H: map from state space to measurement space (I x S)

    output:
      x: state vector (S x 1)
      P: state covariance (S x S)
    """

    # fuse
    HtRi = Fusion.dot(H.T, Ri)
    P = np.linalg.inv(Fusion.dot(HtRi, H))
    x = Fusion.dot(P, HtRi, z)

    # finished
    return (x, P)

  @staticmethod
  def extract(x, P, W):
    """
    Computes weighted joint mean and variance (estimates output from state):
      $$ \mu = W x $$
      $$ \sigma2 = diag(W P W^T) $$

    input:
      x: state vector (S x 1)
      P: state covariance (S x S)
      W: weight matrix (O x S)

    output:
      mu: combined mean (length O)
      sigma2: combined variance (length O)
    """

    # coalesce
    mu = Fusion.dot(W, x)
    sigma2 = np.diag(Fusion.dot(W, P, W.T)).reshape(mu.shape)

    # finished
    return (mu, sigma2)
