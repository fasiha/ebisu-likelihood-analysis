"""
run as 
$ python -m nose test_ebisu3.py
or
$ python test_ebisu3.py
"""

import ebisu3 as ebisu
import unittest
from scipy.stats import gamma as gammarv, binom as binomrv  # type: ignore
import numpy as np

MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


def _gammaUpdateBinomialMonteCarlo(
    a: float,
    b: float,
    t: float,
    k: int,
    n: int,
    size=1_000_000,
) -> ebisu.GammaUpdate:
  # Scipy Gamma random variable is inverted: it needs (a, scale=1/b) for the usual (a, b) parameterization
  halflife = gammarv.rvs(a, scale=1 / b, size=size)
  # pRecall = 2**(-t/halflife)
  pRecall = np.exp(-t / halflife)
  weight = binomrv.pmf(k, n, pRecall)  # this is the likelihood of observing the data

  wsum = np.sum(weight)
  # See https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  postMean = np.sum(weight * halflife) / wsum
  # See https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Weighted_sample_variance
  postVar = np.sum(weight * (halflife - postMean)**2) / wsum

  if False:
    # This is a fancy mixed type log-moment estimator for fitting Gamma rvs from (weighted) samples.
    # It's (much) closer to the maximum likelihood fit than the method-of-moments fit we use.
    # See https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1066334959#Closed-form_estimators
    # However, in Ebisu, we just have central moments of the posterior, not samples, and there doesn't seem to be
    # an easy way to get a closed form "mixed type log-moment estimator" from moments.
    h = halflife
    w = weight
    that2 = np.sum(w * h * np.log(h)) / wsum - np.sum(w * h) / wsum * np.sum(w * np.log(h)) / wsum
    khat2 = np.sum(w * h) / wsum / that2
    fit = (khat2, 1 / that2)

  newA, newB = ebisu._meanVarToGamma(postMean, postVar)
  return ebisu.GammaUpdate(newA, newB, postMean)


def relativeError(actual: float, expected: float) -> float:
  return abs(actual - expected) / abs(expected)


class TestEbisu(unittest.TestCase):

  def test_gamma_update(self):
    """Test BASIC _gammaUpdateNoisy and _gammaUpdateBinomial

    These are the Ebisu v2-style updates, in that there's no boost, just a prior
    on halflife and either quiz type. These have to be correct for the boost
    mechanism to work.
    """
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    a, b = initHlPrior

    noisyTToUpdate: dict[float, list[ebisu.GammaUpdate]] = dict()
    binomResultToUpdate: dict[int, list[ebisu.GammaUpdate]] = dict()
    for fraction in [0.1, 0.5, 1., 2., 10.]:
      t = initHlMean * fraction

      for noisy in [0.1, 0.3, 0.7, 0.9]:
        z = noisy >= 0.5
        q1 = noisy if z else 1 - noisy
        q0 = 1 - q1
        updated = ebisu._gammaUpdateNoisy(a, b, t, q1, q0, z)

        msg = f'q1={q1}, q0={q0}, z={z}, noisy={noisy}, u={updated}'
        if z:
          self.assertTrue(updated.mean >= initHlMean, msg)
        else:
          self.assertTrue(updated.mean <= initHlMean, msg)
        sublist = noisyTToUpdate.setdefault(t, [])
        sublist.append(updated)

      for result in [0, 1]:
        updated = ebisu._gammaUpdateBinomial(a, b, t, result, 1)
        if result:
          self.assertTrue(updated.mean >= initHlMean)
        else:
          self.assertTrue(updated.mean <= initHlMean)
        sublist = binomResultToUpdate.setdefault(result, [])
        sublist.append(updated)

    # Binomial updates should be monotonic in `t`
    for quizResult, updatedList in binomResultToUpdate.items():
      for (l, r) in zip(updatedList, updatedList[1:]):
        # `l` will be from a shorter quiz delay than `r`
        self.assertTrue(l.mean < r.mean)

    # the above test doesn't make sense for `noisyUpdates`: for non-zero q0,
    # we get non-asymptotic results for high `t`: see
    # https://github.com/fasiha/ebisu/issues/52

    # Noisy updates should be monotonic in `z` (the noisy result)
    for quizTime, updatedList in noisyTToUpdate.items():
      for (l, r) in zip(updatedList, updatedList[1:]):
        # `l` will have lower noisy quiz result than `r`
        self.assertTrue(l.mean < r.mean)

  def test_gamma_update_vs_montecarlo(self):
    "Test Gamma-only updates via Monte Carlo"
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)
    a, b = initHlPrior

    for fraction in [0.1, 0.5, 1., 2., 10.]:
      t = initHlMean * fraction
      for result in [0, 1]:
        updated = ebisu._gammaUpdateBinomial(a, b, t, result, 1)
        u2 = _gammaUpdateBinomialMonteCarlo(a, b, t, result, 1, size=1_000_000)
        self.assertLess(relativeError(updated.a, u2.a), .01)
        self.assertLess(relativeError(updated.b, u2.b), .01)
        self.assertLess(relativeError(updated.mean, u2.mean), .01)

  def test_simple(self):
    """Test simple update: boosted"""
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    boostMean = 1.5
    boostBeta = 3.0
    boostPrior = (boostBeta * boostMean, boostBeta)

    nowMs = ebisu._timeMs()

    currentHalflife = initHlMean

    init = ebisu.Model(
        elapseds=[],
        results=[],
        startStrengths=[],
        initHalflifePrior=initHlPrior,
        boostPrior=boostPrior,
        startTime=nowMs,
        currentHalflife=currentHalflife,
        logStrength=0.0,
    )

    updateds: list[ebisu.Model] = []
    for fraction in [0.1, 0.5, 1.0, 2.0, 10.0]:
      for result in [0, 1]:
        elapsedHours = fraction * initHlMean
        updated = ebisu.simpleUpdateRecall(
            init,
            elapsedHours,
            result,
            total=1,
            now=nowMs + elapsedHours * MILLISECONDS_PER_HOUR,
            reinforcement=1.0)

        msg = f'result={result}, fraction={fraction} => currHl={updated.currentHalflife}'
        if result:
          self.assertTrue(updated.currentHalflife >= initHlMean, msg)
        else:
          self.assertTrue(updated.currentHalflife <= initHlMean, msg)

        updateds.append(updated)
    # print(updateds)


if __name__ == '__main__':
  import os
  # get just this file's module name: no `.py` and no path
  name = os.path.basename(__file__).replace(".py", "")
  unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromName(name))