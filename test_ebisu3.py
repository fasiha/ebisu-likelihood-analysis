"""
run as 
$ python -m nose test_ebisu3.py
or
$ python test_ebisu3.py
"""

from functools import cache
import ebisu3 as ebisu
import unittest
from scipy.stats import gamma as gammarv, binom as binomrv, bernoulli  # type: ignore
from scipy.special import logsumexp, loggamma  # type: ignore
from scipy.optimize import shgo, minimize  # type: ignore
import numpy as np
from typing import Optional, Union, Callable
import math
from dataclasses import dataclass, replace
from utils import sequentialImportanceResample
import time

MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
weightedGammaEstimate = ebisu._weightedGammaEstimate


def _gammaToVar(alpha: float, beta: float) -> float:
  return alpha / beta**2


def fullBinomialMonteCarlo(
    hlPrior: tuple[float, float],
    bPrior: tuple[float, float],
    ts: list[float],
    ks: list[int],
    ns: list[int],
    left=0.3,
    size=1_000_000,
):
  hl0s = gammarv.rvs(hlPrior[0], scale=1 / hlPrior[1], size=size)
  boosts = gammarv.rvs(bPrior[0], scale=1 / bPrior[1], size=size)

  logweights = np.zeros(size)

  hls = hl0s.copy()
  for t, k, n in zip(ts, ks, ns):
    logps = -t / hls * np.log(2)
    if k == n:  # success
      logweights += logps
    else:
      logweights += np.log(-np.expm1(logps))
    # This is the likelihood of observing the data, and is more accurate than
    # `binomrv.logpmf(k, n, pRecall)` since `pRecall` is already in log domain

    # Apply boost for successful quizzes
    if ebisu.success(ebisu.BinomialResult(k, n)):  # reuse same rule as ebisu
      hls *= clampLerp(left * hls, hls, np.minimum(boosts, 1.0), boosts, t)

  kishEffectiveSampleSize = np.exp(2 * logsumexp(logweights) - logsumexp(2 * logweights)) / size
  w = np.exp(logweights)
  estb = weightedGammaEstimate(boosts, w)
  esthl0 = weightedGammaEstimate(hl0s, w)
  vars = []
  if True:
    vars = [np.std(w * v) for v in [boosts, hl0s]]
  return dict(
      kishEffectiveSampleSize=kishEffectiveSampleSize,
      posteriorBoost=estb,
      posteriorInitHl=esthl0,
      statsBoost=weightedMeanVarLogw(logweights, boosts),
      statsInitHl=weightedMeanVarLogw(logweights, hl0s),
      # modeHl0=modeHl0,
      # corr=np.corrcoef(np.vstack([hl0s, hls, boosts])),
      # posteriorCurrHl=posteriorCurrHl,
      # estb=estb,
      # esthl0=esthl0,
      # esthl=esthl,
      vars=vars,
  )


def clampLerp(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, x: float):
  # Asssuming x1 <= x <= x2, map x from [x0, x1] to [0, 1]
  mu: Union[float, np.ndarray] = (x - x1) / (x2 - x1)  # will be >=0 and <=1
  ret = np.empty_like(y2)
  idx = x < x1
  ret[idx] = y1[idx]
  idx = x > x2
  ret[idx] = y2[idx]
  idx = np.logical_and(x1 <= x, x <= x2)
  ret[idx] = (y1 * (1 - mu) + y2 * mu)[idx]
  return ret


def weightedMeanVarLogw(logw: np.ndarray, x: np.ndarray) -> tuple[float, float]:
  # [weightedMean] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  # [weightedVar] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Weighted_sample_variance
  logsumexpw = logsumexp(logw)
  mean = np.exp(logsumexp(logw, b=x) - logsumexpw)
  var = np.exp(logsumexp(logw, b=(x - mean)**2) - logsumexpw)
  return (mean, var)


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
  # pRecall = 2**(-t/halflife) FIXME
  pRecall = np.exp(-t / halflife)

  logweight = binomrv.logpmf(k, n, pRecall)  # this is the likelihood of observing the data
  weight = np.exp(logweight)
  # use logpmf because macOS Scipy `pmf` overflows for pRecall around 2.16337e-319?

  wsum = math.fsum(weight)
  # See https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  postMean = math.fsum(weight * halflife) / wsum
  # See https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Weighted_sample_variance
  postVar = math.fsum(weight * (halflife - postMean)**2) / wsum

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


def _gammaUpdateNoisyMonteCarlo(
    a: float,
    b: float,
    t: float,
    q1: float,
    q0: float,
    z: bool,
    size=1_000_000,
) -> ebisu.GammaUpdate:
  halflife = gammarv.rvs(a, scale=1 / b, size=size)
  # pRecall = 2**(-t/halflife) FIXME
  pRecall = np.exp(-t / halflife)

  # this weight is `P(z | pRecall)` and derived and checked via Stan in
  # https://github.com/fasiha/ebisu/issues/52
  # Notably, this expression is NOT used by ebisu, so it's a great independent check
  weight = bernoulli.pmf(z, q1) * pRecall + bernoulli.pmf(z, q0) * (1 - pRecall)

  wsum = math.fsum(weight)
  # for references to formulas, see `_gammaUpdateBinomialMonteCarlo`
  postMean = math.fsum(weight * halflife) / wsum
  postVar = math.fsum(weight * (halflife - postMean)**2) / wsum

  newA, newB = ebisu._meanVarToGamma(postMean, postVar)
  return ebisu.GammaUpdate(newA, newB, postMean)


def relativeError(actual: float, expected: float) -> float:
  return np.abs(actual - expected) / np.abs(expected)


class TestEbisu(unittest.TestCase):

  def test_gamma_update_noisy(self):
    """Test _gammaUpdateNoisy for various q0 and against Monte Carlo

    These are the Ebisu v2-style updates, in that there's no boost, just a prior
    on halflife and either quiz type. These have to be correct for the boost
    mechanism to work.
    """
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    a, b = initHlPrior

    MAX_RELERR_AB = .02
    MAX_RELERR_MEAN = .01
    np.random.seed(seed=233423 + 1)  # for sanity when testing with Monte Carlo
    for fraction in [0.1, 0.5, 1., 2., 10.]:
      t = initHlMean * fraction
      for q0 in [.15, 0, None]:
        prev: Optional[ebisu.GammaUpdate] = None
        for noisy in [0.1, 0.3, 0.7, 0.9]:
          z = noisy >= 0.5
          q1 = noisy if z else 1 - noisy
          q0 = 1 - q1 if q0 is None else q0
          updated = ebisu._gammaUpdateNoisy(a, b, t, q1, q0, z)

          for size in [100_000, 500_000, 1_000_000]:
            u2 = _gammaUpdateNoisyMonteCarlo(a, b, t, q1, q0, z, size=size)
            if (relativeError(updated.a, u2.a) < MAX_RELERR_AB and
                relativeError(updated.b, u2.b) < MAX_RELERR_AB and
                relativeError(updated.mean, u2.mean) < MAX_RELERR_MEAN):
              # found a size that should match the actual tests below
              break

          self.assertLess(relativeError(updated.a, u2.a), MAX_RELERR_AB)
          self.assertLess(relativeError(updated.b, u2.b), MAX_RELERR_AB)
          self.assertLess(relativeError(updated.mean, u2.mean), MAX_RELERR_MEAN)

          msg = f'q0={q0}, z={z}, noisy={noisy}'
          if z:
            self.assertGreaterEqual(updated.mean, initHlMean, msg)
          else:
            self.assertLessEqual(updated.mean, initHlMean, msg)

          if prev:
            # Noisy updates should be monotonic in `z` (the noisy result)
            lt = prev.mean <= updated.mean
            approx = relativeError(prev.mean, updated.mean) < (np.spacing(updated.mean) * 1e3)
            self.assertTrue(
                lt or approx,
                f'{msg}, prev.mean={prev.mean}, updated.mean={updated.mean}, lt={lt}, approx={approx}'
            )
          # Means WILL NOT be monotonic in `t`: for `q0 > 0`,
          # means rise with `t`, then peak, then drop: see
          # https://github.com/fasiha/ebisu/issues/52

          prev = updated

  def test_gamma_update_binom(self):
    """Test BASIC _gammaUpdateBinomial"""
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    a, b = initHlPrior

    maxN = 4
    ts = [fraction * initHlMean for fraction in [0.1, 0.5, 1., 2., 10.]]
    us: dict[tuple[int, int, int], ebisu.GammaUpdate] = dict()
    for tidx, t in enumerate(ts):
      for n in range(1, maxN + 1):
        for result in range(n + 1):
          updated = ebisu._gammaUpdateBinomial(a, b, t, result, n)
          self.assertTrue(np.all(np.isfinite([updated.a, updated.b, updated.mean])))
          if result == n:
            self.assertGreaterEqual(updated.mean, initHlMean, (t, result, n))
          elif result == 0:
            self.assertLessEqual(updated.mean, initHlMean, (t, result, n))

        us[(tidx, result, n)] = updated

    for tidx, k, n in us:
      curr = us[(tidx, k, n)]

      # Binomial updated means should be monotonic in `t`
      prev = us.get((tidx - 1, k, n))
      if prev:
        self.assertTrue(prev.mean < curr.mean)

      # Means should be monotonic in `k`/`result` for fixed `n`
      prev = us.get((tidx, k - 1, n))
      if prev:
        self.assertTrue(prev.mean < curr.mean)

      # And should be monotonic in `n` for fixed `k`/`result`
      prev = us.get((tidx, k, n - 1))
      if prev:
        self.assertTrue(prev.mean < curr.mean)

  def test_gamma_update_vs_montecarlo(self):
    "Test Gamma-only updates via Monte Carlo"
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)
    a, b = initHlPrior

    # These thresholds on relative error between the analytical and Monte Carlo updates
    # should be enough for several trials of this unit test (see `trial` below). Nonetheless
    # I set the seed to avoid test surprises.
    MAX_RELERR_AB = .05
    MAX_RELERR_MEAN = .01
    np.random.seed(seed=233423 + 1)
    for trial in range(1):
      for fraction in [0.1, 1., 10.]:
        t = initHlMean * fraction
        for n in [1, 2, 3, 4]:  # total number of binomial attempts
          for result in range(n + 1):  # number of binomial successes
            updated = ebisu._gammaUpdateBinomial(a, b, t, result, n)
            self.assertTrue(
                np.all(np.isfinite([updated.a, updated.b, updated.mean])), f'k={result}, n={n}')

            # in order to avoid egregiously long tests, scale up the number of Monte Carlo samples
            # to meet the thresholds above.
            for size in [100_000, 500_000, 2_000_000, 5_000_000]:
              u2 = _gammaUpdateBinomialMonteCarlo(a, b, t, result, n, size=size)

              if (relativeError(updated.a, u2.a) < MAX_RELERR_AB and
                  relativeError(updated.b, u2.b) < MAX_RELERR_AB and
                  relativeError(updated.mean, u2.mean) < MAX_RELERR_MEAN):
                # found a size that should match the actual tests below
                break

            msg = f'{(trial, t, result, n, size)}'
            self.assertLess(relativeError(updated.a, u2.a), MAX_RELERR_AB, msg)
            self.assertLess(relativeError(updated.b, u2.b), MAX_RELERR_AB, msg)
            self.assertLess(relativeError(updated.mean, u2.mean), MAX_RELERR_MEAN, msg)

  def test_simple(self):
    """Test simple binomial update: boosted"""
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

    left = 0.3

    for fraction in [0.1, 0.5, 1.0, 2.0, 10.0]:
      for result in [0, 1]:
        elapsedHours = fraction * initHlMean
        updated = ebisu.simpleUpdateRecall(
            init,
            elapsedHours,
            result,
            total=1,
            now=nowMs + elapsedHours * MILLISECONDS_PER_HOUR,
            reinforcement=1.0,
            left=left,
        )

        msg = f'result={result}, fraction={fraction} => currHl={updated.currentHalflife}'
        if result:
          self.assertTrue(updated.currentHalflife >= initHlMean, msg)
        else:
          self.assertTrue(updated.currentHalflife <= initHlMean, msg)

        # this is the unboosted posterior update
        u2 = ebisu._gammaUpdateBinomial(initHlPrior[0], initHlPrior[1], elapsedHours, result, 1)

        # this uses the two-point formula: y=(y2-y1)/(x2-x1)*(x-x1) + y1, where
        # y represents the boost fraction and x represents the time elapsed as
        # a fraction of the initial halflife
        boostFraction = (boostMean - 1) / (1 - left) * (fraction - left) + 1

        # clamp 1 <= boost <= boostMean, and only boost successes
        boost = min(boostMean, max(1, boostFraction)) if result else 1
        self.assertAlmostEqual(updated.currentHalflife, boost * u2.mean)

        for nextResult in [1, 0]:
          for i in range(3):
            nextElapsed, boost = updated.currentHalflife, boostMean
            nextUpdate = ebisu.simpleUpdateRecall(
                updated,
                nextElapsed,
                nextResult,
                now=nowMs + (elapsedHours + (i + 1) * nextElapsed) * MILLISECONDS_PER_HOUR,
                left=left,
            )

            initMean = lambda model: ebisu._gammaToMean(*model.initHalflifePrior)

            # confirm the initial halflife estimate rose/dropped
            if nextResult:
              self.assertGreater(initMean(nextUpdate), 1.05 * initMean(updated))
            else:
              self.assertLess(initMean(nextUpdate), 1.05 * initMean(updated))

            # this checks the scaling applied to take the new Gamma to the initial Gamma in simpleUpdateRecall
            self.assertGreater(nextUpdate.currentHalflife, 1.1 * initMean(nextUpdate))

            # meanwhile this checks the scaling to convert the initial halflife Gamma and the current halflife mean
            currHlPrior, _ = ebisu._currentHalflifePrior(updated)
            self.assertAlmostEqual(updated.currentHalflife,
                                   gammarv.mean(currHlPrior[0], scale=1 / currHlPrior[1]))

            if nextResult:
              # this is an almost tautological test but just as a sanity check, confirm that boosts are being applied?
              next2 = ebisu._gammaUpdateBinomial(currHlPrior[0], currHlPrior[1], nextElapsed,
                                                 nextResult, 1)
              self.assertAlmostEqual(nextUpdate.currentHalflife, next2.mean * boost)
              # don't test this for failures: no boost is applied then

            updated = nextUpdate

            # don't bother to check alpha/beta: a test in Python will just be tautological
            # (we'll repeat the same thing in the test as in the code). That has to happen
            # via Stan?

  def test_full(self, verbose=False):
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

    import mpmath as mp  # type:ignore

    left = 0.3
    for fraction in [0.1, 0.5, 1.5, 9.5]:
      for result in [1, 0]:
        upd = replace(init)
        elapsedHours = fraction * initHlMean
        ebisu._appendQuiz(upd, elapsedHours, ebisu.BinomialResult(result, 1), 1.0)

        for nextResult, nextElapsed in zip([1, 1, 1],
                                           [elapsedHours * 3, elapsedHours * 5, elapsedHours * 7]):
          ebisu._appendQuiz(upd, nextElapsed, ebisu.BinomialResult(nextResult, 1), 1.0)

        tic = time.perf_counter()
        full, fullDebug = ebisu.fullUpdateRecall(
            upd, left=left, size=10_000 if fraction < 9 else 20_000, debug=True)
        toc = time.perf_counter()
        if verbose:
          print(f"fullUpdateRecall: {toc - tic:0.4f} seconds, {fullDebug}")

        @cache
        def posterior2d(b, h):
          return mp.exp(ebisu._posterior(float(b), float(h), upd, 0.3, 1.0))

        method = 'gauss-legendre'
        maxdegree = 4 if fraction < 9 else 5
        tic = time.perf_counter()
        f0 = lambda b, h: posterior2d(b, h)
        den = mp.quad(f0, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)
        fb = lambda b, h: b * posterior2d(b, h)
        numb = mp.quad(fb, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)
        fh = lambda b, h: h * posterior2d(b, h)
        numh = mp.quad(fh, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)

        # second non-central moment
        fh = lambda b, h: h**2 * posterior2d(b, h)
        numh2 = mp.quad(fh, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)
        fb = lambda b, h: b**2 * posterior2d(b, h)
        numb2 = mp.quad(fb, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)
        toc = time.perf_counter()
        if verbose:
          print(f"Numerical integration: {toc - tic:0.4f} seconds")

        boostMeanInt, hl0MeanInt = numb / den, numh / den
        boostVarInt, hl0VarInt = numb2 / den - boostMeanInt**2, numh2 / den - hl0MeanInt**2

        size = 300_000 if fraction < 9 else 1_500_000
        tic = time.perf_counter()
        mc = fullBinomialMonteCarlo(
            init.initHalflifePrior,
            init.boostPrior, [t for t in upd.elapseds[-1]], [r.successes for r in upd.results[-1]],
            [1 for t in upd.elapseds[-1]],
            size=size)
        toc = time.perf_counter()
        if verbose:
          print(
              f"Monte Carlo: {toc - tic:0.4f} seconds, kish={mc['kishEffectiveSampleSize']}, vars={mc['vars']}"
          )

        if verbose:
          print(f'an={full.initHalflifePrior}; mc={mc["posteriorInitHl"]}')
          print(
              f'mean: an={ebisu._gammaToMean(*full.initHalflifePrior)}; mc={ebisu._gammaToMean(*mc["posteriorInitHl"])}; rawMc={mc["statsInitHl"][0]}; int={hl0MeanInt}'
          )
          print(
              f'VAR: an={_gammaToVar(*full.initHalflifePrior)}; mc={_gammaToVar(*mc["posteriorInitHl"])}; rawMc={mc["statsInitHl"][1]}; int={hl0VarInt}'
          )
        if verbose:
          print(f'an={full.boostPrior}; mc={mc["posteriorBoost"]}')
          print(
              f'mean: an={ebisu._gammaToMean(*full.boostPrior)}; mc={ebisu._gammaToMean(*mc["posteriorBoost"])}; rawMc={mc["statsBoost"][0]}; int={boostMeanInt}'
          )
          print(
              f'VAR: an={_gammaToVar(*full.boostPrior)}; mc={_gammaToVar(*mc["posteriorBoost"])}; rawMc={mc["statsBoost"][1]}; int={boostVarInt}'
          )

        self.assertLess(
            np.max(
                relativeError(
                    np.array([full.boostPrior, full.initHalflifePrior]),
                    np.array([mc["posteriorBoost"], mc["posteriorInitHl"]]))),
            .15,
            f'analytical ~ mc, {fraction=}, {result=}',
        )
        self.assertLess(
            relativeError(ebisu._gammaToMean(*full.initHalflifePrior), hl0MeanInt),
            0.05,
            f'analytical ~ numerical integration mean hl0, {fraction=}, {result=}',
        )
        self.assertLess(
            relativeError(ebisu._gammaToMean(*mc["posteriorInitHl"]), hl0MeanInt),
            0.04,
            f'monte carlo ~ numerical integration mean hl0, {fraction=}, {result=}',
        )

        self.assertLess(
            relativeError(ebisu._gammaToMean(*full.boostPrior), boostMeanInt),
            0.05,
            f'analytical ~ numerical integration mean boost, {fraction=}, {result=}',
        )
        self.assertLess(
            relativeError(ebisu._gammaToMean(*mc["posteriorBoost"]), boostMeanInt),
            0.02,
            f'monte carlo ~ numerical integration mean boost, {fraction=}, {result=}',
        )
    return upd


if __name__ == '__main__':
  import os
  # get just this file's module name: no `.py` and no path
  name = os.path.basename(__file__).replace(".py", "")
  unittest.TextTestRunner(failfast=True).run(unittest.TestLoader().loadTestsFromName(name))
