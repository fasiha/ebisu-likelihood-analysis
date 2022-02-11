"""
run as 
$ python -m nose test_ebisu3.py
or
$ python test_ebisu3.py
"""

import ebisu3 as ebisu
import unittest
from scipy.stats import gamma as gammarv, binom as binomrv, bernoulli  # type: ignore
from scipy.special import logsumexp, loggamma  # type: ignore
from scipy.optimize import shgo, minimize  # type: ignore
import numpy as np
from typing import Optional, Union
import math
from dataclasses import dataclass, replace
from utils import sequentialImportanceResample

MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


def _gammaToVar(alpha: float, beta: float) -> float:
  return alpha / beta**2


def fitJointTwoGammas(x, y):
  sumlogs = [np.sum(np.log(v)) for v in [x, y]]
  means = [np.mean(v) for v in [x, y]]
  n = len(x)
  loglik1 = lambda k, sumlog, mean: (
      (k - 1) * sumlog - n * k - n * k * np.log(mean / k) - n * loggamma(k))
  loglik = lambda ks: loglik1(ks[0], sumlogs[0], means[0]) + loglik1(ks[1], sumlogs[1], means[1])
  sol = shgo(lambda ks: -loglik(ks), [(0.01, 50)] * 2)
  kx, ky = sol.x
  thetax, thetay = np.array(means) / sol.x

  alphax, alphay = [kx, ky]
  betax, betay = [1 / thetax, 1 / thetay]

  return dict(
      sol=sol,
      alphax=alphax,
      betax=betax,
      alphay=alphay,
      betay=betay,
      meanX=ebisu._gammaToMean(alphax, betax),
      meanY=ebisu._gammaToMean(alphay, betay),
  )


def fitJointTwoGammasWeighted2(x, y, w):
  "2D search via fixing theta in terms of k"
  wsum = math.fsum(w)

  meanlogs = [math.fsum(w * np.log(v)) / wsum for v in [x, y]]
  means = [math.fsum(w * v) / wsum for v in [x, y]]
  n = len(x)
  loglik1 = lambda k, meanlog, mean: (
      (k - 1) * n * meanlog - n * k - n * k * np.log(mean / k) - n * loggamma(k))
  loglik = lambda ks: loglik1(ks[0], meanlogs[0], means[0]) + loglik1(ks[1], meanlogs[1], means[1])
  sol = shgo(lambda ks: -loglik(ks), [(0.01, 50)] * 2)
  kx, ky = sol.x
  thetax, thetay = np.array(means) / sol.x

  alphax, alphay = [kx, ky]
  betax, betay = [1 / thetax, 1 / thetay]

  return dict(
      sol=sol,
      alphax=alphax,
      betax=betax,
      alphay=alphay,
      betay=betay,
      meanX=ebisu._gammaToMean(alphax, betax),
      meanY=ebisu._gammaToMean(alphay, betay),
  )


def fitJointTwoGammasWeighted(x, y, w):
  "4D search"
  wsum = math.fsum(w)

  meanlogs = [math.fsum(w * np.log(v)) / wsum for v in [x, y]]
  means = [math.fsum(w * v) / wsum for v in [x, y]]
  # divide through by n=len(x):
  loglik1 = lambda k, t, meanlog, mean: (k - 1) * meanlog - mean / t - k * np.log(t) - loggamma(k)
  loglik = lambda kts: (
      loglik1(kts[0], kts[1], meanlogs[0], means[0]) + loglik1(kts[2], kts[3], meanlogs[1], means[1]
                                                              ))
  if False:
    sol = shgo(lambda x: -loglik(x), [(0.01, 50)] * 4)
    kx, thetax, ky, thetay = sol.x
    alphax, alphay = [kx, ky]
    betax, betay = [1 / thetax, 1 / thetay]
  else:
    # convert alpha/beta Gamma parameterization to k/theta: k=alpha, beta=1/theta
    fits = [weightedGammaEstimate(v, w) for v in [x, y]]
    init = np.array([fits[0][0], 1 / fits[0][1], fits[1][0], 1 / fits[1][1]])
    bounds = [(0, np.inf)] * 4

    sol = minimize(lambda x: -loglik(x), init, bounds=bounds, method='Nelder-Mead')
    # Weird, shgo doesn't converge (picks the mid-point of bounds) but Nelder-Mead does much better
    # But point remains the that ML-fitting the posterior samples to bivariate independent Gamma
    alphax = sol.x[0]
    betax = 1 / sol.x[1]
    alphay = sol.x[2]
    betay = 1 / sol.x[3]
  return dict(
      sol=sol,
      alphax=alphax,
      betax=betax,
      alphay=alphay,
      betay=betay,
      meanXY=[ebisu._gammaToMean(alphax, betax),
              ebisu._gammaToMean(alphay, betay)],
  )


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
  # posteriorBoost = weightedMeanLogw(logweights, boosts)
  # posteriorInitHl = weightedMeanLogw(logweights, hl0s)
  # posteriorCurrHl = weightedMeanLogw(logweights, hls)
  # sisHl0s = sequentialImportanceResample(hl0s, w)[0]
  w = np.exp(logweights)
  estb = weightedGammaEstimate(boosts, w)
  esthl0 = weightedGammaEstimate(hl0s, w)
  # esthl = ebisu._gammaToMean(*weightedGammaEstimate(hls, w))
  return dict(
      kishEffectiveSampleSize=kishEffectiveSampleSize,
      posteriorBoost=estb,
      posteriorInitHl=esthl0,
      statsBoost=weightedMeanVarLogw(logweights, boosts),
      statsInitHl=weightedMeanVarLogw(logweights, hl0s),
      # modeHl0=modeHl0,
      # corr=np.corrcoef(np.vstack([hl0s, hls, boosts])),
      fit=fitJointTwoGammasWeighted(boosts, hl0s, w)
      # posteriorCurrHl=posteriorCurrHl,
      # estb=estb,
      # esthl0=esthl0,
      # esthl=esthl,
  )


def weightedGammaEstimate(h, w):
  """
  See https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1067698046#Closed-form_estimators
  """
  wsum = math.fsum(w)
  that2 = np.sum(w * h * np.log(h)) / wsum - np.sum(w * h) / wsum * np.sum(w * np.log(h)) / wsum
  khat2 = np.sum(w * h) / wsum / that2
  fit = (khat2, 1 / that2)
  return fit


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


def weightedMeanLogw(logw: np.ndarray, x: np.ndarray) -> np.ndarray:
  # [weightedMean] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  return np.exp(logsumexp(logw, b=x) - logsumexp(logw))


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

  def test_full(self):
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
    for fraction in [1.5]:
      for result in [1]:
        upd = replace(init)
        elapsedHours = fraction * initHlMean
        ebisu._appendQuiz(upd, elapsedHours, ebisu.BinomialResult(result, 1), 1.0)

        for nextResult, nextElapsed in zip([1, 1, 1],
                                           [elapsedHours * 3, elapsedHours * 5, elapsedHours * 7]):
          ebisu._appendQuiz(upd, nextElapsed, ebisu.BinomialResult(nextResult, 1), 1.0)

        full = ebisu.fullUpdateRecall(upd, left=left)

        import mpmath as mp  # type:ignore
        f0 = lambda b, h: mp.exp(ebisu._posterior(float(b), float(h), upd, 0.3, 1.0))
        den = mp.quad(f0, [0, mp.inf], [0, mp.inf])
        fb = lambda b, h: b * mp.exp(ebisu._posterior(float(b), float(h), upd, 0.3, 1.0))
        numb = mp.quad(fb, [0, mp.inf], [0, mp.inf])
        fh = lambda b, h: h * mp.exp(ebisu._posterior(float(b), float(h), upd, 0.3, 1.0))
        numh = mp.quad(fh, [0, mp.inf], [0, mp.inf])
        boostMeanInt, hl0MeanInt = numb / den, numh / den

        # second non-central moment
        fh = lambda b, h: h**2 * mp.exp(ebisu._posterior(float(b), float(h), upd, 0.3, 1.0))
        numh2 = mp.quad(fh, [0, mp.inf], [0, mp.inf])
        fb = lambda b, h: b**2 * mp.exp(ebisu._posterior(float(b), float(h), upd, 0.3, 1.0))
        numb2 = mp.quad(fb, [0, mp.inf], [0, mp.inf])
        boostVarInt, hl0VarInt = numb2 / den - boostMeanInt**2, numh2 / den - hl0MeanInt**2

        mc = fullBinomialMonteCarlo(
            init.initHalflifePrior,
            init.boostPrior, [t for t in upd.elapseds[-1]], [r.successes for r in upd.results[-1]],
            [1 for t in upd.elapseds[-1]],
            size=1_000_000)

        if True:
          print(f'an={full.initHalflifePrior}; mc={mc["posteriorInitHl"]}')
          print(
              f'mean: an={ebisu._gammaToMean(*full.initHalflifePrior)}; mc={ebisu._gammaToMean(*mc["posteriorInitHl"])}; rawMc={mc["statsInitHl"][0]}; int={hl0MeanInt}'
          )
          print(
              f'VAR: an={_gammaToVar(*full.initHalflifePrior)}; mc={_gammaToVar(*mc["posteriorInitHl"])}; rawMc={mc["statsInitHl"][1]}; int={hl0VarInt}'
          )

          print(f'an={full.boostPrior}; mc={mc["posteriorBoost"]}')
          print(
              f'mean: an={ebisu._gammaToMean(*full.boostPrior)}; mc={ebisu._gammaToMean(*mc["posteriorBoost"])}; rawMc={mc["statsBoost"][0]}; int={boostMeanInt}'
          )
          print(
              f'VAR: an={_gammaToVar(*full.boostPrior)}; mc={_gammaToVar(*mc["posteriorBoost"])}; rawMc={mc["statsBoost"][1]}; int={boostVarInt}'
          )

        self.assertLess(
            relativeError(
                ebisu._gammaToMean(*full.initHalflifePrior),
                ebisu._gammaToMean(*mc["posteriorInitHl"])),
            0.1,
            'analytical ~ monte carlo mean hl0',
        )
        self.assertLess(
            relativeError(hl0MeanInt, ebisu._gammaToMean(*mc["posteriorInitHl"])),
            0.005,
            'numerical integration ~ monte carlo mean hl0',
        )

        self.assertLess(
            relativeError(
                ebisu._gammaToMean(*full.boostPrior), ebisu._gammaToMean(*mc["posteriorBoost"])),
            0.1,
            'analytical ~ monte carlo mean boost',
        )
        self.assertLess(
            relativeError(boostMeanInt, ebisu._gammaToMean(*mc["posteriorBoost"])),
            0.005,
            'numerical integration ~ numerical integration mean boost',
        )
        return upd


def vizPosterior(ret: ebisu.Model,
                 size=10_000_000,
                 left=0.3,
                 right=1.0,
                 fit2total=202,
                 weightPower=0.0):
  import pylab as plt  #type:ignore
  plt.ion()

  mc = fullBinomialMonteCarlo(
      ret.initHalflifePrior,
      ret.boostPrior,
      [t for t in ret.elapseds[-1]],
      [r.successes for r in ret.results[-1]],
      [1 for t in ret.elapseds[-1]],
      size=size,
  )
  print(f'{mc=}')

  MIN_BOOST = 1.0
  ab, bb = ret.boostPrior
  ah, bh = ret.initHalflifePrior
  minBoost, midBoost, maxBoost = gammarv.ppf([.01, 0.7, 0.9999], ab, scale=1.0 / bb)
  minHalflife, midHalflife, maxHalflife = gammarv.ppf([0.01, 0.7, 0.9999], ah, scale=1.0 / bh)
  bvec = np.linspace(minBoost, maxBoost, 101)
  hvec = np.linspace(minHalflife, maxHalflife, 101)
  bmat, hmat = np.meshgrid(bvec, hvec)

  posterior2d = lambda b, h: ebisu._posterior(b, h, ret, left, right)

  z = np.vectorize(posterior2d)(bmat, hmat)
  opt = shgo(lambda x: -posterior2d(x[0], x[1]), [(MIN_BOOST, maxBoost),
                                                  (minHalflife, maxHalflife)])
  bestb, besth = opt.x

  peak = np.max(z)
  cutoff = peak - 10
  idx = z.ravel() > cutoff
  fitall = ebisu._fitJointToTwoGammas(
      bmat.ravel()[idx], hmat.ravel()[idx], z.ravel()[idx], weightPower=1.0)
  print(f'size={bmat.ravel()[idx].size}')
  fitall['meanX'] = ebisu._gammaToMean(fitall['alphax'], fitall['betax'])
  fitall['meanY'] = ebisu._gammaToMean(fitall['alphay'], fitall['betay'])

  def mkfit(maxBoost, maxHalflife, nb=201, nh=201, weightPower=0.0):
    bs = []
    hs = []
    posteriors = []
    bvec = np.linspace(MIN_BOOST, maxBoost, nb)
    for b in bvec:
      posteriors.append(posterior2d(b, besth))
      bs.append(b)
      hs.append(besth)
    hvec = np.linspace(minHalflife, maxHalflife, nh)
    for h in hvec:
      posteriors.append(posterior2d(bestb, h))
      bs.append(bestb)
      hs.append(h)

    if True:  # if omit tiny posteriors
      maxp = np.max(posteriors)
      cutoff = 6
      bs, hs, posteriors = zip(
          *[(b, h, p) for b, h, p in zip(bs, hs, posteriors) if p > (maxp - cutoff)])
    print(f'cutoff len={len(bs)}')
    fit = ebisu._fitJointToTwoGammas(bs, hs, posteriors, weightPower=weightPower)
    fit['meanX'] = ebisu._gammaToMean(fit['alphax'], fit['betax'])
    fit['meanY'] = ebisu._gammaToMean(fit['alphay'], fit['betay'])
    fit['varX'] = _gammaToVar(fit['alphax'], fit['betax'])
    fit['varY'] = _gammaToVar(fit['alphay'], fit['betay'])

    if True:
      remmax = lambda v: np.array(v) - np.max(v)
      fig, ax = plt.subplots(2, 1)
      ax[0].plot(
          bvec,
          remmax([posterior2d(b, besth) for b in bvec]),
          marker='.',
          label='analytical posterior')
      ax[0].plot(
          bvec,
          remmax(gammarv.logpdf(bvec, fit['alphax'], scale=1 / fit['betax'])),
          marker='.',
          linestyle='--',
          label='posterior curve fit')
      ax[0].plot(
          bvec,
          remmax(gammarv.logpdf(bvec, mc['fit']['alphax'], scale=1 / mc['fit']['betax'])),
          marker='.',
          label='Monte Carlo fit')
      ax[0].legend()

      ax[1].plot(
          hvec,
          remmax([posterior2d(bestb, h) for h in hvec]),
          marker='.',
          label='analytical posterior')
      ax[1].plot(
          hvec,
          remmax(gammarv.logpdf(hvec, fit['alphay'], scale=1 / fit['betay'])),
          marker='.',
          linestyle='--',
          label='posterior curve fit')
      ax[1].plot(
          hvec,
          remmax(gammarv.logpdf(hvec, mc['fit']['alphay'], scale=1 / mc['fit']['betay'])),
          marker='.',
          label='Monte Carlo fit')
      ax[1].legend()

    return fit

  def mkfit2(maxBoost, maxHalflife, total=202):
    bvec = np.linspace(MIN_BOOST, maxBoost, 101)
    hvec = np.linspace(minHalflife, maxHalflife, 101)

    bs = np.random.rand(total) * (maxBoost - MIN_BOOST) + MIN_BOOST
    hs = np.random.rand(total) * (maxHalflife - minHalflife) + minHalflife

    bs = np.hstack([bs, bvec])
    hs = np.hstack([hs, hvec])

    posteriors = np.vectorize(posterior2d)(bs, hs)
    fit = ebisu._fitJointToTwoGammas(bs, hs, posteriors, weightPower=3.0)
    fit['meanX'] = ebisu._gammaToMean(fit['alphax'], fit['betax'])
    fit['meanY'] = ebisu._gammaToMean(fit['alphay'], fit['betay'])
    return fit

  rescalec = lambda im, top: im.set_clim(im.get_clim()[1] - np.array([top, 0]))

  def imshow(x, y, z, ax=plt):

    def extents(f):
      delta = f[1] - f[0]
      return [f[0] - delta / 2, f[-1] + delta / 2]

    return ax.imshow(
        z, aspect='auto', interpolation='none', extent=extents(x) + extents(y), origin='lower')

  fig, ax = plt.subplots()
  im = imshow(bvec, hvec, z, ax=ax)
  ax.set_xlabel('boost')
  ax.set_ylabel('init halflife')
  fig.colorbar(im)
  rescalec(im, 20)

  fit = mkfit(maxBoost, maxHalflife, weightPower=weightPower)
  fit2 = mkfit2(maxBoost, maxHalflife, total=fit2total)
  return dict(fig=fig, ax=ax, im=im, bestb=bestb, besth=besth, fit=fit, fit2=fit2, fitall=fitall)


def integrals(model):
  import mpmath as mp  # type:ignore
  f0 = lambda b, h: mp.exp(ebisu._posterior(float(b), float(h), model, 0.3, 1.0))
  den = mp.quad(f0, [0, mp.inf], [0, mp.inf])
  fb = lambda b, h: b * mp.exp(ebisu._posterior(float(b), float(h), model, 0.3, 1.0))
  numb = mp.quad(fb, [0, mp.inf], [0, mp.inf])
  fh = lambda b, h: h * mp.exp(ebisu._posterior(float(b), float(h), model, 0.3, 1.0))
  numh = mp.quad(fh, [0, mp.inf], [0, mp.inf])
  return dict(meanXY=[numb / den, numh / den], vals=[numb, numh, den])


if __name__ == '__main__':
  import os
  # get just this file's module name: no `.py` and no path
  name = os.path.basename(__file__).replace(".py", "")
  # unittest.TextTestRunner(failfast=True).run(unittest.TestLoader().loadTestsFromName(name))
  t = TestEbisu()
  t.test_full()

if False:
  t = TestEbisu()
  upd = t.test_full()
  clean = replace(upd)
  clean.elapseds = [[]]
  clean.results = [[]]
  clean.startStrengths = [[]]
  from scipy.optimize import shgo  # type: ignore
  import pylab as plt  #type:ignore
  plt.ion()

  rescalec = lambda im, top: im.set_clim(im.get_clim()[1] - np.array([top, 0]))

  v = vizPosterior(upd, size=1_000_000, fit2total=2002, weightPower=0.0)
  print({k: v['fit'][k] for k in v['fit'] if k != "sol"})
  print(integrals(upd))
  # print([f'{k}={v[k]["meanY"]}' for k in ['fitall', 'fit2', 'fit']])
  # v2 = vizPosterior(clean, size=100_000, fit2total=2002, weightPower=0.0)
  import utils
  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')
  train, TEST_TRAIN = utils.traintest(df)

  t = train[10]
  data = replace(clean)
  data.elapseds = [[t for t in t.dts_hours]]
  data.results = [[ebisu.BinomialResult(1 if x >= 2 else 0, 1) for x in t.results]]
  data.startStrengths = [[1.0 for t in t.dts_hours]]
  datav = vizPosterior(data, size=1_000_000, fit2total=2002, weightPower=0.0)
  print([f'{k}={datav[k]["meanY"]}' for k in ['fitall', 'fit2', 'fit']])

  mom = integrals(data)
  print(mom)

  # datav2 = vizPosterior(data, size=1_000_000, fit2total=2002, weightPower=0.0, left=0.8)
