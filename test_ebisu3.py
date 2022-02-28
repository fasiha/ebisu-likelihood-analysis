"""
run as 
$ python -m nose test_ebisu3.py
or
$ python test_ebisu3.py
"""

from itertools import product
from functools import cache
import ebisu3 as ebisu
import unittest
from scipy.stats import gamma as gammarv, binom as binomrv, bernoulli  # type: ignore
from scipy.special import logsumexp  # type: ignore
import numpy as np
from typing import Optional, Union
import math
import time
from copy import deepcopy

MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
weightedGammaEstimate = ebisu._weightedGammaEstimate


def _gammaToVar(alpha: float, beta: float) -> float:
  return alpha / beta**2


def fullBinomialMonteCarlo(
    hlPrior: tuple[float, float],
    bPrior: tuple[float, float],
    ts: list[float],
    results: list[ebisu.Result],
    left=0.3,
    size=1_000_000,
):
  hl0s = gammarv.rvs(hlPrior[0], scale=1 / hlPrior[1], size=size)
  boosts = gammarv.rvs(bPrior[0], scale=1 / bPrior[1], size=size)

  logweights = np.zeros(size)

  hls = hl0s.copy()
  for t, res in zip(ts, results):
    logps = -t / hls * np.log(2)
    success = ebisu.success(res)
    if isinstance(res, ebisu.BinomialResult):
      if success:
        logweights += logps
      else:
        logweights += np.log(-np.expm1(logps))
      # This is the likelihood of observing the data, and is more accurate than
      # `binomrv.logpmf(k, n, pRecall)` since `pRecall` is already in log domain
    else:
      q0, q1 = res.q0, res.q1
      if not success:
        q0, q1 = 1 - q0, 1 - q1
      logpfails = np.log(-np.expm1(logps))
      logweights += logsumexp(np.vstack([logps + np.log(q1), logpfails + np.log(q0)]), axis=0)
    # Apply boost for successful quizzes
    if success:  # reuse same rule as ebisu
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
  e, a = np.array(expected), np.array(actual)
  return np.abs(a - e) / np.abs(e)


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

    init = ebisu.initModel(initHlPrior, boostPrior)

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

        msg = f'result={result}, fraction={fraction} => currHl={updated.pred.currentHalflife}'
        if result:
          self.assertTrue(updated.pred.currentHalflife >= initHlMean, msg)
        else:
          self.assertTrue(updated.pred.currentHalflife <= initHlMean, msg)

        # this is the unboosted posterior update
        u2 = ebisu._gammaUpdateBinomial(initHlPrior[0], initHlPrior[1], elapsedHours, result, 1)

        # this uses the two-point formula: y=(y2-y1)/(x2-x1)*(x-x1) + y1, where
        # y represents the boost fraction and x represents the time elapsed as
        # a fraction of the initial halflife
        boostFraction = (boostMean - 1) / (1 - left) * (fraction - left) + 1

        # clamp 1 <= boost <= boostMean, and only boost successes
        boost = min(boostMean, max(1, boostFraction)) if result else 1
        self.assertAlmostEqual(updated.pred.currentHalflife, boost * u2.mean)

        for nextResult in [1, 0]:
          for i in range(3):
            nextElapsed, boost = updated.pred.currentHalflife, boostMean
            nextUpdate = ebisu.simpleUpdateRecall(
                updated,
                nextElapsed,
                nextResult,
                now=nowMs + (elapsedHours + (i + 1) * nextElapsed) * MILLISECONDS_PER_HOUR,
                left=left,
            )

            initMean = lambda model: ebisu._gammaToMean(*model.prob.initHl)

            # confirm the initial halflife estimate rose/dropped
            if nextResult:
              self.assertGreater(initMean(nextUpdate), 1.05 * initMean(updated))
            else:
              self.assertLess(initMean(nextUpdate), 1.05 * initMean(updated))

            # this checks the scaling applied to take the new Gamma to the initial Gamma in simpleUpdateRecall
            self.assertGreater(nextUpdate.pred.currentHalflife, 1.1 * initMean(nextUpdate))

            # meanwhile this checks the scaling to convert the initial halflife Gamma and the current halflife mean
            currHlPrior, _ = ebisu._currentHalflifePrior(updated)
            self.assertAlmostEqual(updated.pred.currentHalflife,
                                   gammarv.mean(currHlPrior[0], scale=1 / currHlPrior[1]))

            if nextResult:
              # this is an almost tautological test but just as a sanity check, confirm that boosts are being applied?
              next2 = ebisu._gammaUpdateBinomial(currHlPrior[0], currHlPrior[1], nextElapsed,
                                                 nextResult, 1)
              self.assertAlmostEqual(nextUpdate.pred.currentHalflife, next2.mean * boost)
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

    init = ebisu.initModel(initHlPrior, boostPrior)

    import mpmath as mp  # type:ignore

    left = 0.3
    # simulate a variety of 4-quiz trajectories:
    for fraction, result, lastNoisy in product([0.1, 0.5, 1.5, 9.5], [1, 0], [False, True]):
      upd = deepcopy(init)
      elapsedHours = fraction * initHlMean
      upd = ebisu.simpleUpdateRecall(upd, elapsedHours, result)

      for nextResult, nextElapsed, nextTotal in zip(
          [1, 1, 1 if not lastNoisy else (0.8 if result else 0.2)],
          [elapsedHours * 3, elapsedHours * 5, elapsedHours * 7],
          [1, 1, 2 if not lastNoisy else 1],
      ):
        upd = ebisu.simpleUpdateRecall(upd, nextElapsed, nextResult, total=nextTotal, q0=0.05)

      ### Full Ebisu update (max-likelihood to enhanced Monte Carlo proposal)
      # 100_000 samples is probably WAY TOO MANY for practical purposes but
      # here I want to ascertain that this approach is correct as you crank up
      # the number of samples. If we have confidence that this estimator behaves
      # correctly, we can in practice use 1_000 or 10_000 samples and accept a
      # less accurate model but remain confident that the *means* of this posterior
      # are accurate.
      tmp: tuple[ebisu.Model, dict] = ebisu.fullUpdateRecall(
          upd, left=left, size=100_000, debug=True)
      full, fullDebug = tmp

      ### Numerical integration via mpmath
      # This method stops being accurate when you have tens of quizzes but it's matches
      # the other methods well for ~4. It also can only compute posterior moments.
      @cache
      def posterior2d(b, h):
        return mp.exp(ebisu._posterior(float(b), float(h), upd, 0.3, 1.0))

      def integration(maxdegree: int, wantVar: bool = False):
        method = 'gauss-legendre'
        f0 = lambda b, h: posterior2d(b, h)
        den = mp.quad(f0, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)
        fb = lambda b, h: b * posterior2d(b, h)
        numb = mp.quad(fb, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)
        fh = lambda b, h: h * posterior2d(b, h)
        numh = mp.quad(fh, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)

        # second non-central moment
        if wantVar:
          fh = lambda b, h: h**2 * posterior2d(b, h)
          numh2 = mp.quad(fh, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)
          fb = lambda b, h: b**2 * posterior2d(b, h)
          numb2 = mp.quad(fb, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method)

        boostMeanInt, hl0MeanInt = numb / den, numh / den
        if wantVar:
          boostVarInt, hl0VarInt = numb2 / den - boostMeanInt**2, numh2 / den - hl0MeanInt**2
          return boostMeanInt, hl0MeanInt, boostVarInt, hl0VarInt
        return boostMeanInt, hl0MeanInt

      boostMeanInt, hl0MeanInt = integration(5)

      AB_ERR = .05
      FULL_INT_MEAN_ERR = 0.03
      MC_INT_MEAN_ERR = 0.03
      ### Raw Monte Carlo simulation (without max likelihood enhanced proposal)
      # Because this method can be inaccurate and slow, try it with a small number
      # of samples and increase it quickly if we don't meet tolerances.
      for size in [10_000, 100_000, 1_000_000, 10_000_000]:
        mc = fullBinomialMonteCarlo(
            init.prob.initHlPrior,
            init.prob.boostPrior,
            upd.quiz.elapseds[-1],
            upd.quiz.results[-1],
            size=size)
        ab_err = np.max(
            relativeError([full.prob.boost, full.prob.initHl],
                          [mc["posteriorBoost"], mc["posteriorInitHl"]]))
        full_int_mean_err_hl0 = relativeError(ebisu._gammaToMean(*full.prob.initHl), hl0MeanInt)
        mc_int_mean_err_hl0 = relativeError(ebisu._gammaToMean(*mc["posteriorInitHl"]), hl0MeanInt)
        full_int_mean_err_b = relativeError(ebisu._gammaToMean(*full.prob.boost), boostMeanInt)
        mc_int_mean_err_b = relativeError(ebisu._gammaToMean(*mc["posteriorBoost"]), boostMeanInt)
        if (ab_err < AB_ERR and full_int_mean_err_hl0 < FULL_INT_MEAN_ERR and
            mc_int_mean_err_hl0 < MC_INT_MEAN_ERR and full_int_mean_err_b < FULL_INT_MEAN_ERR and
            mc_int_mean_err_b < MC_INT_MEAN_ERR):
          break
      if verbose:
        errs = [
            float(x) for x in [
                ab_err, full_int_mean_err_hl0, mc_int_mean_err_hl0, full_int_mean_err_b,
                mc_int_mean_err_b
            ]
        ]
        indiv_ab_err = relativeError([full.prob.boost, full.prob.initHl],
                                     [mc["posteriorBoost"], mc["posteriorInitHl"]]).ravel()

        print(
            f"size={size:0.2g}, max={max(errs):0.3f}, errs={', '.join([f'{e:0.3f}' for e in errs])}, ab_err={indiv_ab_err}"
        )

      ### Finally, compare all three updates above.
      # Since numerical integration only gave us means, we can compare its means to
      # (a) full Ebisu update and (b) raw Monte Carlo. Also of course compare the
      # fit (α, β) for both random variables (initial halflife and boost) between
      # full Ebisu vs raw Monte Carlo.
      ab_err = np.max(
          relativeError([full.prob.boost, full.prob.initHl],
                        [mc["posteriorBoost"], mc["posteriorInitHl"]]))
      full_int_mean_err_hl0 = relativeError(ebisu._gammaToMean(*full.prob.initHl), hl0MeanInt)
      mc_int_mean_err_hl0 = relativeError(ebisu._gammaToMean(*mc["posteriorInitHl"]), hl0MeanInt)
      full_int_mean_err_b = relativeError(ebisu._gammaToMean(*full.prob.boost), boostMeanInt)
      mc_int_mean_err_b = relativeError(ebisu._gammaToMean(*mc["posteriorBoost"]), boostMeanInt)

      self.assertLess(ab_err, AB_ERR, f'analytical ~ mc, {fraction=}, {result=}, {lastNoisy=}')
      self.assertLess(
          full_int_mean_err_hl0, FULL_INT_MEAN_ERR,
          f'analytical ~ numerical integration mean hl0, {fraction=}, {result=}, {lastNoisy=}')
      self.assertLess(
          mc_int_mean_err_hl0, MC_INT_MEAN_ERR,
          f'monte carlo ~ numerical integration mean hl0, {fraction=}, {result=}, {lastNoisy=}')

      self.assertLess(
          full_int_mean_err_b, FULL_INT_MEAN_ERR,
          f'analytical ~ numerical integration mean boost, {fraction=}, {result=}, {lastNoisy=}')
      self.assertLess(
          mc_int_mean_err_b, MC_INT_MEAN_ERR,
          f'monte carlo ~ numerical integration mean boost, {fraction=}, {result=}, {lastNoisy=}')
    return upd


if __name__ == '__main__':
  import os
  # get just this file's module name: no `.py` and no path
  name = os.path.basename(__file__).replace(".py", "")
  unittest.TextTestRunner(failfast=True).run(unittest.TestLoader().loadTestsFromName(name))
