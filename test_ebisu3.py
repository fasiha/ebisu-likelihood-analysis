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
from copy import deepcopy
import pickle
from datetime import datetime
import mpmath as mp  # type:ignore
import csv
import io

MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
weightedGammaEstimate = ebisu._weightedGammaEstimate
weightedMeanVarLogw = ebisu._weightedMeanVarLogw


def numericalIntegration(upd: ebisu.Model, maxdegree: int, left=0.3, right=1.0):

  @cache
  def posterior2d(b, h):
    if b == 0 or h == 0:
      return 0
    return mp.exp(ebisu._posterior(float(b), float(h), upd, left, right))

  method = 'tanh-sinh'
  errs = dict()
  res = dict()
  f0 = lambda b, h: posterior2d(b, h)
  res['den'], errs['den'] = mp.quad(
      f0, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method, error=True)
  fb = lambda b, h: b * posterior2d(b, h)
  res['numb'], errs['numb'] = mp.quad(
      fb, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method, error=True)
  fh = lambda b, h: h * posterior2d(b, h)
  res['numh'], errs['numh'] = mp.quad(
      fh, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method, error=True)

  # second non-central moment
  fh = lambda b, h: h**2 * posterior2d(b, h)
  res['numh2'], errs['numh2'] = mp.quad(
      fh, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method, error=True)
  fb = lambda b, h: b**2 * posterior2d(b, h)
  res['numb2'], errs['numb2'] = mp.quad(
      fb, [0, mp.inf], [0, mp.inf], maxdegree=maxdegree, method=method, error=True)

  boostMeanInt, hl0MeanInt = res['numb'] / res['den'], res['numh'] / res['den']

  bSecondMoment = res['numb2'] / res['den']
  hSecondMoment = res['numh2'] / res['den']

  boostVarInt, hl0VarInt = bSecondMoment - boostMeanInt**2, hSecondMoment - hl0MeanInt**2
  return [(boostMeanInt, boostVarInt, bSecondMoment, mp.sqrt(bSecondMoment)),
          (hl0MeanInt, hl0VarInt, hSecondMoment, mp.sqrt(hSecondMoment))]


def fullBinomialMonteCarlo(
    hlPrior: tuple[float, float],
    bPrior: tuple[float, float],
    ts: list[float],
    results: list[ebisu.Result],
    left=0.3,
    right=1.0,
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
      logweights += ebisu._logBinomPmfLogp(res.total, res.successes, logps)
      # This is the likelihood of observing the data, and is more accurate than
      # `binomrv.logpmf(k, n, pRecall)` since `pRecall` is already in log domain
    else:
      q0, q1 = res.q0, res.q1
      logpfails = np.log(-np.expm1(logps))
      # exact transcription of log_mix from Stan and https://github.com/fasiha/ebisu/issues/52
      logweights += logsumexp(
          np.vstack([
              logps + bernoulli.logpmf(int(success), q1),
              logpfails + bernoulli.logpmf(int(success), q0)
          ]),
          axis=0)
    # Apply boost for successful quizzes
    if success:  # reuse same rule as ebisu
      hls *= clampLerp(left * hls, right * hls, np.ones(size), np.maximum(boosts, 1.0), t)

  kishEffectiveSampleSize = np.exp(2 * logsumexp(logweights) - logsumexp(2 * logweights)) / size
  return dict(
      kishEffectiveSampleSize=kishEffectiveSampleSize,
      statsBoost=weightedMeanVarLogw(logweights, boosts),
      statsInitHl=weightedMeanVarLogw(logweights, hl0s),
      size=size,
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


results = dict()
testStartTime = datetime.utcnow().isoformat()

# seed = 29907812
seed = np.random.randint(1, 1_000_000_000)
# seed = 708572856  # fails?
# seed = 473107421 # fails kish?
print(f'{seed=}')


class TestEbisu(unittest.TestCase):

  def setUp(self):
    np.random.seed(seed=seed)  # for sanity when testing with Monte Carlo

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

    now = ebisu._timeMs()
    init = ebisu.initModel(initHlPrior=initHlPrior, boostPrior=boostPrior, now=now)

    left = 0.3
    right = 1.0

    for fraction, result in product([0.1, 0.5, 1.0, 2.0, 10.0], [0, 1]):
      elapsedHours = fraction * initHlMean
      now2 = now + elapsedHours * MILLISECONDS_PER_HOUR
      updated = ebisu.updateRecall(init, result, total=1, now=now2, left=left, right=right)

      msg = f'result={result}, fraction={fraction} => currHl={updated.pred.currentHalflifeHours}'
      if result:
        self.assertTrue(updated.pred.currentHalflifeHours >= initHlMean, msg)
      else:
        self.assertTrue(updated.pred.currentHalflifeHours <= initHlMean, msg)

      # this is the unboosted posterior update
      u2 = ebisu._gammaUpdateBinomial(initHlPrior[0], initHlPrior[1], elapsedHours, result, 1)

      # this uses the two-point formula: y=(y2-y1)/(x2-x1)*(x-x1) + y1, where
      # y represents the boost fraction and x represents the time elapsed as
      # a fraction of the initial halflife
      boostFraction = (boostMean - 1) / (right - left) * (fraction - left) + 1

      # clamp 1 <= boost <= boostMean, and only boost successes
      boost = min(boostMean, max(1, boostFraction)) if result else 1
      self.assertAlmostEqual(
          updated.pred.currentHalflifeHours,
          boost * u2.mean,
          msg=f'{fraction=}, {result=}, {boost=}')

      for nextResult in [1, 0]:
        for i in range(3):
          nextElapsed, boost = updated.pred.currentHalflifeHours, boostMean
          now2 += nextElapsed * MILLISECONDS_PER_HOUR
          nextUpdate = ebisu.updateRecall(updated, nextResult, now=now2, left=left, right=right)

          initMean = lambda model: ebisu._gammaToMean(*model.prob.initHl)

          # confirm the initial halflife estimate rose/dropped
          if nextResult:
            self.assertGreater(initMean(nextUpdate), 1.05 * initMean(updated))
          else:
            self.assertLess(
                initMean(nextUpdate), 1.05 * initMean(updated), msg=f'{fraction=}, {result=}')

          # this checks the scaling applied to take the new Gamma to the initial Gamma in simpleUpdateRecall
          self.assertGreater(nextUpdate.pred.currentHalflifeHours, 1.1 * initMean(nextUpdate))

          # meanwhile this checks the scaling to convert the initial halflife Gamma and the current halflife mean
          currHlPrior, _ = ebisu._currentHalflifePrior(updated)
          self.assertAlmostEqual(updated.pred.currentHalflifeHours,
                                 gammarv.mean(currHlPrior[0], scale=1 / currHlPrior[1]))

          if nextResult:
            # this is an almost tautological test but just as a sanity check, confirm that boosts are being applied?
            next2 = ebisu._gammaUpdateBinomial(currHlPrior[0], currHlPrior[1], nextElapsed,
                                               nextResult, 1)
            self.assertAlmostEqual(nextUpdate.pred.currentHalflifeHours, next2.mean * boost)
            # don't test this for failures: no boost is applied then

          updated = nextUpdate

        # don't bother to check alpha/beta: a test in Python will just be tautological
        # (we'll repeat the same thing in the test as in the code). That has to happen
        # via Stan?

  def test_1_then_0s(self, verbose=False):
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    boostMean = 1.5
    boostBeta = 3.0
    boostPrior = (boostBeta * boostMean, boostBeta)

    now = ebisu._timeMs()
    base = ebisu.initModel(initHlPrior=initHlPrior, boostPrior=boostPrior, now=now)

    ts = [20., 10., 5., 4., 3., 2., 1.]  # hours elapsed for each quiz
    correct_ts = [ts[0]]  # just one success

    fulls: list[ebisu.Model] = [base]
    for t in ts:
      now += t * MILLISECONDS_PER_HOUR
      tmp = ebisu.updateRecall(fulls[-1], int(t in correct_ts), now=now)
      fulls.append(ebisu.updateRecallHistory(tmp, size=10_000))
    if verbose:
      print('\n'.join([
          f'FULL curr={m.pred.currentHalflifeHours}, init={ebisu._gammaToMean(*m.prob.initHl)}, bmean={ebisu._gammaToMean(*m.prob.boost)}'
          for m in fulls[1:]
      ]))

    # require monotonically decreasing initHalflife, current halflife, and boost
    # since we have an initial success followed by a string of failures
    m = lambda tup: ebisu._gammaToMean(*tup)
    # print('hls', [f.pred.currentHalflifeHours for f in fulls])
    for left, right in zip(fulls[1:], fulls[2:]):
      self.assertGreater(left.pred.currentHalflifeHours, right.pred.currentHalflifeHours)
      self.assertGreater(m(left.prob.initHl), m(right.prob.initHl))
      self.assertGreaterEqual(m(left.prob.initHl), m(right.prob.initHl))
    # print('bs', [m(f.prob.boost) for f in fulls])
    self.assertLess(m(fulls[-1].prob.boost), 1.0, 'mean boost<1 reached')

    # add another failure
    now += 1 * MILLISECONDS_PER_HOUR
    s = ebisu.updateRecall(fulls[-1], 0, now=now)

    # even though mean boost<1, the curr halflife should be equal to
    # init halflife (within machine precision) since successes can only boost
    # by scalar >=1.
    self.assertAlmostEqual(m(s.prob.initHl), s.pred.currentHalflifeHours)

  def test_full(self, verbose=True):
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    boostMean = 1.5
    boostBeta = 3.0
    boostPrior = (boostBeta * boostMean, boostBeta)

    now = ebisu._timeMs()
    init = ebisu.initModel(initHlPrior=initHlPrior, boostPrior=boostPrior, now=now)

    left = 0.3
    # simulate a variety of 4-quiz trajectories:
    # for fraction, result, lastNoisy in product([0.1], [0], [True]):
    for fraction, result, lastNoisy in product([0.1, 0.5, 1.5, 9.5], [0, 1], [False, True]):
      np.random.seed(seed=seed)  # for sanity when testing with Monte Carlo
      thisKey = (fraction, result, lastNoisy)
      results[thisKey] = dict()
      upd = deepcopy(init)
      elapsedHours = fraction * initHlMean
      thisNow = now + elapsedHours * MILLISECONDS_PER_HOUR
      upd = ebisu.updateRecall(upd, result, now=thisNow)

      for nextResult, nextElapsed, nextTotal in zip(
          [1, 1, 1 if not lastNoisy else (0.2)],
          [elapsedHours * 3, elapsedHours * 5, elapsedHours * 7],
          [1, 2, 2 if not lastNoisy else 1],
      ):
        thisNow += nextElapsed * MILLISECONDS_PER_HOUR
        upd = ebisu.updateRecall(upd, nextResult, total=nextTotal, q0=0.05, now=thisNow)

      ### Full Ebisu update (max-likelihood to enhanced Monte Carlo proposal)
      # This many samples is probably WAY TOO MANY for practical purposes but
      # here I want to ascertain that this approach is correct as you crank up
      # the number of samples. If we have confidence that this estimator behaves
      # correctly, we can in practice use 1_000 or 10_000 samples and accept a
      # less accurate model but remain confident that the *means* of this posterior
      # are accurate.
      tmp: tuple[ebisu.Model, dict] = ebisu.updateRecallHistory(
          upd, left=left, size=1_000_000, debug=True)
      full, fullDebug = tmp
      bEbisuSamplesStats, hEbisuSamplesStats = fullDebug['stats']

      ### Numerical integration via mpmath
      # This method stops being accurate when you have tens of quizzes but it's matches
      # the other methods well for ~4. It also can only compute posterior moments.
      bInt, hInt = numericalIntegration(upd, 6, left=left)

      MEAN_ERR = 0.01
      M2_ERR_BINOMIAL = 0.02
      M2_ERR_NOISY = 0.05
      M2_ERR = M2_ERR_NOISY if lastNoisy else M2_ERR_BINOMIAL
      MIN_KISH_EFFICIENCY = 0.5  # between 0 and 1, higher is better
      M2_FIT_ERR = 0.05
      # TRUE_POST_VAR_ERR = 0.01
      # FIT_VAR_ERR = 0.05

      ### Raw Monte Carlo simulation (without max likelihood enhanced proposal)
      # Because this method can be inaccurate and slow, try it with a small number
      # of samples and increase it quickly if we don't meet tolerances.
      for size in [1_000_000, 20_000_000]:
        mc = fullBinomialMonteCarlo(
            init.prob.initHlPrior,
            init.prob.boostPrior, [r.hoursElapsed for r in upd.quiz.results[-1]],
            upd.quiz.results[-1],
            size=size)
        mean_err = max(
            relativeError(bEbisuSamplesStats[0], mc['statsBoost'][0]),
            relativeError(hEbisuSamplesStats[0], mc['statsInitHl'][0]))
        m2_err = max(
            relativeError(bEbisuSamplesStats[2], mc['statsBoost'][2]),
            relativeError(hEbisuSamplesStats[2], mc['statsInitHl'][2]))
        results[thisKey][f'mc{size}/initHl/stats'] = mc['statsInitHl']
        results[thisKey][f'mc{size}/boost/stats'] = mc['statsBoost']

        if (mean_err < MEAN_ERR and m2_err < M2_ERR):
          break

      ### Finally, compare all three updates above.
      # Since numerical integration only gave us means, we can compare its means to
      # (a) full Ebisu update and (b) raw Monte Carlo.
      results[thisKey]['mc/initHl/stats'] = mc['statsInitHl']
      results[thisKey]['mc/boost/stats'] = mc['statsBoost']
      results[thisKey]['mc/size'] = mc['size']
      results[thisKey]['ebisuSamples/initHl/stats'] = hEbisuSamplesStats
      results[thisKey]['ebisuSamples/boost/stats'] = bEbisuSamplesStats
      results[thisKey]['ebisuSamples/size'] = fullDebug['size']
      results[thisKey]['ebisuSamples/initSize'] = fullDebug['initSize']
      results[thisKey]['ebisuSamples/kish'] = fullDebug['kish']
      results[thisKey]['int/initHl/stats'] = [float(x) for x in hInt]
      results[thisKey]['int/boost/stats'] = [float(x) for x in bInt]
      results[thisKey]['input'] = upd.to_json()
      results[thisKey]['output'] = full.to_json()

      # Ebisu posterior weighted samples vs numerical integration (nothing to do with Monte Carlo)
      self.assertLessEqual(
          max(
              relativeError(bEbisuSamplesStats[0], bInt[0]),
              relativeError(hEbisuSamplesStats[0], hInt[0])), MEAN_ERR,
          f'mean(ebisu posterior samples) = numerical integral mean, {fraction=}, {result=}, {lastNoisy=}'
      )
      self.assertLessEqual(
          max(
              relativeError(bEbisuSamplesStats[2], bInt[2]),
              relativeError(hEbisuSamplesStats[2], hInt[2])), M2_ERR,
          f'2nd moment(ebisu posterior samples) = numerical integral 2nd moment, {fraction=}, {result=}, {lastNoisy=}'
      )

      # Ebisu posterior Gamma fit vs Ebisu posterior samples (nothing to do with Monte Carlo)
      bEbisu = gammaToStats(*full.prob.boost)
      hEbisu = gammaToStats(*full.prob.initHl)
      self.assertLessEqual(
          max(
              relativeError(bEbisuSamplesStats[0], bEbisu[0]),
              relativeError(hEbisuSamplesStats[0], hEbisu[0])), MEAN_ERR,
          f'mean(ebisu posterior samples) = mean(ebisu Gamma fit), {fraction=}, {result=}, {lastNoisy=}'
      )
      # FIXME TODO why is sometimes 2nd moment so bad between posterior samples and fit?
      # self.assertLessEqual(
      #     max(
      #         relativeError(bEbisuSamplesStats[2], bEbisu[2]),
      #         relativeError(hEbisuSamplesStats[2], hEbisu[2])), M2_FIT_ERR,
      #     f'2nd moment(ebisu posterior samples) = 2nd moment(ebisu Gamma fit), {fraction=}, {result=}, {lastNoisy=}'
      # )

      # Finally get to Monte Carlo
      self.assertLess(
          mean_err, MEAN_ERR,
          f'Ebisu Gamma fit mean = Monte Carlo posterior mean, {fraction=}, {result=}, {lastNoisy=}'
      )
      self.assertLess(
          m2_err, M2_ERR,
          f'Ebisu Gamma fit 2nd moment = Monte Carlo posterior 2nd moment, {fraction=}, {result=}, {lastNoisy=}'
      )

      # check Kish efficiency
      self.assertGreater(fullDebug['kish'], MIN_KISH_EFFICIENCY,
                         f'Ebisu samples Kish efficiency, {fraction=}, {result=}, {lastNoisy=}')
    pickleName = f'finalres-{datetime.utcnow().isoformat().replace(":","")}.pickle'
    with open(pickleName, 'wb') as fid:
      results['date'] = datetime.utcnow().isoformat()
      results['seed'] = seed
      pickle.dump(results, fid)
    pickleToCsv(pickleName)
    return upd


def gammaToMean(a, b):
  return a / b


def gammaToStd(a, b):
  return np.sqrt(a) / b


def gammaToVar(a, b):
  return a / (b)**2


def gammaToMeanStd(a, b):
  return (gammaToMean(a, b), gammaToStd(a, b))


def gammaToMeanVar(a, b):
  return (gammaToMean(a, b), gammaToVar(a, b))


def gammaToStats(a: float, b: float):
  mean = gammaToMean(a, b)
  var = gammaToVar(a, b)
  k = a
  t = 1 / b
  m2 = t**2 * k * (k + 1)  # second non-central moment
  return (mean, var, m2, np.sqrt(m2))


def pickleToCsv(s: str):
  with open(s, 'rb') as fid:
    data = pickle.load(fid)

  output = io.StringIO()
  writer = None
  for key in data:
    if type(key) == tuple:
      row = data[key]
      diffs = dict()
      diffs['desc'] = "/".join([str(e) for e in key])
      diffs['kish'] = row['ebisuSamples/kish']
      diffs['ebisu size'] = row['ebisuSamples/size']
      diffs['mc size'] = row['mc/size']

      eh = relativeError(row['ebisuSamples/initHl/stats'], row['int/initHl/stats'])
      eb = relativeError(row['ebisuSamples/boost/stats'], row['int/boost/stats'])
      diffs['ebisuSamples vs int: initHl: mean'] = eh[0]
      diffs['ei: initHl: m2'] = eh[2]
      diffs['ei: boost: mean'] = eb[0]
      diffs['ei: boost: m2'] = eb[2]

      eh = relativeError(row['ebisuSamples/initHl/stats'], row['mc/initHl/stats'])
      eb = relativeError(row['ebisuSamples/boost/stats'], row['mc/boost/stats'])
      diffs['ebisuSamples vs mc: initHl: mean'] = eh[0]
      diffs['em: initHl: m2'] = eh[2]
      diffs['em: boost: mean'] = eb[0]
      diffs['em: boost: m2'] = eb[2]

      eh = relativeError(row['int/initHl/stats'], row['mc/initHl/stats'])
      eb = relativeError(row['int/boost/stats'], row['mc/boost/stats'])
      diffs['int vs mc: initHl: mean'] = eh[0]
      diffs['im: initHl: m2'] = eh[2]
      diffs['im: boost: mean'] = eb[0]
      diffs['im: boost: m2'] = eb[2]

      if not writer:
        writer = csv.DictWriter(output, fieldnames=[k for k in diffs])
        writer.writeheader()
      writer.writerow(diffs)
  with open(f'{s}.csv', 'w') as fid2:
    fid2.write(output.getvalue())


if __name__ == '__main__':
  import os
  # get just this file's module name: no `.py` and no path
  name = os.path.basename(__file__).replace(".py", "")
  unittest.TextTestRunner(failfast=True).run(unittest.TestLoader().loadTestsFromName(name))
