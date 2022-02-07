from scipy.linalg import lstsq  #type:ignore
from math import fsum
import numpy as np  # type:ignore
from scipy.optimize import shgo  #type: ignore
from scipy.stats import gamma as gammarv  # type: ignore
from scipy.special import kv, kve, gammaln, gamma, betaln, logsumexp  #type: ignore
from dataclasses import dataclass, replace
from time import time_ns
from functools import cache
from typing import Union


@dataclass
class BinomialResult:
  successes: int
  total: int


@dataclass
class NoisyBinaryResult:
  result: float
  q1: float
  q0: float


Result = Union[BinomialResult, NoisyBinaryResult]


def success(res: Result) -> bool:
  if isinstance(res, NoisyBinaryResult):
    return res.result > 0.5
  elif isinstance(res, BinomialResult):
    return res.total == res.successes
  else:
    raise Exception("unknown result type")


@dataclass
class Model:
  elapseds: list[list[float]]  # I think this has to be hours; possibly empty

  # same length as `elapseds`, and each sub-list has the same length
  results: list[list[Result]]

  # 0 < x <= 1 (reinforcement). Same length/sub-lengths as `elapseds`
  startStrengths: list[list[float]]

  # priors
  initHalflifePrior: tuple[float, float]  # alpha and beta
  boostPrior: tuple[float, float]  # alpha and beta

  # just for developer ease, these can be stored in SQL, etc.
  # halflife is proportional to `logStrength - (startTime * CONSTANT) / halflife`
  # where `CONSTANT` converts `startTime` to same units as `halflife`.
  startTime: float  # unix epoch
  currentHalflife: float  # mean (so _currentHalflifePrior works). Same units as `elapseds`
  logStrength: float


def _appendQuiz(model: Model, elapsed: float, result: Result, startStrength: float) -> None:
  # IMPURE

  if len(model.elapseds) == 0:
    model.elapseds = [[elapsed]]
  else:
    model.elapseds[-1].append(elapsed)

  if len(model.results) == 0:
    model.results = [[result]]
  else:
    model.results[-1].append(result)

  if len(model.startStrengths) == 0:
    model.startStrengths = [[startStrength]]
  else:
    model.startStrengths[-1].append(startStrength)


def _gammaToMean(alpha: float, beta: float) -> float:
  return alpha / beta


def _meanVarToGamma(mean, var) -> tuple[float, float]:
  a = mean**2 / var
  b = mean / var
  return a, b


def _logmeanlogVarToGamma(logmean, logvar) -> tuple[float, float]:
  loga = 2 * logmean - logvar
  logb = logmean - logvar
  return np.exp(loga), np.exp(logb)


def _timeMs() -> float:
  return time_ns() / 1_000_000


LN2 = np.log(2)
MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


def _intGammaPdf(a: float, b: float, logDomain: bool):
  # \int_0^∞ h^(a-1) \exp(-b h) dh$, via Wolfram Alpha, etc.
  if not logDomain:
    return b**(-a) * gamma(a)
  return -a * np.log(b) + gammaln(a)


def _intGammaPdfExp(a: float, b: float, c: float, logDomain: bool):
  # $s(a,b,c) = \int_0^∞ h^(a-1) \exp(-b h - c / h) dh$, via sympy
  if c == 0:
    return _intGammaPdf(a, b, logDomain=logDomain)

  z = 2 * np.sqrt(b * c)  # arg to kv
  if not logDomain:
    return 2 * (c / b)**(a * 0.5) * kv(a, z)
  # `kve = kv * exp(z)` -> `log(kve) = log(kv) + z` -> `log(kv) = log(kve) - z`
  return LN2 + np.log(c / b) * (a * 0.5) + np.log(kve(a, z)) - z


def _currentHalflifePrior(model: Model) -> tuple[tuple[float, float], float]:
  # if X ~ Gamma(a, b), (c*X) ~ Gamma(a, 1/c*b)
  a0, b0 = model.initHalflifePrior
  boosted = model.currentHalflife / _gammaToMean(a0, b0)
  return (a0, b0 / boosted), boosted


@dataclass
class GammaUpdate:
  a: float
  b: float
  mean: float


def _gammaUpdateNoisy(a: float, b: float, t: float, q1: float, q0: float, z: bool) -> GammaUpdate:
  """Core Ebisu v2-style Bayesian update on noisy binary quizzes

  Assuming a halflife $h ~ Gamma(a, b)$, a hidden quiz result $x ~
  Bernoulli(2^(t/h))$ (for $t$ time elapsed since review), and an
  observed *noisy* quiz report $z|x ~ Bernoulli(q0, q1)$, this function
  computes moments of the true posterior $h|z$, which is a nonstandard
  distribution, approximates it to a new $Gamma(newA, newB)$ and returns
  that approximate posterior.

  Note that all probabilistic parameters are assumed to be known: $a, b,
  q0, q1$, as are data $t, z$. Only $x$, the true quiz result is
  unknown, as well as of course the true halflife.

  See also `_gammaUpdateBinomial`.
  """
  qz = (q0, q1) if z else (1 - q0, 1 - q1)

  def moment(n):
    an = a + n
    return _intGammaPdfExp(an, b, t, logDomain=False) * (qz[1] - qz[0]) + qz[0] * gamma(an) / b**an

  m0 = moment(0)
  mean = moment(1) / m0
  m2 = moment(2) / m0
  var = m2 - mean**2
  newAlpha, newBeta = _meanVarToGamma(mean, var)
  return GammaUpdate(a=newAlpha, b=newBeta, mean=mean)


@cache
def _binomln(n, k):
  "Log of scipy.special.binom calculated entirely in the log domain"
  return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def _gammaUpdateBinomial(a: float, b: float, t: float, k: int, n: int) -> GammaUpdate:
  """Core Ebisu v2-style Bayesian update on binomial quizzes

  Assuming a halflife $h ~ Gamma(a, b)$ and a Binomial quiz at time $t$
  resulting in $k ~ Binomial(n, 2^(t/h))$ successes out of a total $n$
  trials, this function computes moments of the true posterior $h|k$,
  which is a nonstandard distribution, approximates it to a new
  $Gamma(newA, newB)$ and returns that approximate posterior.

  Note that all probabilistic parameters are assumed to be known ($a,
  b$), as are data parameters $t, n$, and experimental result $k$.

  See also `_gammaUpdateNoisy`.
  """

  def logmoment(nth) -> float:
    loglik = []
    scales = []
    for i in range(0, n - k + 1):
      loglik.append(_binomln(n - k, i) + _intGammaPdfExp(a + nth, b, t * (k + i), logDomain=True))
      scales.append((-1)**i)
    return logsumexp(loglik, b=scales)

  logm0 = logmoment(0)
  logmean = logmoment(1) - logm0
  logm2 = logmoment(2) - logm0
  logvar = logsumexp([logm2, 2 * logmean], b=[1, -1])
  newAlpha, newBeta = _logmeanlogVarToGamma(logmean, logvar)

  return GammaUpdate(a=newAlpha, b=newBeta, mean=np.exp(logmean))


def simpleUpdateRecall(
    model: Model,
    elapsed: float,
    successes: Union[float, int],
    total: int = 1,
    now: Union[None, float] = None,
    q0: Union[None, float] = None,
    reinforcement: float = 1.0,
    left=0.3,
    right=1.0,
) -> Model:
  (a, b), totalBoost = _currentHalflifePrior(model)
  t = elapsed
  resultObj: Union[NoisyBinaryResult, BinomialResult]

  if total == 1 and (0 < successes < 1):
    q1 = max(successes, 1 - successes)  # between 0.5 and 1
    q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5
    z = successes >= 0.5
    updated = _gammaUpdateNoisy(a, b, t, q1, q0, z)
    resultObj = NoisyBinaryResult(result=successes, q1=q1, q0=q0)

  else:
    assert successes == np.floor(successes), "float `successes` implies `total==1`"
    assert total, "non-zero binomial trials"
    k = int(successes)
    n = total
    updated = _gammaUpdateBinomial(a, b, t, k, n)
    resultObj = BinomialResult(successes=k, total=n)

  mean, newAlpha, newBeta = (updated.mean, updated.a, updated.b)

  ret = replace(model)  # clone
  ret.initHalflifePrior = (newAlpha, newBeta * totalBoost)
  _appendQuiz(ret, elapsed, resultObj, reinforcement)
  boostMean = _gammaToMean(ret.boostPrior[0], ret.boostPrior[1])
  if success(resultObj):
    boostedHl = mean * _clampLerp2(left * model.currentHalflife, right * model.currentHalflife,
                                   min(boostMean, 1.0), boostMean, t)
  else:
    boostedHl = mean
  if reinforcement > 0:
    ret.currentHalflife = boostedHl
    ret.startTime = now or _timeMs()
    ret.logStrength = np.log(reinforcement)
  else:
    ret.currentHalflife = boostedHl

  return ret


def _clampLerp2(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
  if x <= x1:
    return y1
  if x >= x2:
    return y2
  mu = (x - x1) / (x2 - x1)
  return (y1 * (1 - mu) + y2 * mu)


def _makeLogPrecalls_Halflives(b: float,
                               h: float,
                               results: list[Result],
                               elapseds: list[float],
                               startStrengths: list[float],
                               left=0.3,
                               right=1.0) -> list[tuple[float, float]]:
  maxb = b
  from itertools import accumulate
  from typing import TypedDict

  class Reduced(TypedDict):
    h: float  # halflife for NEXT quiz
    r: float
    t0: float
    logp: float

  def reduction(prev: Reduced, curr: tuple[Result, float, float]) -> Reduced:
    res, t, r = curr
    logp = -(t + prev["t0"]) / prev["h"] * LN2 + (np.log(prev["r"]) if prev["r"] > 0 else 0)

    if success(res) and r > 0:
      newh = prev["h"] * _clampLerp2(left * prev["h"], right * prev["h"], min(b, 1.0), maxb, t)
    else:
      newh = prev["h"]

    t0 = 0 if r > 0 else prev["t0"] + t
    return Reduced(h=newh, r=r, t0=t0, logp=logp)

  init: Reduced = dict(h=h, r=1.0, t0=0.0, logp=1)
  acc = accumulate(zip(results, elapseds, startStrengths), reduction, initial=init)
  next(acc)  # skip initial
  return [(a["logp"], a["h"]) for a in acc]


def _posterior(b: float, h: float, ret: Model, left: float, right: float):
  ab, bb = ret.boostPrior
  ah, bh = ret.initHalflifePrior

  logb = np.log(b)
  logh = np.log(h)
  logprior = -bb * b - bh * h + (ab - 1) * logb + (ah - 1) * logh
  logpHls = _makeLogPrecalls_Halflives(
      b, h, ret.results[-1], ret.elapseds[-1], ret.startStrengths[-1], left=left, right=right)

  loglik = []
  for (res, (logPrecall, _halflife)) in zip(ret.results[-1], logpHls):
    if isinstance(res, NoisyBinaryResult):
      # noisy binary/Bernoulli
      logPfail = np.log(-np.expm1(logPrecall))
      z = success(res)
      if z:
        # Stan has this nice function, log_mix, which is perfect for this...
        loglik.append(logsumexp([logPrecall + np.log(res.q1), logPfail + np.log(res.q0)]))
      else:
        loglik.append(logsumexp([logPrecall + np.log(1 - res.q1), logPfail + np.log(1 - res.q0)]))
    else:
      # binomial
      if success(res):
        loglik.append(logPrecall)
      else:
        loglik.append(np.log(-np.expm1(logPrecall)))
  logposterior = fsum(loglik + [logprior])
  return logposterior


def _fitJointToTwoGammas(x: Union[list[float], np.ndarray],
                         y: Union[list[float], np.ndarray],
                         logPosterior: Union[list[float], np.ndarray],
                         weightPower=1.0) -> dict:
  # four-dimensional weighted least squares
  x = np.array(x)
  y = np.array(y)
  logPosterior = np.array(logPosterior)
  assert x.size == y.size
  assert x.size == logPosterior.size

  A = np.vstack([np.log(x), -x, np.log(y), -y, np.ones_like(x)]).T
  weights = np.diag(np.exp(logPosterior - np.max(logPosterior))**weightPower)
  sol = lstsq(np.dot(weights, A), np.dot(weights, logPosterior))
  t = sol[0]
  alphax = t[0] + 1
  betax = t[1]
  alphay = t[2] + 1
  betay = t[3]

  assert all(x > 0 for x in [alphax, betax, alphay, betay]), 'positive gamma parameters'
  return dict(sol=sol, alphax=alphax, betax=betax, alphay=alphay, betay=betay)


def flatten(list_of_lists):
  "Flatten one level of nesting"
  from itertools import chain
  return chain.from_iterable(list_of_lists)


def fullUpdateRecall(
    model: Model,
    now: Union[None, float] = None,
    q0: Union[None, float] = None,
    reinforcement: float = 1.0,
    left=0.3,
    right=1.0,
) -> Model:
  ret = replace(model)
  # assume this is done outside # _appendQuiz(ret, elapsed, res, reinforcement)

  posterior2d = lambda b, h: _posterior(b, h, ret, left, right)

  MIN_BOOST = 1.0
  ab, bb = ret.boostPrior
  ah, bh = ret.initHalflifePrior
  maxBoost = gammarv.ppf(0.99, ab, scale=1.0 / bb)
  minHalflife, maxHalflife = gammarv.ppf([0.01, 0.99], ah, scale=1.0 / bh)
  opt = shgo(lambda x: -posterior2d(x[0], x[1]), [(MIN_BOOST, maxBoost),
                                                  (minHalflife, maxHalflife)])
  bestb, besth = opt.x
  bs = []
  hs = []
  posteriors = []
  for b in np.linspace(MIN_BOOST, maxBoost, 101):
    posteriors.append(posterior2d(b, besth))
    bs.append(b)
    hs.append(besth)
  for h in np.linspace(minHalflife, maxHalflife, 101):
    posteriors.append(posterior2d(bestb, h))
    bs.append(bestb)
    hs.append(h)
  fit = _fitJointToTwoGammas(bs, hs, posteriors, weightPower=0.0)

  # update prior(s)
  ret.boostPrior = (fit['alphax'], fit['betax'])
  ret.initHalflifePrior = (fit['alphay'], fit['betay'])
  # update SQL-friendly scalars
  bmean, hmean = [_gammaToMean(*prior) for prior in [ret.boostPrior, ret.initHalflifePrior]]
  futureHalflife = _makeLogPrecalls_Halflives(
      bmean,
      hmean,
      ret.results[-1],
      ret.elapseds[-1],
      ret.startStrengths[-1],
      left=left,
      right=right)[-1][1]
  if reinforcement > 0:
    ret.currentHalflife = futureHalflife
    ret.startTime = now or _timeMs()
    ret.logStrength = np.log(reinforcement)
  else:
    ret.currentHalflife = futureHalflife
  return ret


def _predictRecall(model: Model, elapsedHours=None, logDomain=True) -> float:
  if elapsedHours is None:
    now = _timeMs()
    elapsedHours = (now - model.startTime) / MILLISECONDS_PER_HOUR
  logPrecall = -elapsedHours / model.currentHalflife * LN2 + model.logStrength
  return logPrecall if logDomain else np.exp(logPrecall)


def _predictRecallBayesian(model: Model, elapsedHours=None, logDomain=True) -> float:
  if elapsedHours is None:
    now = _timeMs()
    elapsedHours = (now - model.startTime) / MILLISECONDS_PER_HOUR

  (a, b), _totalBoost = _currentHalflifePrior(model)
  logPrecall = _intGammaPdfExp(
      a, b, elapsedHours * LN2, logDomain=True) + a * np.log(b) - gammaln(a) + model.logStrength
  return logPrecall if logDomain else np.exp(logPrecall)


def reinitializeWithNewHalflife(
    model: Model,
    newMean: float,
    newStd: float,
    startTime: Union[float, None] = None,
    strength: float = 1.0,
) -> Model:
  ret = replace(model)
  ret.elapseds.append([])
  ret.results.append([])
  ret.startStrengths.append([])

  ret.initHalflifePrior = _meanVarToGamma(newMean, newStd**2)
  ret.currentHalflife = newMean
  if startTime:
    ret.startTime = startTime
  else:
    ret.startTime = _timeMs()
  ret.logStrength = np.log(strength)
  return ret
