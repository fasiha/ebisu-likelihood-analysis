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
  currentHalflife: float  # mean or mode? Same units as `elapseds`
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


def _intGammaPdfExp(a: float, b: float, c: float, logDomain: bool):
  # $s(a,b,c) = \int_0^∞ h^(a-1) \exp(-b h - c / h) dh$, via sympy
  z = 2 * np.sqrt(b * c)  # arg to kv
  if not logDomain:
    return 2 * (c / b)**(a * 0.5) * kv(a, z)
  # s = lambda a, b, c: 2 * (c / b)**(a * 0.5) * kv(-a, 2 * np.sqrt(b * c))
  # `kve = kv * exp(z)` -> `log(kve) = log(kv) + z` -> `log(kv) = log(kve) - z`
  return LN2 + np.log(c / b) * (a * 0.5) + np.log(kve(a, z)) - z


def _currentHalflifePrior(model: Model) -> tuple[tuple[float, float], float]:
  # if X ~ Gamma(a, b), (c*X) ~ Gamma(a, c*b)
  a0, b0 = model.initHalflifePrior
  boosted = model.currentHalflife / _gammaToMean(a0, b0)
  return (a0, boosted * b0), boosted


def _simpleUpdateNoisy(model: Model,
                       elapsed: float,
                       result: float,
                       now: Union[None, float] = None,
                       q0: Union[None, float] = None,
                       reinforcement: float = 1.0) -> Model:
  q1 = max(result, 1 - result)  # between 0.5 and 1
  q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5
  z = result >= 0.5

  qz = (q0, q1) if z else (1 - q0, 1 - q1)

  s = lambda a, b, c: _intGammaPdfExp(a, b, c, logDomain=False)

  (a, b), totalBoost = _currentHalflifePrior(model)
  t = elapsed

  def moment(n):
    an = a + n
    return s(an, b, t) * qz[1] + qz[0] * gamma(an) / b**an - s(an, b, t) * qz[0]

  m0 = moment(0)
  mean = moment(1) / m0
  m2 = moment(2) / m0
  var = m2 - mean**2
  newAlpha, newBeta = _meanVarToGamma(mean, var)

  ret = replace(model)  # clone
  ret.initHalflifePrior = (newAlpha, newBeta / totalBoost)
  _appendQuiz(ret, elapsed, NoisyBinaryResult(result=result, q1=q1, q0=q0), reinforcement)
  boostMean = _gammaToMean(ret.boostPrior[0], ret.boostPrior[1])
  if reinforcement > 0:
    ret.currentHalflife = mean * boostMean
    ret.startTime = now or _timeMs()
    ret.logStrength = np.log(reinforcement)
  else:
    ret.currentHalflife = mean * boostMean

  return ret


@cache
def _binomln(n, k):
  "Log of scipy.special.binom calculated entirely in the log domain"
  return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def _simpleUpdateBinomial(model: Model,
                          elapsed: float,
                          successes: int,
                          total: int,
                          now: Union[None, float] = None,
                          reinforcement: float = 1.0) -> Model:
  k = successes
  n = total
  (a, b), totalBoost = _currentHalflifePrior(model)
  t = elapsed

  def logmoment(nth) -> float:
    loglik = []
    scales = []
    for i in range(0, n - k + 1):
      loglik.append(_binomln(n, i) + _intGammaPdfExp(a + nth, b, t * (k + i), logDomain=True))
      scales.append((-1)**i)
    return logsumexp(loglik, b=scales)

  logm0 = logmoment(0)
  logmean = logmoment(1) - logm0
  logm2 = logmoment(2) - logm0
  logvar = logsumexp([logm2, 2 * logmean], b=[1, -1])
  newAlpha, newBeta = _logmeanlogVarToGamma(logmean, logvar)

  mean = np.exp(logmean)

  ret = replace(model)  # clone
  # update prior(s)
  ret.initHalflifePrior = (newAlpha, newBeta / totalBoost)
  # ensure we add THIS quiz
  _appendQuiz(ret, elapsed, BinomialResult(successes=successes, total=total), reinforcement)
  boostMean = _gammaToMean(ret.boostPrior[0], ret.boostPrior[1])
  # update SQL-friendly scalars
  if reinforcement > 0:
    ret.currentHalflife = mean * boostMean
    ret.startTime = now or _timeMs()
    ret.logStrength = np.log(reinforcement)
  else:
    ret.currentHalflife = mean * boostMean

  return ret


def simpleUpdateRecall(
    model: Model,
    elapsed: float,
    successes: Union[float, int],
    total: int = 1,
    now: Union[None, float] = None,
    q0: Union[None, float] = None,
    reinforcement: float = 1.0,
) -> Model:
  if total == 1 and (0 < successes < 1):
    return _simpleUpdateNoisy(
        model, elapsed, successes, now=now, q0=q0, reinforcement=reinforcement)
  assert successes == np.floor(successes), "float `successes` implies `total==1`"
  return _simpleUpdateBinomial(
      model, elapsed, int(successes), total, now=now, reinforcement=reinforcement)


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
                         weightPower=1.0) -> dict:  # wls 4d
  x = np.array(x)
  y = np.array(y)
  logPosterior = np.array(logPosterior)

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


def fullUpdateRecall(
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
  if len(model.elapseds[-1]) == 0 or len(model.elapseds[-1]) < 2:
    return simpleUpdateRecall(
        model, elapsed, successes, total=total, now=now, q0=q0, reinforcement=reinforcement)
  ret = replace(model)
  # ensure we add THIS quiz
  res: Result
  if total == 1 and 0 < successes < 1:
    q1 = max(successes, 1 - successes)
    res = NoisyBinaryResult(result=successes, q1=q1, q0=1 - q1 if q0 is None else q0)
  else:
    assert successes == np.floor(successes), "float `successes` implies `total==1`"
    res = BinomialResult(successes=int(successes), total=total)
  _appendQuiz(ret, elapsed, res, reinforcement)

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
  fit = _fitJointToTwoGammas(bs, hs, posteriors, weightPower=3.0)

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
