import numpy as np
from scipy.special import kv, kve, gamma, betaln, logsumexp  #type: ignore
from dataclasses import dataclass, replace
from time import time_ns
from functools import cache
from typing import Union


@dataclass
class Model:
  elapseds: list[float]  # your choice of time units; possibly empty

  # First element: ints (binomial quizzes) or floats (fuzzy noisy-Bernoulli)
  # second element: total number of trials (1 implies binary, fuzzy or not-fuzzy)
  # same length as `elapseds`
  results: list[tuple[Union[float, int], int]]

  startStrengths: list[float]  # 0 < x <= 1 (reinforcement). Same length as `elpseds`

  # priors
  halflifePrior: tuple[float, float]  # alpha and beta
  boostPrior: tuple[float, float]  # alpha and beta

  # just for developer ease, these can be stored in SQL, etc.
  # halflife is proportional to `logStrength - (startTime * CONSTANT) / halflife`
  # where `CONSTANT` converts `startTime` to same units as `halflife`.
  startTime: float  # unix epoch
  halflife: float  # mean or mode? Same units as `elapseds`
  logStrength: float


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

  # $s(a,b,c) = \int_0^∞ h^(a-1) \exp(-b h - c / h) dh$, via sympy
  s = lambda a, b, c: 2 * (c / b)**(a * 0.5) * kv(-a, 2 * np.sqrt(b * c))

  a, b = model.halflifePrior
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
  ret.halflifePrior = (newAlpha, newBeta)
  ret.elapseds.append(elapsed)
  ret.results.append((result, 1))
  ret.startStrengths.append(reinforcement)
  boostMean = _gammaToMean(ret.boostPrior[0], ret.boostPrior[1])
  if reinforcement > 0:
    ret.halflife = mean * boostMean
    ret.startTime = now or _timeMs()
    ret.logStrength = np.log(reinforcement)
  else:
    ret.halflife = mean * boostMean

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
  LN2 = np.log(2)

  def logs(a: float, b: float, c: float):
    # s = lambda a, b, c: 2 * (c / b)**(a * 0.5) * kv(-a, 2 * np.sqrt(b * c))
    # `kve = kv * exp(z)` -> `log(kve) = log(kv) + z` -> `log(kv) = log(kve) - z`
    z = 2 * np.sqrt(b * c)  # arg to kv
    return LN2 + np.log(c / b) * (a * 0.5) + np.log(kve(a, z)) - z

  k = successes
  n = total
  a, b = model.halflifePrior
  t = elapsed

  def logmoment(nth) -> float:
    loglik = []
    scales = []
    for i in range(0, n - k + 1):
      loglik.append(_binomln(n, i) + logs(a + nth, b, t * (k + i)))
      scales.append((-1)**i)
    return logsumexp(loglik, b=scales)

  logm0 = logmoment(0)
  logmean = logmoment(1) - logm0
  logm2 = logmoment(2) - logm0
  logvar = logsumexp([logm2, 2 * logmean], b=[1, -1])
  newAlpha, newBeta = _logmeanlogVarToGamma(logmean, logvar)

  mean = np.exp(logmean)

  ret = replace(model)  # clone
  ret.halflifePrior = (newAlpha, newBeta)
  ret.elapseds.append(elapsed)
  ret.results.append((successes, total))
  ret.startStrengths.append(reinforcement)
  boostMean = _gammaToMean(ret.boostPrior[0], ret.boostPrior[1])
  if reinforcement > 0:
    ret.halflife = mean * boostMean
    ret.startTime = now or _timeMs()
    ret.logStrength = np.log(reinforcement)
  else:
    ret.halflife = mean * boostMean

  return ret


def simpleUpdate(model: Model,
                 elapsed: float,
                 successes: Union[float, int],
                 total: int = 1,
                 now: Union[None, float] = None,
                 q0: Union[None, float] = None,
                 reinforcement: float = 1.0) -> Model:
  if total == 1 and (0 < successes < 1):
    return _simpleUpdateNoisy(
        model, elapsed, successes, now=now, q0=q0, reinforcement=reinforcement)
  assert successes == np.floor(successes), "float `successes` implies `total==1`"
  return _simpleUpdateBinomial(
      model, elapsed, int(successes), total, now=now, reinforcement=reinforcement)