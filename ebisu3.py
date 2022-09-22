from scipy.linalg import lstsq  #type:ignore
from math import fsum
import numpy as np  # type:ignore
from scipy.stats import gamma as gammarv  # type: ignore
from scipy.special import kv, kve, gammaln, gamma, betaln, logsumexp  #type: ignore
from dataclasses import dataclass
from time import time_ns
from functools import cache
from typing import Union, Callable
from copy import deepcopy


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
    return res.successes >= res.total / 2
  else:
    raise Exception("unknown result type")


@dataclass
class Quiz:
  elapseds: list[list[float]]  # I think this has to be hours; possibly empty

  # same length as `elapseds`, and each sub-list has the same length
  results: list[list[Result]]

  # 0 < x <= 1 (reinforcement). Same length/sub-lengths as `elapseds`
  startStrengths: list[list[float]]


@dataclass
class Probability:
  # priors: fixed at model creation time
  initHlPrior: tuple[float, float]  # alpha and beta
  boostPrior: tuple[float, float]  # alpha and beta

  # posteriors: these change after quizzes
  initHl: tuple[float, float]  # alpha and beta
  boost: tuple[float, float]  # alpha and beta


@dataclass
class Predict:
  # just for developer ease, these can be stored in SQL, etc.
  # halflife is proportional to `logStrength - (startTime * CONSTANT) / halflife`
  # where `CONSTANT` converts `startTime` to same units as `halflife`.
  startTime: float  # unix epoch
  currentHalflife: float  # mean (so _currentHalflifePrior works). Same units as `elapseds`
  logStrength: float


@dataclass
class Model:
  quiz: Quiz
  prob: Probability
  pred: Predict


def initModel(initHlPrior: Union[tuple[float, float], None] = None,
              boostPrior: Union[tuple[float, float], None] = None,
              initHlMean: Union[float, None] = None,
              initHlStd: Union[float, None] = None,
              boostMean: Union[float, None] = None,
              boostStd: Union[float, None] = None) -> Model:
  """
  Create brand new Ebisu model

  Must provide either `initHlPrior` (Gamma random variable's α and β) or both
  `initHlMean` and `initHlStd` (Gamma random variable's mean and standard
  deviation σ), and similarly for `boostPrior` vs `boostMean` and `boostStd`.
  """
  if initHlPrior:
    hl0 = initHlPrior
  elif initHlMean is not None and initHlStd is not None:
    hl0 = _meanVarToGamma(initHlMean, initHlStd**2)
  else:
    raise ValueError('init halflife prior not specified')

  if boostPrior:
    b = boostPrior
  elif boostMean is not None and boostStd is not None:
    b = _meanVarToGamma(boostMean, boostStd**2)
  else:
    raise ValueError('boost prior not specified')

  assert _gammaToMean(*hl0) > 0, 'init halflife mean should be positive'
  assert _gammaToMean(*b) >= 1.0, 'boost mean should be >= 1'
  return Model(
      quiz=Quiz(elapseds=[], results=[], startStrengths=[]),
      prob=Probability(initHlPrior=hl0, boostPrior=b, initHl=hl0, boost=b),
      pred=Predict(startTime=0, currentHalflife=_gammaToMean(*hl0), logStrength=0.0))


def updateRecall(
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

  ret = deepcopy(model)  # clone
  ret.prob.initHl = (newAlpha, newBeta * totalBoost)
  _appendQuiz(ret, elapsed, resultObj, reinforcement)
  boostMean = _gammaToMean(*ret.prob.boost)
  if success(resultObj):
    boostedHl = mean * _clampLerp2(left * model.pred.currentHalflife,
                                   right * model.pred.currentHalflife, 1, max(1.0, boostMean), t)
  else:
    boostedHl = mean
  if reinforcement > 0:
    ret.pred.currentHalflife = boostedHl
    ret.pred.startTime = now or _timeMs()
    ret.pred.logStrength = np.log(reinforcement)
  else:
    ret.pred.currentHalflife = boostedHl

  return ret


def _increaseGammaVar(a: float, b: float, factor: float):
  mean = a / b
  var = a / b**2
  return _meanVarToGamma(mean, var * factor)


def updateRecallHistory(
    model: Model,
    left=0.3,
    right=1.0,
    size=10_000,
    debug=False,
) -> Union[Model, tuple[Model, dict]]:
  ret = deepcopy(model)
  if len(ret.quiz.elapseds[-1]) <= 2:
    # not enough quizzes to update boost
    return ret
  ab, bb = ret.prob.boostPrior
  ah, bh = ret.prob.initHlPrior
  minBoost, maxBoost = gammarv.ppf([.01, 0.9999], ab, scale=1.0 / bb)
  minHalflife, maxHalflife = gammarv.ppf([0.01, 0.9999], ah, scale=1.0 / bh)

  posterior2d = lambda b, h: _posterior(b, h, ret, left, right)
  bs, hs = np.random.rand(2, 400)
  bs = bs * (maxBoost - minBoost) + minBoost
  hs = hs * (maxHalflife - minHalflife) + minHalflife
  posteriors = np.vectorize(posterior2d, [float])(bs, hs)
  fit = _fitJointToTwoGammas(bs, hs, posteriors, weightPower=1.0)

  betterFit = _monteCarloImprove((fit['alphax'], fit['betax']), (fit['alphay'], fit['betay']),
                                 posterior2d,
                                 size=size,
                                 debug=debug)

  # update prior(s)
  ret.prob.boost = (betterFit['alphax'], betterFit['betax'])
  ret.prob.initHl = (betterFit['alphay'], betterFit['betay'])
  # update SQL-friendly scalars
  bmean, hmean = [_gammaToMean(*prior) for prior in [ret.prob.boost, ret.prob.initHl]]
  _, extra = _posterior(bmean, hmean, ret, left, right, extra=True)
  ret.pred.currentHalflife = extra['currentHalflife']
  if debug:
    return ret, dict(kish=betterFit['kish'], stds=betterFit['stds'])
  return ret


def _appendQuiz(model: Model, elapsed: float, result: Result, startStrength: float) -> None:
  # IMPURE

  if len(model.quiz.elapseds) == 0:
    model.quiz.elapseds = [[elapsed]]
  else:
    model.quiz.elapseds[-1].append(elapsed)

  if len(model.quiz.results) == 0:
    model.quiz.results = [[result]]
  else:
    model.quiz.results[-1].append(result)

  if len(model.quiz.startStrengths) == 0:
    model.quiz.startStrengths = [[startStrength]]
  else:
    model.quiz.startStrengths[-1].append(startStrength)


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
  # if X ~ Gamma(a, b), (c*X) ~ Gamma(a, b/c)
  a0, b0 = model.prob.initHl
  boosted = model.pred.currentHalflife / _gammaToMean(a0, b0)
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


def _clampLerp2(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
  if x <= x1:
    return y1
  if x >= x2:
    return y2
  mu = (x - x1) / (x2 - x1)
  return (y1 * (1 - mu) + y2 * mu)


def _clampLerp3(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
  mu = (x - x1) / (x2 - x1)
  y = (y1 * (1 - mu) + y2 * mu)
  return min(y2, max(y1, y))


def _posterior(b: float, h: float, ret: Model, left: float, right: float, extra=False):
  "log posterior up to a constant offset"
  ab, bb = ret.prob.boostPrior
  ah, bh = ret.prob.initHlPrior

  logb = np.log(b)
  logh = np.log(h)
  logprior = -bb * b - bh * h + (ab - 1) * logb + (ah - 1) * logh

  loglik = []
  currHalflife = h
  for (res, e) in zip(ret.quiz.results[-1], ret.quiz.elapseds[-1]):
    logPrecall = -e / currHalflife * LN2
    if isinstance(res, NoisyBinaryResult):
      # noisy binary/Bernoulli
      logPfail = np.log(-np.expm1(logPrecall))
      z = success(res)
      if z:
        # Stan has this nice function, log_mix, which is perfect for this...
        loglik.append(logsumexp([logPrecall + np.log(res.q1), logPfail + np.log(res.q0)]))
        currHalflife *= _clampLerp2(left * currHalflife, right * currHalflife, 1, max(1, b), e)
      else:
        loglik.append(logsumexp([logPrecall + np.log(1 - res.q1), logPfail + np.log(1 - res.q0)]))
    else:
      # binomial
      if success(res):
        loglik.append(logPrecall)
        currHalflife *= _clampLerp2(left * currHalflife, right * currHalflife, 1, max(1, b), e)
      else:
        loglik.append(np.log(-np.expm1(logPrecall)))
  logposterior = fsum(loglik + [logprior])
  if extra:
    return logposterior, dict(currentHalflife=currHalflife)
  return logposterior


def _fitJointToTwoGammas(x: Union[list[float], np.ndarray],
                         y: Union[list[float], np.ndarray],
                         logPosterior: Union[list[float], np.ndarray],
                         weightPower=0.0) -> dict:
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
  return dict(
      sol=sol,
      alphax=alphax,
      betax=betax,
      alphay=alphay,
      betay=betay,
      meanX=_gammaToMean(alphax, betax),
      meanY=_gammaToMean(alphay, betay))


def _monteCarloImprove(xprior: tuple[float, float],
                       yprior: tuple[float, float],
                       logposterior: Callable[[float, float], float],
                       size=10_000,
                       debug=False):
  x = gammarv.rvs(xprior[0], scale=1 / xprior[1], size=size)
  y = gammarv.rvs(yprior[0], scale=1 / yprior[1], size=size)
  f = np.vectorize(logposterior, otypes=[float])
  logp = f(x, y)
  logw = logp - (
      gammarv.logpdf(x, xprior[0], scale=1 / xprior[1]) +
      gammarv.logpdf(y, yprior[0], scale=1 / yprior[1]))
  w = np.exp(logw)
  alphax, betax = _weightedGammaEstimate(x, w)
  alphay, betay = _weightedGammaEstimate(y, w)
  return dict(
      x=[alphax, betax],
      y=[alphay, betay],
      alphax=alphax,
      betax=betax,
      alphay=alphay,
      betay=betay,
      kish=_kishLog(logw) if debug else -1,
      stds=[np.std(w * v) for v in [x, y]] if debug else [],
  )


def _weightedGammaEstimate(h, w):
  """
  See https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1067698046#Closed-form_estimators
  """
  wsum = fsum(w)
  whsum = fsum(w * h)
  that2 = np.sum(w * h * np.log(h)) / wsum - whsum / wsum * np.sum(w * np.log(h)) / wsum
  khat2 = whsum / wsum / that2
  fit = (khat2, 1 / that2)
  return fit


def _kishLog(logweights) -> float:
  "kish effective sample fraction, given log-weights"
  return np.exp(2 * logsumexp(logweights) - logsumexp(2 * logweights)) / logweights.size


def _predictRecall(model: Model, elapsedHours=None, logDomain=True) -> float:
  if elapsedHours is None:
    now = _timeMs()
    elapsedHours = (now - model.pred.startTime) / MILLISECONDS_PER_HOUR
  logPrecall = -elapsedHours / model.pred.currentHalflife * LN2 + model.pred.logStrength
  return logPrecall if logDomain else np.exp(logPrecall)


def _predictRecallBayesian(model: Model, elapsedHours=None, logDomain=True) -> float:
  if elapsedHours is None:
    now = _timeMs()
    elapsedHours = (now - model.pred.startTime) / MILLISECONDS_PER_HOUR

  (a, b), _totalBoost = _currentHalflifePrior(model)
  logPrecall = _intGammaPdfExp(
      a, b, elapsedHours * LN2,
      logDomain=True) + a * np.log(b) - gammaln(a) + model.pred.logStrength
  return logPrecall if logDomain else np.exp(logPrecall)


def resetHalflife(
    model: Model,
    initHlMean: float,
    initHlStd: float,
    startTime: Union[float, None] = None,
    strength: float = 1.0,
) -> Model:
  ret = deepcopy(model)
  ret.quiz.elapseds.append([])
  ret.quiz.results.append([])
  ret.quiz.startStrengths.append([])

  ret.prob.initHlPrior = _meanVarToGamma(initHlMean, initHlStd**2)
  ret.pred.currentHalflife = initHlMean
  ret.pred.startTime = startTime or _timeMs()
  ret.pred.logStrength = np.log(strength)
  return ret
