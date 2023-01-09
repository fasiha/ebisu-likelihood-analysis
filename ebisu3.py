from scipy.linalg import lstsq  #type:ignore
from scipy.optimize import minimize  #type:ignore
from math import fsum
import numpy as np  # type:ignore
from scipy.stats import gamma as gammarv  # type: ignore
from scipy.stats import uniform as uniformrv  # type: ignore
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
  hoursElapsed: float


@dataclass
class NoisyBinaryResult:
  result: float
  q1: float
  q0: float
  hoursElapsed: float


Result = Union[BinomialResult, NoisyBinaryResult]


def success(res: Result) -> bool:
  if isinstance(res, NoisyBinaryResult):
    return res.result > 0.5
  elif isinstance(res, BinomialResult):
    return res.successes * 2 > res.total
  else:
    raise Exception("unknown result type")


@dataclass
class Quiz:
  results: list[list[Result]]

  # 0 < x <= 1 (reinforcement). Same length/sub-lengths as `results`
  startStrengths: list[list[float]]

  # same length as `results`. Timestamp of the first item in each sub-array of
  # `results`
  startTimestampMs: list[float]


@dataclass
class Probability:
  # priors: fixed at model creation time
  initHlPrior: tuple[float, float]  # alpha and beta
  boostPrior: tuple[float, float]  # alpha and beta

  # posteriors: these change after quizzes
  initHl: tuple[float, float]  # alpha and beta
  boost: tuple[float, float]  # alpha and beta
  # we need both prior (belief before any quizzes are received) and posterior
  # (after quizzes) because when updating using all history
  # (`updateRecallHistory`), we need to know what to start from.


@dataclass
class Predict:
  # just for developer ease, these can be stored in SQL, etc.
  lastEncounterMs: float  # milliseconds since unix epoch
  currentHalflifeHours: float  # mean (so _currentHalflifePrior works). Same units as `elapseds`
  logStrength: float
  # recall probability is proportional to:
  # `logStrength - ((NOW_MS - lastEncounterMs) * HOURS_PER_MILLISECONDS) / currentHalflifeHours`
  # where NOW_MS is milliseconds since Unix epoch.


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
              boostStd: Union[float, None] = None,
              now: Union[float, None] = None) -> Model:
  """
  Create brand new Ebisu model

  Must provide either `initHlPrior` (Gamma random variable's α and β) or both
  `initHlMean` and `initHlStd` (Gamma random variable's mean and standard
  deviation, in hours), and similarly for `boostPrior` vs `boostMean` and
  `boostStd` (unitless).
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
  now = now or _timeMs()
  return Model(
      quiz=Quiz(results=[], startStrengths=[], startTimestampMs=[now]),
      prob=Probability(initHlPrior=hl0, boostPrior=b, initHl=hl0, boost=b),
      pred=Predict(lastEncounterMs=now, currentHalflifeHours=_gammaToMean(*hl0), logStrength=0.0))


def resetHalflife(
    model: Model,
    initHlMean: float,
    initHlStd: float,
    now: Union[float, None] = None,
    strength: float = 1.0,
) -> Model:
  now = now or _timeMs()

  ret = deepcopy(model)
  ret.quiz.results.append([])
  ret.quiz.startStrengths.append([])
  ret.quiz.startTimestampMs.append(now)

  ret.prob.initHlPrior = _meanVarToGamma(initHlMean, initHlStd**2)
  ret.pred.currentHalflifeHours = initHlMean
  ret.pred.lastEncounterMs = now
  ret.pred.logStrength = np.log(strength)
  return ret


def updateRecall(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    now: Union[None, float] = None,
    q0: Union[None, float] = None,
    reinforcement: float = 1.0,
    left=0.3,
    right=1.0,
) -> Model:
  now = now or _timeMs()
  (a, b), totalBoost = _currentHalflifePrior(model)
  t = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  resultObj: Union[NoisyBinaryResult, BinomialResult]

  if (0 < successes < 1):
    assert total == 1, "float `successes` implies total==1"
    q1 = max(successes, 1 - successes)  # between 0.5 and 1
    q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5
    z = successes >= 0.5
    updated = _gammaUpdateNoisy(a, b, t, q1, q0, z)
    resultObj = NoisyBinaryResult(result=successes, q1=q1, q0=q0, hoursElapsed=t)

  else:  # int, or float outside (0, 1) band
    assert successes == np.floor(successes), "float `successes` must be between 0 and 1"
    assert successes >= 0, "negative `successes` meaningless"
    assert total > 0, "positive binomial trials"
    k = int(successes)
    n = total
    updated = _gammaUpdateBinomial(a, b, t, k, n)
    resultObj = BinomialResult(successes=k, total=n, hoursElapsed=t)

  mean, newAlpha, newBeta = (updated.mean, updated.a, updated.b)

  ret = deepcopy(model)  # clone
  ret.prob.initHl = (newAlpha, newBeta * totalBoost)
  _appendQuizImpure(ret, resultObj, reinforcement)
  boostMean = _gammaToMean(*ret.prob.boost)
  if success(resultObj):
    boostedHl = mean * clampLerp(left * model.pred.currentHalflifeHours,
                                 right * model.pred.currentHalflifeHours, 1, max(1.0, boostMean), t)
  else:
    boostedHl = mean

  if reinforcement > 0:
    ret.pred.currentHalflifeHours = boostedHl
    ret.pred.lastEncounterMs = now
    ret.pred.logStrength = np.log(reinforcement)
  else:
    ret.pred.currentHalflifeHours = boostedHl

  return ret


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


def expandGamma(a: float, b: float, factor: float) -> tuple[float, float]:
  m, v = gammaToMeanVar(a, b)
  return _meanVarToGamma(m, v * factor)


def expand(thresh: float, minBoost: float, minHalflife: float, maxBoost: float, maxHalflife: float,
           n: int, lpVector):
  bvec = np.linspace(minBoost, maxBoost, int(np.sqrt(n)))
  hvec = np.linspace(minHalflife, maxHalflife, int(np.sqrt(n)))
  bs, hs = np.meshgrid(bvec, hvec)
  posteriors = lpVector(bs, hs)
  nz0, nz1 = np.nonzero(np.diff(np.sign(posteriors - np.max(posteriors) + abs(thresh)), axis=1))
  return bvec[np.max(nz1)] * 1.2, hvec[np.max(nz0)] * 1.2


def updateRecallHistory(
    model: Model,
    left=0.3,
    right=1.0,
    size=10_000,
    debug=False,
) -> Union[Model, tuple[Model, dict]]:
  ret = deepcopy(model)
  if len(ret.quiz.results[-1]) <= 2:
    # not enough quizzes to update boost
    return ret
  ab, bb = ret.prob.boostPrior
  ah, bh = ret.prob.initHlPrior
  minBoost, maxBoost = gammarv.ppf([.01, 0.9999], ab, scale=1.0 / bb)
  minHalflife, maxHalflife = gammarv.ppf([0.01, 0.9999], ah, scale=1.0 / bh)
  # print(f'for fit: b={[minBoost, maxBoost]}, hl={[minHalflife, maxHalflife]}')

  lpScalar = lambda b, h: _posterior(b, h, ret, left, right)
  lpVector = np.vectorize(lpScalar, otypes=[float])

  initSize = 600

  MIXTURE = True
  # MIXTURE = False
  unifWeight = 0.1
  weightPower = 2

  # print(f'{MIXTURE=}, {unifWeight=}')
  # print(f'for sharp: b={[minBoost, maxBoost]}, hl={[minHalflife, maxHalflife]}, {MIXTURE=}')

  maxBoost, maxHalflife = expand(10, minBoost / 3, minHalflife / 3, maxBoost * 3, maxHalflife * 3,
                                 initSize, lpVector)
  minBoost = 0
  minHalflife = 0

  if not MIXTURE:
    fit, bs, hs, posteriors = None, None, None, None
    betterFit = _monteCarloImprove(
        generateX=lambda size: uniformrv.rvs(size=size, loc=minBoost, scale=maxBoost - minBoost),
        generateY=lambda size: uniformrv.rvs(
            size=size, loc=minHalflife, scale=maxHalflife - minHalflife),
        logpdfX=lambda x: uniformrv.logpdf(x, loc=minBoost, scale=maxBoost - minBoost),
        logpdfY=lambda y: uniformrv.logpdf(y, loc=minHalflife, scale=maxHalflife - minHalflife),
        logposterior=lpScalar,
        size=size,
        debug=debug,
    )
  else:
    try:
      bs, hs = np.random.rand(2, initSize)
      bs = bs * (maxBoost - minBoost) + minBoost
      hs = hs * (maxHalflife - minHalflife) + minHalflife
      posteriors = lpVector(bs, hs)
      fit = _fitJointToTwoGammas(bs, hs, posteriors, weightPower=weightPower)
      # print(f'{weightPower=}')
    except AssertionError as e:
      if "positive gamma parameters" in e.args[0]:
        print('something bad happened but trying again:', e)
        initSize *= 2
        bs, hs = np.random.rand(2, initSize)
        bs = bs * (maxBoost - minBoost) + minBoost
        hs = hs * (maxHalflife - minHalflife) + minHalflife
        posteriors = lpVector(bs, hs)
        fit = _fitJointToTwoGammas(bs, hs, posteriors, weightPower=weightPower)
      else:
        raise e

    def mix(aWeight, genA, genB, logpdfA, logpdfB):

      def gen(size: int):
        numA = np.sum(np.random.rand(size) < aWeight)
        a = genA(size=numA)
        b = genB(size=size - numA)
        ret = np.hstack([a, b])
        np.random.shuffle(ret)  # why is this necessary??
        return ret

      def logpdf(x):
        lpA = logpdfA(x)
        lpB = logpdfB(x)
        return logsumexp(np.vstack([lpA, lpB]), axis=0, b=np.array([[aWeight, 1 - aWeight]]).T)

      return dict(gen=gen, logpdf=logpdf)

    xmix = mix(unifWeight,
               lambda size: uniformrv.rvs(size=size, loc=minBoost, scale=maxBoost - minBoost),
               lambda size: gammarv.rvs(fit['alphax'], scale=1 / fit['betax'], size=size),
               lambda x: uniformrv.logpdf(x, loc=minBoost, scale=maxBoost - minBoost),
               lambda x: gammarv.logpdf(x, fit['alphax'], scale=1 / fit['betax']))
    ymix = mix(
        unifWeight,
        lambda size: uniformrv.rvs(size=size, loc=minHalflife, scale=maxHalflife - minHalflife),
        lambda size: gammarv.rvs(fit['alphay'], scale=1 / fit['betay'], size=size),
        lambda x: uniformrv.logpdf(x, loc=minHalflife, scale=maxHalflife - minHalflife),
        lambda x: gammarv.logpdf(x, fit['alphay'], scale=1 / fit['betay']))

    betterFit = _monteCarloImprove(
        generateX=xmix['gen'],
        generateY=ymix['gen'],
        logpdfX=xmix['logpdf'],
        logpdfY=ymix['logpdf'],
        logposterior=lpScalar,
        size=size,
        debug=debug,
    )
  # update prior(s)
  ret.prob.boost = (betterFit['alphax'], betterFit['betax'])
  ret.prob.initHl = (betterFit['alphay'], betterFit['betay'])
  # update SQL-friendly scalars
  bmean, hmean = [_gammaToMean(*prior) for prior in [ret.prob.boost, ret.prob.initHl]]
  _, extra = _posterior(bmean, hmean, ret, left, right, extra=True)
  ret.pred.currentHalflifeHours = extra['currentHalflife']
  if debug:
    return ret, dict(
        origfit=fit,
        kish=betterFit['kish'],
        stats=betterFit['stats'],
        betterFit=betterFit,
        bs=bs,
        hs=hs,
        posteriors=posteriors,
        size=size,
        initSize = initSize,
    )
  return ret


def _appendQuizImpure(model: Model, result: Result, startStrength: float) -> None:
  # IMPURE

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
HOURS_PER_MILLISECONDS = 1 / 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


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
  boosted = model.pred.currentHalflifeHours / _gammaToMean(a0, b0)
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


def clampLerp(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
  mu = (x - x1) / (x2 - x1)
  y = (y1 * (1 - mu) + y2 * mu)
  return min(y2, max(y1, y))


@cache
def _logFactorial(n: int) -> float:
  return gammaln(n + 1)


def _logComb(n: int, k: int) -> float:
  return _logFactorial(n) - _logFactorial(k) - _logFactorial(n - k)


def _logBinomPmfLogp(n: int, k: int, logp: float) -> float:
  assert (n >= k >= 0)
  logcomb = _logComb(n, k)
  if n - k > 0:
    logq = np.log(-np.expm1(logp))
    return logcomb + k * logp + (n - k) * logq
  return logcomb + k * logp


def _noisyBinaryToLogPmfs(quiz: NoisyBinaryResult) -> tuple[float, float]:
  z = quiz.result > 0.5
  return _noisyHelper(z, quiz.q1, quiz.q0)


@cache
def _noisyHelper(z: bool, q1: float, q0: float) -> tuple[float, float]:
  return (_logBernPmf(z, q1), _logBernPmf(z, q0))


def _logBernPmf(z: Union[int, bool], p: float) -> float:
  return np.log(p) if z else np.log(1 - p)


def _posterior(b: float, h: float, ret: Model, left: float, right: float, extra=False):
  "log posterior up to a constant offset"
  ab, bb = ret.prob.boostPrior
  ah, bh = ret.prob.initHlPrior

  logb = np.log(b)
  logh = np.log(h)
  logprior = -bb * b - bh * h + (ab - 1) * logb + (ah - 1) * logh

  loglik = []
  currHalflife = h
  for res in ret.quiz.results[-1]:
    logPrecall = -res.hoursElapsed / currHalflife * LN2
    if isinstance(res, NoisyBinaryResult):
      # noisy binary/Bernoulli
      q1LogPmf, q0LogPmf = _noisyBinaryToLogPmfs(res)
      logPfail = np.log(-np.expm1(logPrecall))
      # Stan has this nice function, log_mix, which is perfect for this...
      loglik.append(np.log(np.exp(logPrecall + q1LogPmf) + np.exp(logPfail + q0LogPmf)))
      # logsumexp is 3x slower??
      # loglik.append(logsumexp([logPrecall + q1LogPmf, logPfail + q0LogPmf]))

      if (res.result > 0.5):
        currHalflife *= clampLerp(left * currHalflife, right * currHalflife, 1, max(1, b),
                                  res.hoursElapsed)
    else:
      # binomial
      loglik.append(_logBinomPmfLogp(res.total, res.successes, logPrecall))
      if success(res):
        currHalflife *= clampLerp(left * currHalflife, right * currHalflife, 1, max(1, b),
                                  res.hoursElapsed)
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


def _monteCarloImprove(generateX: Callable[[int], np.ndarray],
                       generateY: Callable[[int], np.ndarray],
                       logpdfX: Callable[[np.ndarray], np.ndarray],
                       logpdfY: Callable[[np.ndarray], np.ndarray],
                       logposterior: Callable[[float, float], float],
                       size=10_000,
                       debug=False):
  x = generateX(size)
  y = generateY(size)
  f = np.vectorize(logposterior, otypes=[float])
  logp = f(x, y)
  logw = logp - (logpdfX(x) + logpdfY(y))
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
      # stds=[np.std(w * v) for v in [x, y]] if debug else [],
      stats=[_weightedMeanVarLogw(logw, samples) for samples in [x, y]] if debug else [],
      closedFit=[(alphax, betax), (alphay, betay)] if debug else [],
      # maxLikFit=[_weightedGammaEstimateMaxLik(z, w) for z in [x, y]] if debug else [],
      logw=logw,
      logp=logp,
      xs=x,
      ys=y)


def _weightedMeanVarLogw(logw: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
  # [weightedMean] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  # [weightedVar] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Weighted_sample_variance
  logsumexpw = logsumexp(logw)
  mean = np.exp(logsumexp(logw, b=x) - logsumexpw)
  m2 = np.exp(logsumexp(logw, b=x**2) - logsumexpw)
  var = m2 - mean**2
  return (mean, var, m2, np.sqrt(m2))


def _weightedGammaEstimate(h, w):
  """
  See https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1067698046#Closed-form_estimators
  """
  wsum = fsum(w)
  whsum = fsum(w * h)
  t = fsum(w * h * np.log(h)) / wsum - whsum / wsum * fsum(w * np.log(h)) / wsum
  k = whsum / wsum / t
  return (k, 1 / t)
  # this is the bias-corrected form on Wikipedia
  # n = len(h)
  # t2 = n / (n - 1) * t
  # k2 = k - (3 * k - 2 / 3 * (k / (1 + k) - 0.8 * k / (1 + k)**2)) / n
  # fit2 = (k2, 1 / t2)
  # return fit2


def _weightedGammaEstimateMaxLik(x, w):
  "Maximum likelihood Gamma fit given weighted samples"
  est = _weightedGammaEstimate(x, w)
  wsum = fsum(w)
  meanLnX = fsum(np.log(x) * w) / wsum
  meanX = fsum(w * x) / wsum
  n = len(x)

  def opt(input):
    k = input[0]
    lik = (k - 1) * n * meanLnX - n * k - n * k * np.log(meanX / k) - n * gammaln(k)
    return -lik

  res = minimize(opt, [est[0]], bounds=[[.01, np.inf]])
  k = res.x[0]
  b = k / meanX  # b = 1/theta, theta = meanX / k
  return (k, b)


def _kishLog(logweights) -> float:
  "kish effective sample fraction, given log-weights"
  return np.exp(2 * logsumexp(logweights) - logsumexp(2 * logweights)) / logweights.size


def predictRecall(model: Model, now: Union[float, None] = None, logDomain=True) -> float:
  now = now or _timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  logPrecall = -elapsedHours / model.pred.currentHalflifeHours * LN2 + model.pred.logStrength
  return logPrecall if logDomain else np.exp(logPrecall)


def _predictRecallBayesian(model: Model, now: Union[float, None] = None, logDomain=True) -> float:
  now = now or _timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS

  (a, b), _totalBoost = _currentHalflifePrior(model)
  logPrecall = _intGammaPdfExp(
      a, b, elapsedHours * LN2,
      logDomain=True) + a * np.log(b) - gammaln(a) + model.pred.logStrength
  return logPrecall if logDomain else np.exp(logPrecall)


def gammaToMode(a, b):
  return (a - b) / b if a >= 1 else 0
