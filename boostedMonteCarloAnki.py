import utils
import ebisu  #type:ignore
import typing
import numpy as np
from scipy.stats import gamma as gammarv, beta as betarv  #type:ignore
from scipy.special import logsumexp, betaln  #type:ignore

Farr = typing.Union[float, np.ndarray]


def clampLerp(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, x: float):
  # Asssuming x1 <= x <= x2, map x from [x0, x1] to [0, 1]
  mu: Farr = (x - x1) / (x2 - x1)  # will be >=0 and <=1
  ret = np.empty_like(y2)
  idx = x < x1
  ret[idx] = y1[idx]
  idx = x > x2
  ret[idx] = y2[idx]
  idx = np.logical_and(x1 <= x, x <= x2)
  ret[idx] = (y1 * (1 - mu) + y2 * mu)[idx]
  return ret


def _meanVarToGamma(mean, var) -> tuple[float, float]:
  a = mean**2 / var
  b = mean / var
  return a, b


def _meanVarToBeta(mean, var) -> tuple[float, float]:
  """Fit a Beta distribution to a mean and variance."""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


def weightedMean(w: Farr, x: Farr) -> float:
  return np.sum(w * x) / np.sum(w)


def weightedMeanLogw(logw: np.ndarray, x: np.ndarray) -> np.ndarray:
  return np.exp(logsumexp(logw, b=x) - logsumexp(logw))


def weightedMeanVarLogw(logw: np.ndarray, x: np.ndarray):
  logsumw = logsumexp(logw)
  logmean = logsumexp(logw, b=x) - logsumw
  mean = np.exp(logmean)
  logvar = logsumexp(logw, b=(x - mean)**2) - logsumw
  return (mean, np.exp(logvar))


def gammafit(logw: np.ndarray, x: np.ndarray):
  mean, var = weightedMeanVarLogw(logw, x)
  a, b = _meanVarToGamma(mean, var)
  mode = a / b
  return dict(a=a, b=b, mean=mean, var=var, mode=mode)


def betafit(logw: np.ndarray, x: np.ndarray):
  mean, var = weightedMeanVarLogw(logw, x)
  a, b = _meanVarToBeta(mean, var)
  return dict(a=a, b=b, mean=mean, var=var)


def weightedMeanVar(w: Farr, x: Farr):
  mean = weightedMean(w, x)
  var = np.sum(w * (x - mean)**2) / np.sum(w)
  return dict(mean=mean, var=var)


def binomln(n, k):
  "Log of scipy.special.binom calculated entirely in the log domain"
  assert np.logical_and(0 <= k, k <= n).all(), "0 <= k <= n"
  return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def ankiFitEasyHard(xs: list[int],
                    ts: list[float],
                    hlMode: float,
                    hlBeta: float,
                    boostMode: float,
                    boostBeta: float,
                    size=100_000):
  hlAlpha = hlBeta * hlMode + 1.0
  hl0s: np.ndarray = gammarv.rvs(hlAlpha, scale=1.0 / hlBeta, size=size)

  boostAlpha = boostBeta * boostMode + 1.0
  boosts: np.ndarray = gammarv.rvs(boostAlpha, scale=1.0 / boostBeta, size=size)

  hardFactors: np.ndarray = betarv.rvs(3.0, 2.0, size=size)  # 0 to 1
  easyFactors: np.ndarray = betarv.rvs(2.0, 3.0, size=size)  # 0 to 1
  easyHardTotal = 50

  logweights = np.zeros(size)
  hls = hl0s.copy()
  for i, (x, t) in enumerate(zip(xs, ts)):
    logps = -t / hls
    if x == 1:  # failure
      logweights += np.log(-np.expm1(logps))  # log(1-p) = log(1-exp(logp)) = log(-expm1(logp))
    elif x == 3 or True:  # pass (normal)
      logweights += logps  # log(p)
    elif x == 2 or x == 4:  # hard or easy
      n = easyHardTotal
      if x == 2:
        k = np.round(np.exp(logps) * easyHardTotal * hardFactors)
      else:
        ps = np.exp(logps)
        k = np.round(easyHardTotal * (ps + (1 - ps) * easyFactors))
      # print(np.mean(k))
      logweights += binomln(n, k) + k * logps + (n - k) * np.log(-np.expm1(logps))
      # binomial pdf: approximate the "p observed" as a scaled value
    else:
      raise Exception(f'unknown result {x}')

    hls *= clampLerp(0.8 * hls, hls, np.minimum(boosts, 1.0), boosts, t)
    # print(f' mean hl{i+1}={weightedMeanLogw(logweights, hls)}')
  kishEffectiveSampleSize = np.exp(2 * logsumexp(logweights) - logsumexp(2 * logweights))
  # https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples

  print(
      f'mean boost ={weightedMeanLogw(logweights, boosts):0.4g}, neff={kishEffectiveSampleSize:0.2g}, {gammafit(logweights, boosts)}'
  )
  print(
      f'mean inithl={weightedMeanLogw(logweights, hl0s):0.4g}, neff={kishEffectiveSampleSize:0.2g}, {gammafit(logweights, hl0s)}'
  )
  print(
      f'mean finlhl={weightedMeanLogw(logweights, hls):0.4g}, neff={kishEffectiveSampleSize:0.2g}, {gammafit(logweights, hls)}'
  )
  # print(
  #     f'mean hardsc={weightedMeanLogw(logweights, hardFactors):0.4g}, {betafit(logweights, hardFactors)}'
  # )

  return dict(
      logweights=logweights,
      boosts=boosts,
      hl0s=hl0s,
      hardFactors=hardFactors,
      kishEffectiveSampleSize=kishEffectiveSampleSize)


def post(xs: list[int],
         ts: list[float],
         alphaBeta: float,
         initHalflife: float,
         boostMode: float,
         boostBeta: float,
         nsamples=5_000_000,
         returnDetails=False):
  bools = [x > 1 for x in xs]
  p: np.ndarray = betarv.rvs(alphaBeta, alphaBeta, size=nsamples)

  boostAlpha = boostBeta * boostMode + 1
  boost: np.ndarray = gammarv.rvs(boostAlpha, scale=1 / boostBeta, size=nsamples)

  logp = np.log(p)
  prevTimeHorizon: np.ndarray = np.ones_like(boost) * initHalflife
  logweight = np.zeros_like(boost)
  precalls: list[float] = []
  logprecallsEbisu: list[float] = []
  for x, t in zip(bools, ts):
    boostedDelta = t / prevTimeHorizon

    # not cheating here but need to move this to likelihood to ensure data isolation
    weight = np.exp(logweight)
    # mv = weightedMeanVar(weight, p)
    # postBeta = _meanVarToBeta(mv['mean'], mv['var'])
    # meanHorizon = weightedMean(weight, prevTimeHorizon)
    # model = (postBeta[0], postBeta[1], meanHorizon)
    # logprecallsEbisu.append(ebisu.predictRecall(model, t))
    # Above: this suffers from Jensen ineqality: collapsing horizon's richness to a mean
    # This uses Monte Carlo to exactly represent the precall.
    # They'll agree only for the first quiz.
    precalls.append(weightedMean(weight, p**boostedDelta))

    logweight += boostedDelta * logp if x else np.log(-np.expm1(boostedDelta * logp))

    thisBoost: np.ndarray = clampLerp(0.8 * prevTimeHorizon, prevTimeHorizon,
                                      np.minimum(boost, 1.0), boost, t)
    prevTimeHorizon = prevTimeHorizon * thisBoost
  weight = np.exp(logweight)

  mv = weightedMeanVar(weight, p)
  postBeta = _meanVarToBeta(mv['mean'], mv['var'])
  meanHorizon = weightedMean(weight, prevTimeHorizon)
  model = (postBeta[0], postBeta[1], meanHorizon)
  if returnDetails:
    return model, dict(
        weight=weight,
        p=p,
        boost=boost,
        logprecalls=np.log(precalls),
        logprecallsEbisu=logprecallsEbisu)
  return model


def overlap(thisdf, thatdf):
  hits = np.logical_and(
      min(thatdf.timestamp) <= thisdf.timestamp, thisdf.timestamp <= max(thatdf.timestamp))
  # `hits` is as long as `thisdf`
  overlapFraction = sum(hits) / len(hits)

  for t in thisdf.timestamp:
    sum(thatdf.timestamp < t)

  return overlapFraction


def overlap2(thiscard: utils.Card, thatcard: utils.Card):
  ts = np.array(thiscard.absts_hours)
  hits = np.logical_and(min(thatcard.absts_hours) <= ts, ts <= max(thatcard.absts_hours))
  # `hits` is as long as `thisdf`
  overlapFraction = sum(hits) / len(hits)

  dts_hours_that: list[typing.Union[None, float]] = []
  thatts = np.array(thatcard.absts_hours)
  for t in thiscard.absts_hours:
    num = sum(thatts < t)
    dts_hours_that.append(None if num == 0 else (t - thatcard.absts_hours[num - 1]))

  return overlapFraction, dts_hours_that


if __name__ == "__main__":
  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, _ = utils.traintest(df)
  # train = train[::10]  # further subdivide, for computational purposes
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  initHl = 0.25
  boostMode = 1.4
  boostBeta = 10.0 / 3
  initAB = 2.0
  if True:
    for t in train[0:3]:
      res = ankiFitEasyHard(
          t.results,
          t.dts_hours,
          hlMode=0.25,
          hlBeta=10.0,
          boostMode=1.5,
          boostBeta=10.0,
          size=1_000_000)
      print('---')
  if False:
    model, res = post(
        train[0].results,
        train[0].dts_hours,
        initAB,
        initHl,
        boostMode,
        boostBeta,
        returnDetails=True)
    print('estimate of initial model:', ebisu.rescaleHalflife((model[0], model[1], initHl)))
    print('estimate of final model:', ebisu.rescaleHalflife(model))

    mv = weightedMeanVar(res['weight'], res['boost'])
    postGamma = _meanVarToGamma(mv['mean'], mv['var'])
    postGammaMode = (postGamma[0] - 1) / postGamma[1]
    print('estimate of boost:', postGammaMode)

  thiscard = train[0]
  thatcard = None
  for t in train[1:]:
    if overlap(thiscard.df, t.df) > 0.5:
      thatcard = t
      break
  if thatcard:
    print("ok!")

  # ts = [t for t in train if overlap(train[0].df, t.df) > 0.8 and overlap(t.df, train[0].df) > 0.5]
"""
boostBeta = 10:
estimate of inital model: (2.6759692857154893, 2.6759692857154893, 9.163320510417671)
estimate of boost: 1.477778729146228

boostBeta = 10/3:
estimate of inital model: (2.395692291697088, 2.395692291697088, 8.573529067668835)
estimate of boost: 1.5118281149632586
"""
