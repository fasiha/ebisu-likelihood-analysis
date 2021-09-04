import ebisu  #type:ignore
import typing
import numpy as np
from scipy.stats import gamma as gammarv, beta as betarv  #type:ignore
from scipy.special import logsumexp  #type:ignore

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


def weightedMeanVar(w: Farr, x: Farr):
  mean = weightedMean(w, x)
  var = np.sum(w * (x - mean)**2) / np.sum(w)
  return dict(mean=mean, var=var)


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


if __name__ == "__main__":
  import demo
  df = demo.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, _ = demo.traintest(df)
  train = train[::10]  # further subdivide, for computational purposes
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  dts_hours, results, _ = demo.dfToVariables(train[0]['df'])
  initHl = 0.25
  boostMode = 1.4
  boostBeta = 10.0 / 3
  initAB = 2.0
  model, res = post(results, dts_hours, initAB, initHl, boostMode, boostBeta, returnDetails=True)
  print('estimate of inital model:', ebisu.rescaleHalflife(model))

  mv = weightedMeanVar(res['weight'], res['boost'])
  postGamma = _meanVarToGamma(mv['mean'], mv['var'])
  postGammaMode = (postGamma[0] - 1) / postGamma[1]
  print('estimate of boost:', postGammaMode)
"""
boostBeta = 10:
estimate of inital model: (2.6759692857154893, 2.6759692857154893, 9.163320510417671)
estimate of boost: 1.477778729146228

boostBeta = 10/3:
estimate of inital model: (2.395692291697088, 2.395692291697088, 8.573529067668835)
estimate of boost: 1.5118281149632586
"""

#%%

# %%
