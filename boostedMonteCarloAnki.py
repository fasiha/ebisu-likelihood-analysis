import typing
import numpy as np
from scipy.stats import gamma as gammarv, beta as betarv  #type:ignore

Farr = typing.Union[float, np.ndarray]


def clampLerp(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, x: float):
  # Asssuming x1 <= x <= x2, map x from [x0, x1] to [0, 1]
  mu: Farr = (x - x1) / (x2 - x1)  # will be >=0 and <=1
  # branchless: hoping it's faster (cache misses, etc.) than the equivalent:
  # `y1 if (x < x1) else y2 if (x > x2) else (y1 * (1 - mu) + y2 * mu)`
  if type(mu) == float:
    return (x < x1) * y1 + (x > x2) * y2 + (x1 <= x <= x2) * (y1 * (1 - mu) + y2 * mu)

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


def weightedMean(w: Farr, x: Farr) -> Farr:
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
         returnDetails=False):
  bools = [x > 1 for x in xs]
  N = 5_000_000
  p: np.ndarray = betarv.rvs(alphaBeta, alphaBeta, size=N)

  boostAlpha = boostBeta * boostMode + 1
  boost: np.ndarray = gammarv.rvs(boostAlpha, scale=1 / boostBeta, size=N)

  logp = np.log(p)
  previousHalflife: np.ndarray = np.ones_like(boost) * initHalflife
  logweight: Farr = 0.0
  for x, t in zip(bools, ts):
    boostedDelta = t / previousHalflife
    logweight += boostedDelta * logp if x else np.log(-np.expm1(boostedDelta * logp))

    thisBoost: np.ndarray = clampLerp(0.8 * previousHalflife, previousHalflife,
                                      np.minimum(boost, 1.0), boost, t)
    previousHalflife = previousHalflife * thisBoost
  weight = np.exp(logweight)

  mv = weightedMeanVar(weight, p)
  postBeta = _meanVarToBeta(mv['mean'], mv['var'])
  model = (postBeta[0], postBeta[1], initHl)
  if returnDetails:
    return model, dict(weight=weight, p=p, boost=boost)
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
  import ebisu  #type:ignore
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