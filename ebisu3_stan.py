import numpy as np
import pylab as plt  # type:ignore
import pandas as pd  # type:ignore
import json
import typing

from cmdstanpy import CmdStanModel  # type:ignore

import ebisu3 as ebisu
import utils

plt.ion()

ConvertAnkiMode = typing.Literal['approx', 'binary']


def convertAnkiResultToBinomial(result: int, mode: ConvertAnkiMode) -> typing.Tuple[int, int]:
  if mode == 'approx':
    # Try to approximate hard to easy with binomial: this is tricky and ad hoc
    if result == 1:  # fail
      return (0, 1)
    elif result == 2:  # hard
      return (1, 2)
    elif result == 3:  # good
      return (1, 1)
    elif result == 4:  # easy
      return (2, 2)
    else:  # easy
      raise Exception('unknown Anki result')
  elif mode == 'binary':
    # hard or better is pass
    return (int(result > 1), 1)


def binomialStan(
    card: utils.Card,
    mode: ConvertAnkiMode,
    # above: data
    # below: algorithm parameters
    hlMeanStd: typing.Tuple[float, float],
    boostMeanStd: typing.Tuple[float, float],
    left=0.3,
    right=1.0,
    # below: Stan/Monte Carlo parameters
    chains=2,
    iter_warmup=10_000,
    iter_sampling=50_000):

  # dump all `data` to JSON
  with open('ebisu3_binomial_data.json', 'w') as fid:
    successes, totals = zip(*[convertAnkiResultToBinomial(r, mode) for r in card.results])
    alphaHl, betaHl = ebisu._meanVarToGamma(hlMeanStd[0], hlMeanStd[1]**2)
    alphaBoost, betaBoost = ebisu._meanVarToGamma(boostMeanStd[0], boostMeanStd[1]**2)
    json.dump(
        dict(
            # quiz history
            T=len(card.results),
            successes=successes,
            totals=totals,
            t=card.dts_hours,
            # algorithm parameters
            left=left,
            right=right,
            alphaHl=alphaHl,
            betaHl=betaHl,
            alphaBoost=alphaBoost,
            betaBoost=betaBoost),
        fid)

  # initialize the Stan model: compiles the binary (or loads from cache), etc.
  model = CmdStanModel(stan_file="ebisu3_binomial.stan")

  # run the fit
  fit = model.sample(
      data="ebisu3_binomial_data.json",
      chains=chains,
      iter_warmup=iter_warmup,
      iter_sampling=iter_sampling,
      adapt_delta=0.98,
      show_progress=True)

  # convert MCMC samples to a dataframe so we can poke it easily
  fitdf = pd.DataFrame({
      k: v.ravel()
      for k, v in fit.stan_variables().items()
      if 1 == len([s for s in v.shape if s > 1])
  })

  return fit, fitdf


def gammaToMean(a, b):
  return a / b


def gammaToStd(a, b):
  return np.sqrt(a) / b


def gammaToMeanStd(a, b):
  return (gammaToMean(a, b), gammaToStd(a, b))


if __name__ == '__main__':
  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = utils.traintest(df)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  fracs = [0.8, 0.85, 0.9, 0.95]
  for card in [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]:
    hlMeanStd = (10., 10.)
    boostMeanStd = (1.5, 0.7)
    convertMode: ConvertAnkiMode = 'binary'

    fit, fitdf = binomialStan(
        card,
        mode=convertMode,
        hlMeanStd=hlMeanStd,
        boostMeanStd=boostMeanStd,
        iter_warmup=20_000,
        iter_sampling=100_000)

    # This is a silly way to fit posterior MCMC samples to two new Gammas but it's
    # how Ebisu3 does it so let's check its math
    hPostStan = ebisu._meanVarToGamma(np.mean(fitdf.hl0), np.var(fitdf.hl0))
    bPostStan = ebisu._meanVarToGamma(np.mean(fitdf.boost), np.var(fitdf.boost))

    model = ebisu.initModel(
        initHlMean=hlMeanStd[0],
        initHlStd=hlMeanStd[1],
        boostMean=boostMeanStd[0],
        boostStd=boostMeanStd[1])
    for ankiResult, elapsedTime in zip(card.results, card.dts_hours):
      s, t = convertAnkiResultToBinomial(ankiResult, convertMode)
      model = ebisu.updateRecall(model, elapsedTime, successes=s, total=t)

    model, debug = ebisu.updateRecallHistory(model, size=50_000, debug=True)

    print(f"# Card {card.key}")
    print('## Stan')
    print(f'hl0   mean={gammaToMean(*hPostStan):0.4g}, std={gammaToStd(*hPostStan):0.4g}')
    print(f'boost mean={gammaToMean(*bPostStan):0.4g}, std={gammaToStd(*bPostStan):0.4g}')
    print('## Ebisu v3')
    print(
        f'hl0   mean={gammaToMean(*model.prob.initHl):0.4g}, std={gammaToStd(*model.prob.initHl):0.4g}'
    )
    print(
        f'boost mean={gammaToMean(*model.prob.boost):0.4g}, std={gammaToStd(*model.prob.boost):0.4g}'
    )

    diagnosis = fit.diagnose()
    if 'no problems detected' not in diagnosis:
      print(diagnosis)

    pd.plotting.scatter_matrix(fitdf, hist_kwds=dict(bins=100))
    plt.suptitle(f"# Card {card.key}")
