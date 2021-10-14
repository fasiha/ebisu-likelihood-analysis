import utils
import ebisu  #type:ignore
import typing
import numpy as np
from scipy.stats import gamma as gammarv, beta as betarv  #type:ignore
from scipy.special import logsumexp, betaln  #type:ignore
import pandas as pd  #type:ignore

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


def ankiFitEasyHardQuad(xs: list[int], ts: list[float], priors, clamp):
  from math import prod

  def clampLerp2(x1, x2, y1, y2, x):
    if x <= x1:
      return y1
    if x >= x2:
      return y2
    mu = (x - x1) / (x2 - x1)
    return (y1 * (1 - mu) + y2 * mu)

  def integrand(b, h, expectation):
    if priors == 'gamma':
      ab = 10 * 1.4 + 1
      bb = 10.0
      ah = 10 * .25 + 1
      bh = 10.0

      prior = b**(ab - 1) * np.exp(-bb * b - bh * h) * h**(ah - 1)
    elif priors == 'exp':
      bb = 1.0
      bh = 0.5
      prior = np.exp(-bb * b - bh * h)
    else:
      raise Exception('unknown priors')

    lik = np.ones_like(prior)
    hs = [h]
    for t in ts[1:]:
      old = hs[-1]
      if clamp:
        hs.append(old * clampLerp2(0.8 * old, old, np.minimum(b, 1.0), b, t))
      else:
        hs.append(old * b)
    lik *= np.exp(sum([-t / h for x, t, h in zip(xs, ts, hs) if x > 1]))
    lik *= prod([1 - np.exp(-t / h) for x, t, h in zip(xs, ts, hs) if x <= 1])
    if expectation == '':
      extra = 1.0
    elif expectation == 'b':
      extra = b
    elif expectation == 'h':
      extra = h
    else:
      raise Exception('unknown expectation')
    return extra * lik * prior

  from scipy.integrate import dblquad, nquad  #type:ignore

  den = dblquad(lambda a, b: integrand(a, b, ''), 0, np.inf, 0, np.inf, epsabs=1e-10, epsrel=1e-10)
  eb = dblquad(lambda a, b: integrand(a, b, 'b'), 0, np.inf, 0, np.inf, epsabs=1e-10, epsrel=1e-10)
  eh = dblquad(lambda a, b: integrand(a, b, 'h'), 0, np.inf, 0, np.inf, epsabs=1e-10, epsrel=1e-10)
  return dict(eb=eb[0] / den[0], eh=eh[0] / den[0], quad=dict(den=den, eb=eb, eh=eh))


def ankiFitEasyHardMpmath(xs: list[int], ts: list[float], priors, clamp, dps=15):
  from math import prod
  import mpmath as mp  # type:ignore
  mp.mp.dps = dps

  def clampLerp2(x1, x2, y1, y2, x):
    if x <= x1:
      return y1
    if x >= x2:
      return y2
    mu = (x - x1) / (x2 - x1)
    return (y1 * (1 - mu) + y2 * mu)

  def integrand(b, h, expectation):
    if priors == 'gamma':
      ab = 2 * 1.4 + 1
      bb = 2.0
      ah = 2 * .25 + 1
      bh = 2.0

      mkconstant = lambda a, b: b**a / mp.gamma(a)
      prior = mkconstant(ab, bb) * mkconstant(ah, bh) * (
          b**(ab - 1) * mp.exp(-bb * b - bh * h) * h**(ah - 1))
    elif priors == 'exp':
      bb = 1.0
      bh = 0.5
      prior = bb * bh * mp.exp(-bb * b - bh * h)
    else:
      raise Exception('unknown priors')
    lik = 1
    hs = [h]
    for t in ts[1:]:
      old = hs[-1]
      if clamp:
        hs.append(old * clampLerp2(0.8 * old, old, min(b, 1.0), b, t))
      else:
        hs.append(old * b)
    lik *= mp.exp(sum([-t / h for x, t, h in zip(xs, ts, hs) if x > 1]))
    lik *= prod([1 - mp.exp(-t / h) for x, t, h in zip(xs, ts, hs) if x <= 1])
    if expectation == '':
      extra = 1.0
    elif expectation == 'b':
      extra = b
    elif expectation == 'h':
      extra = h
    else:
      raise Exception('unknown expectation')
    return extra * lik * prior

  maxdegree = 8
  print('maxdegree=', maxdegree)
  den = mp.quad(
      lambda a, b: integrand(a, b, ''), [0, mp.inf], [0, mp.inf], error=True, maxdegree=maxdegree)
  eb = mp.quad(
      lambda a, b: integrand(a, b, 'b'), [0, mp.inf], [0, mp.inf], error=True, maxdegree=maxdegree)
  eh = mp.quad(
      lambda a, b: integrand(a, b, 'h'), [0, mp.inf], [0, mp.inf], error=True, maxdegree=maxdegree)
  return dict(eb=eb[0] / den[0], eh=eh[0] / den[0], quad=dict(den=den, eb=eb, eh=eh))


def ankiFitEasyHardMAP(xs: list[int], ts: list[float], priors, clamp):
  from math import fsum

  def clampLerp2(x1, x2, y1, y2, x):
    if x <= x1:
      return y1
    if x >= x2:
      return y2
    mu = (x - x1) / (x2 - x1)
    return (y1 * (1 - mu) + y2 * mu)

  def posterior(b, h):
    logb = np.log(b)
    logh = np.log(h)
    if priors == 'gamma':
      ab = 10 * 1.4 + 1
      bb = 10.0
      ah = 10 * .25 + 1
      bh = 10.0

      logprior = -bb * b - bh * h + (ab - 1) * logb + (ah - 1) * logh
      # prior = b**(ab - 1) * np.exp(-bb * b - bh * h) * h**(ah - 1)
    elif priors == 'exp':
      bb = 1.0
      bh = 0.5
      logprior = -bb * b - bh * h
      # prior = np.exp(-bb * b - bh * h)
    else:
      raise Exception('unknown priors')
    hs = [h]
    for t in ts[1:]:
      old = hs[-1]
      if clamp:
        hs.append(old * clampLerp2(0.8 * old, old, min(b, 1.0), b, t))
      else:
        hs.append(old * b)
    # lik = 1
    # lik *= np.exp(sum([-t / h for x, t, h in zip(xs, ts, hs) if x > 1]))
    # lik *= prod([1 - np.exp(-t / h) for x, t, h in zip(xs, ts, hs) if x <= 1])
    loglik = [-t / h if x > 1 else np.log(-np.expm1(-t / h)) for x, t, h in zip(xs, ts, hs)]

    return fsum(loglik + [logprior])

  bvec = np.linspace(0.5, 5, 101)
  hvec = np.linspace(0.5, 5, 101)
  f = np.vectorize(posterior)
  bmat, hmat = np.meshgrid(bvec, hvec)
  z = f(bmat, hmat)

  import pylab as plt  #types:ignore
  rescalec = lambda im, top: im.set_clim(im.get_clim()[1] - np.array([top, 0]))

  def imshow(x, y, z, ax=plt):

    def extents(f):
      delta = f[1] - f[0]
      return [f[0] - delta / 2, f[-1] + delta / 2]

    return ax.imshow(
        z, aspect='auto', interpolation='none', extent=extents(x) + extents(y), origin='lower')

  fig, ax = plt.subplots()
  im = imshow(bvec, hvec, z, ax=ax)
  ax.set_xlabel('boost')
  ax.set_ylabel('init halflife')
  fig.colorbar(im)
  rescalec(im, 20)

  return dict(z=z, bvec=bvec, hvec=hvec, bmat=bmat, hmat=hmat, fig=fig, ax=ax, im=im)


def ankiFitEasyHardStan(xs: list[int], ts: list[float]):
  import json
  from cmdstanpy import CmdStanModel  #type:ignore

  data = dict(T=len(xs), x=[int(x) for x in xs], t=ts)
  with open('ankiFitEasyHard.json', 'w') as fid:
    json.dump(data, fid)

  model = CmdStanModel(stan_file="ankiFitEasyHard.stan")
  fit = model.sample(
      data="ankiFitEasyHard.json",
      chains=2,
      iter_warmup=30_000,
      iter_sampling=60_000,
      adapt_delta=0.98,
      show_progress=True)
  return fit


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


def summary(t: utils.Card):
  print("\n".join([f'{x}@{t:0.2f} {"🔥" if x<2 else ""}' for x, t in zip(t.results, t.dts_hours)]))


def testquad():
  print('# MPMATH')
  expmp = ankiFitEasyHardMpmath([3, 3, 3], [0.9, 3.3, 14.5], 'exp', True)
  print('## exp')
  print(expmp)

  gammp = ankiFitEasyHardMpmath([3, 3, 3], [0.9, 3.3, 14.5], 'gamma', True)
  print('## gam')
  print(gammp)

  print('# SCIPY')
  exp = ankiFitEasyHardQuad([3, 3, 3], [0.9, 3.3, 14.5], 'exp', True)
  print('## exp')
  print(exp)

  gam = ankiFitEasyHardQuad([3, 3, 3], [0.9, 3.3, 14.5], 'gamma', True)
  print('## gamma')
  print(gam)


if __name__ == "__main__":
  import pylab as plt  #type:ignore
  plt.ion()

  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = utils.traintest(df)
  # train = train[::10]  # further subdivide, for computational purposes
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  rescalec = lambda im, top: im.set_clim(im.get_clim()[1] - np.array([top, 0]))

  if True:
    fracs = [0.7, 0.8, 0.9]
    subtrain = [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]
    reses = []
    for t in subtrain:
      res = ankiFitEasyHardMAP(t.results, t.dts_hours, 'gamma', True)
      reses.append(res)

  if False:
    fracs = [0.7, 0.8, 0.9]
    subtrain = [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]
    reses = []
    for t in subtrain:
      res = ankiFitEasyHardMpmath(t.results, t.dts_hours, 'gamma', True, dps=15)
      print(res)
      reses.append(res)

    # testquad()
  if False:
    g = df[df.cid == 1300038031016].copy()
    dts_hours, results, ts_hours = utils.dfToVariables(g)
    print("\n".join([f'{x}@{t:0.2f} {"🔥" if x<2 else ""}' for x, t in zip(results, dts_hours)]))
    raise Exception('')

  if False:
    fits = []
    fracs = [0.7, 0.8, 0.9]
    fracs = [0.7]
    subtrain = [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]
    for t in subtrain:
      fit = ankiFitEasyHardStan(t.results, t.dts_hours)
      fits.append(fit)
      print(fit.summary())
      print(fit.diagnose())
      fitdf = pd.DataFrame({
          k: v.ravel()
          for k, v in fit.stan_variables().items()
          if 1 == len([s for s in v.shape if s > 1])
      })
      print('median:',
            fitdf.median().values, 'lp median:', np.median(fit.method_variables()["lp__"], axis=0))
      print("\n".join([
          f'{x}@{t:0.2f} {p:0.2f}{"🔥" if x<2 else ""}'
          for x, t, p in zip(t.results, t.dts_hours,
                             fit.stan_variables()['prob'].mean(axis=0))
      ]))
      pd.plotting.scatter_matrix(fitdf, hist_kwds=dict(bins=100))

      print('---')

  if False:
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
# For train[2]
NO easy/hard
median: [2.89923 3.38762] lp median: [-87.5495  -87.56525]

BINOMIAL: hard=1, easy=2, out of 2
median: [2.84198  3.169745] lp median: [-149.881 -149.89 ]

BINOMIAL only easy=2 of 2
median: [2.89488  3.472195] lp median: [-88.52505 -88.5313 ]


clamping left value, uninformative prior between 0 and 1: median -> 0.8! Huh.

clampleft and clampwidth:
median: [2.81087  3.2101   0.650987 2.678775] lp median: [-152.252  -152.2855]

clampWidth only (clampLeft=0.8)
median: [2.804665 3.21721  2.153885] lp median: [-150.236 -150.224]

# train[0]
clampleft and clampwidth:
median: [1.649785  1.98162   0.41851   0.8821025] lp median: [-83.5881 -83.5535]

clampWidth only (clampLeft=0.8)
median: [1.660475 2.050365 0.652474] lp median: [-81.9799  -81.96575]
"""
"""TOY DATA

no clamp:

# MPMATH
## exp
{'eb': mpf('2.8634392862906148'), 'eh': mpf('4.7606810766369136'), 'quad': {'den': (mpf('0.056114811374196646'), mpf('1.0e-19')), 'eb': (mpf('0.16068135543166212'), mpf('1.0e-21')), 'eh': (mpf('0.26714472062818784'), mpf('1.0e-19'))}}
## gam
{'eb': mpf('2.3173210884391482'), 'eh': mpf('0.94378028824502802'), 'quad': {'den': (mpf('2.3525200533613245e-12'), mpf('1.0e-17')), 'eb': (mpf('5.451544330630188e-12'), mpf('1.0e-22')), 'eh': (mpf('2.2202620540635595e-12'), mpf('1.0e-17'))}}
# SCIPY
## exp
{'eb': 2.8634392862916513, 'eh': 4.760681076637943, 'quad': {'den': (0.056114811374175996, 9.980147706392323e-11), 'eb': (0.16068135543166115, 9.96352802156054e-11), 'eh': (0.26714472062814726, 9.80622793033094e-11)}}
## gamma
{'eb': 2.3171043781014133, 'eh': 0.9439202808031762, 'quad': {'den': (2.35086222804235e-12, 5.2123119808928056e-12), 'eb': (5.447193160910173e-12, 1.2216277034852655e-11), 'eh': (2.2190265344233156e-12, 4.589201991469663e-12)}}

WITH clamp:

# MPMATH
## exp
{'eb': mpf('2.923423191378165'), 'eh': mpf('4.2658585890674985'), 'quad': {'den': (mpf('0.038266724598723378'), mpf('1.0e-5')), 'eb': (mpf('0.11186983014998923'), mpf('1.0e-5')), 'eh': (mpf('0.16324043580494466'), mpf('1.0e-5'))}}
## gam
{'eb': mpf('2.3173210916402449'), 'eh': mpf('0.94378027504609563'), 'quad': {'den': (mpf('2.3525200439207764e-12'), mpf('1.0e-17')), 'eb': (mpf('5.4515443162840505e-12'), mpf('1.0e-22')), 'eh': (mpf('2.2202620141030033e-12'), mpf('1.0e-17'))}}
# SCIPY
/Users/fasih/Dropbox/Anki/likelihood-demo/lib/python3.9/site-packages/scipy/integrate/quadpack.py:879: IntegrationWarning: The maximum number of subdivisions (50) has been achieved.
  If increasing the limit yields no improvement it is advised to analyze 
  the integrand in order to determine the difficulties.  If the position of a 
  local difficulty can be determined (singularity, discontinuity) one will 
  probably gain from splitting up the interval and calling the integrator 
  on the subranges.  Perhaps a special-purpose integrator should be used.
  quad_r = quad(f, low, high, args=args, full_output=self.full_output,
## exp
{'eb': 2.9231658425494307, 'eh': 4.265681801997901, 'quad': {'den': (0.038225211702996725, 1.2266907273146143e-10), 'eb': (0.11173863317442079, 1.4428062950480765e-08), 'eh': (0.1630565899389903, 9.985758454321393e-10)}}
## gamma
{'eb': 2.317104378233146, 'eh': 0.9439202799611425, 'quad': {'den': (2.350862227404872e-12, 5.2123119808928056e-12), 'eb': (5.447193159742755e-12, 1.2216277034852655e-11), 'eh': (2.219026531842082e-12, 4.589201991469663e-12)}}
"""
""" SUBTRAIN, mpmath, dps=15
# gamma, beta=10
{'eb': mpf('4.4371327827181259'), 'eh': mpf('0.76835326467461573'), 'quad': {'den': (mpf('9.6323905210106289e-19'), mpf('9.6323905049456013e-19')), 'eb': (mpf('4.2740195756719594e-18'), mpf('4.2740195630239893e-18')), 'eh': (mpf('7.40107870343934e-19'), mpf('7.4010786936491187e-19'))}}
{'eb': mpf('4.9033380875508898'), 'eh': mpf('0.67982918147402172'), 'quad': {'den': (mpf('1.0682441316238334e-22'), mpf('1.0682441207038355e-22')), 'eb': (mpf('5.2379621373938687e-22'), mpf('5.2379620514209078e-22')), 'eh': (mpf('7.2622353361625778e-23'), mpf('7.2622352666211345e-23'))}}
{'eb': mpf('4.903351273843529'), 'eh': mpf('1.6269537178728586'), 'quad': {'den': (mpf('1.5916704896764054e-33'), mpf('1.5916704583850888e-33')), 'eb': (mpf('7.8045195230939558e-33'), mpf('7.8045192767379419e-33')), 'eh': (mpf('2.5895742208075411e-33'), mpf('2.58957416990099e-33'))}}

# gamma, beta=2
{'eb': mpf('3.6798252065763015'), 'eh': mpf('1.935061359161069'), 'quad': {'den': (mpf('6.2364835275217659e-6'), mpf('1.0e-7')), 'eb': (mpf('2.2949169284972484e-5'), mpf('1.0e-6')), 'eh': (mpf('1.2067978291151887e-5'), mpf('1.0e-6'))}}
{'eb': mpf('3.6281375019261355'), 'eh': mpf('2.267483752144944'), 'quad': {'den': (mpf('1.3272549552559431e-8'), mpf('1.0e-8')), 'eb': (mpf('4.8154634777813824e-8'), mpf('1.0e-8')), 'eh': (mpf('3.0095290459967159e-8'), mpf('1.0e-8'))}}
{'eb': mpf('4.8364182702505127'), 'eh': mpf('5.2203842564821343'), 'quad': {'den': (mpf('2.8800361355635346e-13'), mpf('1.0e-13')), 'eb': (mpf('1.3929059385021162e-12'), mpf('1.0e-12')), 'eh': (mpf('1.5034895300195523e-12'), mpf('1.0e-12'))}}
same but with constants:
{'eb': mpf('3.6798252031421286'), 'eh': mpf('1.9350613563697521'), 'quad': {'den': (mpf('5.9060159273508187e-5'), mpf('1.0e-6')), 'eb': (mpf('0.00021733106259624372'), mpf('1.0e-5')), 'eh': (mpf('0.00011428503191120834'), mpf('1.0e-5'))}}
{'eb': mpf('3.6281366143368046'), 'eh': mpf('2.2674831973395406'), 'quad': {'den': (mpf('1.2569248005731195e-7'), mpf('1.0e-8')), 'eb': (mpf('4.5602948904273211e-7'), mpf('1.0e-7')), 'eh': (mpf('2.8500558656189017e-7'), mpf('1.0e-7'))}}
{'eb': mpf('4.9247464287717575'), 'eh': mpf('5.3421612382820909'), 'quad': {'den': (mpf('2.6784386871115843e-12'), mpf('1.0e-12')), 'eb': (mpf('1.319063135903689e-11'), mpf('1.0e-11')), 'eh': (mpf('1.4308651333402678e-11'), mpf('1.0e-11'))}}
constants don't seem to help reduce error at default dps
same with quad instead of quadgl:
{'eb': mpf('3.6863879855795738'), 'eh': mpf('1.9331212015853561'), 'quad': {'den': (mpf('6.0602899280021719e-5'), mpf('1.0e-6')), 'eb': (mpf('0.00022340579979716107'), mpf('1.0e-6')), 'eh': (mpf('0.00011715274947575189'), mpf('1.0e-6'))}}
{'eb': mpf('3.6347629269971375'), 'eh': mpf('2.3236293230633454'), 'quad': {'den': (mpf('1.2958862497152868e-7'), mpf('1.0e-9')), 'eb': (mpf('4.7102392980704791e-7'), mpf('1.0e-8')), 'eh': (mpf('3.0111592891930289e-7'), mpf('1.0e-12'))}}
{'eb': mpf('5.0798360098663444'), 'eh': mpf('5.47481627443267'), 'quad': {'den': (mpf('2.7162668149647606e-12'), mpf('1.0e-16')), 'eb': (mpf('1.3798189979062953e-11'), mpf('1.0e-11')), 'eh': (mpf('1.4871061764270465e-11'), mpf('1.0e-11'))}}
quad with maxdegrees=8 MUCH SLOWER:
maxdegree= 8
{'eb': mpf('3.6818011722301374'), 'eh': mpf('1.9361193310793698'), 'quad': {'den': (mpf('5.9801860438925261e-5'), mpf('1.0e-10')), 'eb': (mpf('0.0002201785598655781'), mpf('1.0e-10')), 'eh': (mpf('0.00011578353803031381'), mpf('1.0e-9'))}}
maxdegree= 8
{'eb': mpf('3.6788346049551319'), 'eh': mpf('2.3253028836372853'), 'quad': {'den': (mpf('1.2756315246058749e-7'), mpf('1.0e-9')), 'eb': (mpf('4.6928373958917662e-7'), mpf('1.0e-9')), 'eh': (mpf('2.9662296626246674e-7'), mpf('1.0e-10'))}}
maxdegree= 8
{'eb': mpf('4.977199148961355'), 'eh': mpf('5.3926403252507749'), 'quad': {'den': (mpf('2.8154877546231069e-12'), mpf('1.0e-13')), 'eb': (mpf('1.4013243256221243e-11'), mpf('1.0e-13')), 'eh': (mpf('1.5182912800830325e-11'), mpf('1.0e-13'))}}

# exp
{'eb': mpf('3.4017221073207571'), 'eh': mpf('4.1995335093012054'), 'quad': {'den': (mpf('0.00025732638102475168'), mpf('1.0e-5')), 'eb': (mpf('0.00087535283912874233'), mpf('1.0e-5')), 'eh': (mpf('0.0010806507599406545'), mpf('1.0e-6'))}}
{'eb': mpf('3.2443691267612786'), 'eh': mpf('5.0666280535106907'), 'quad': {'den': (mpf('1.0480515787537876e-6'), mpf('1.0e-7')), 'eb': (mpf('3.4002661853622052e-6'), mpf('1.0e-6')), 'eh': (mpf('5.3100875304401095e-6'), mpf('1.0e-10'))}}
{'eb': mpf('5.1497050501036155'), 'eh': mpf('11.973957434125394'), 'quad': {'den': (mpf('1.4393492914181767e-8'), mpf('1.0e-8')), 'eb': (mpf('7.4122243148792456e-8'), mpf('1.0e-8')), 'eh': (mpf('1.7234707148279795e-7'), mpf('1.0e-7'))}}

# gamma with dps=30, beta=10
{'eb': mpf('2.68826746659870445439388775173029'), 'eh': mpf('0.997113671423464195401819981491926'), 'quad': {'den': (mpf('2.61854221433326359644004729975889e-15'), mpf('1.0e-16')), 'eb': (mpf('7.03934184470744429565169183660199e-15'), mpf('1.0e-16')), 'eh': (mpf('2.61098424111116815397533360916485e-15'), mpf('1.0e-21'))}}
{'eb': mpf('2.6572217819552988780875320152'), 'eh': mpf('1.14993761558596158437532741216667'), 'quad': {'den': (mpf('1.13648074522571051421397568593537e-18'), mpf('1.0e-20')), 'eb': (mpf('3.01988139098654852045228984811882e-18'), mpf('1.0e-19')), 'eh': (mpf('1.30688195832421024348015294746962e-18'), mpf('1.0e-20'))}}
{'eb': mpf('4.90335127384352915968144148996443'), 'eh': mpf('1.6269537178728587811268130839529'), 'quad': {'den': (mpf('1.59167048967640526473350890292076e-33'), mpf('1.59167045838508876451360035613473e-33')), 'eb': (mpf('7.80451952309395558354143951001413e-33'), mpf('7.80451927673794189640474678188905e-33')), 'eh': (mpf('2.58957422080754123623101615132951e-33'), mpf('2.58957416990099003651163933055161e-33'))}}

# gamma with dps=30, beta=2
{'eb': mpf('3.67857545505472302394153361212092'), 'eh': mpf('1.93966083242436524698270673659994'), 'quad': {'den': (mpf('0.00000626450405789605890856275966186957'), mpf('0.0000001')), 'eb': (mpf('0.0000230444508654671538478151375162568'), mpf('0.0000001')), 'eh': (mpf('0.0000121510131556644836038570839502729'), mpf('0.0000001'))}}
{'eb': mpf('3.64184308600241914573959165114038'), 'eh': mpf('2.30796552039987044005813846327807'), 'quad': {'den': (mpf('0.0000000137309977339843860337211628701152'), mpf('0.000000001')), 'eb': (mpf('0.0000000500061391614259207935308907938409'), mpf('0.00000001')), 'eh': (mpf('0.0000000316906693307247152905236397024935'), mpf('0.000000001'))}}
{'eb': mpf('4.91922487759379059856781318353515'), 'eh': mpf('5.31546561599123462718403796477323'), 'quad': {'den': (mpf('3.04397675962068556206235761653669e-13'), mpf('1.0e-14')), 'eb': (mpf('1.49740062027434102828135562800919e-12'), mpf('1.0e-14')), 'eh': (mpf('1.61801538016401697163488965987879e-12'), mpf('1.0e-13'))}}

# gamma with dps=60, beta=10
{'eb': mpf('2.68925408278419661229179493033106919520707879446191063530031666'), 'eh': mpf('0.996653648194098407888134925058481826121084420422586517373613489'), 'quad': {'den': (mpf('0.00000000000000262028997955962164054937227037515833196096683813603978442628267'), mpf('0.0000000000000001')), 'eb': (mpf('0.00000000000000704662552560923158441572797519750029094985439133849111636601242'), mpf('0.000000000000001')), 'eh': (mpf('0.00000000000000261152156745453645479103276396047908438540346051096218016465034'), mpf('0.00000000000000001'))}}
{'eb': mpf('2.65735862252318426391021632622206594477068792804933463352582562'), 'eh': mpf('1.14723556314037665349867765229569515914458128340278738604196241'), 'quad': {'den': (mpf('0.00000000000000000114515423597174507793785033558364234717353164330521052666098177'), mpf('0.00000000000000000001')), 'eb': (mpf('0.00000000000000000304308548307846600725475218845741578302092690105891941272427615'), mpf('0.0000000000000000001')), 'eh': (mpf('0.00000000000000000131376166478763273596605721127403877173378730448634896049821644'), mpf('0.00000000000000000001'))}}
{'eb': mpf('3.49252832704364503328474191992689504670746647734995171730099365'), 'eh': mpf('2.7000233833363129611259486449927745539472794910481954779625646'), 'quad': {'den': (mpf('1.2999116119807059451002034995508856349418972733278762743313675e-31'), mpf('1.0e-32')), 'eb': (mpf('4.53997812749558277629188801011384566410154618378098324550414469e-31'), mpf('1.0e-32')), 'eh': (mpf('3.50979174861830612004490012425211733204158847679739091651570922e-31'), mpf('1.0e-32'))}}
"""