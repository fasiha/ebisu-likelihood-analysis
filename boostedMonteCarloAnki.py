from functools import cache
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


@cache
def binomln(n, k):
  "Log of scipy.special.binom calculated entirely in the log domain"
  # assert np.logical_and(0 <= k, k <= n).all(), "0 <= k <= n"
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


#


def clampLerp2(x1, x2, y1, y2, x):
  if x <= x1:
    return y1
  if x >= x2:
    return y2
  mu = (x - x1) / (x2 - x1)
  return (y1 * (1 - mu) + y2 * mu)


def makeHalflives(b, h, ts, clamp):
  hs = [h]
  for t in ts[1:]:
    old = hs[-1]
    if clamp:
      hs.append(old * clampLerp2(0.8 * old, old, min(b, 1.0), b, t))
    else:
      hs.append(old * b)
  return hs


def ankiFitEasyHardMAP(xs: list[int], ts: list[float], priors, clamp, binomial):
  from math import fsum

  def posterior(b, h):
    logb = np.log(b)
    logh = np.log(h)
    if priors == 'gamma':
      ab = 2 * 1.4 + 1
      bb = 2.0
      ah = 2 * .25 + 1
      bh = 2.0

      logprior = -bb * b - bh * h + (ab - 1) * logb + (ah - 1) * logh
      # prior = b**(ab - 1) * np.exp(-bb * b - bh * h) * h**(ah - 1)
    elif priors == 'exp':
      bb = 1.0
      bh = 0.5
      logprior = -bb * b - bh * h
      # prior = np.exp(-bb * b - bh * h)
    else:
      raise Exception('unknown priors')
    hs = makeHalflives(b, h, ts, clamp)

    if binomial:
      loglik = []
      for x, t, h in zip(xs, ts, hs):
        if x == 1:
          loglik.append(np.log(-np.expm1(-t / h)))
        elif x == 3:
          loglik.append(-t / h)
        elif x == 2 or x == 4:
          n = 3
          logp = -t / h
          k = 2 if x < 3 else 3
          loglik.append(binomln(n, k) + k * logp + (n - k) * np.log(-np.expm1(logp)))
        else:
          raise Exception('unknown result')
    else:
      loglik = [-t / h if x > 1 else np.log(-np.expm1(-t / h)) for x, t, h in zip(xs, ts, hs)]
    return fsum(loglik + [logprior])

  bvec = np.linspace(0.5, 15, 101)
  hvec = np.linspace(0.1, 15, 101)
  f = np.vectorize(posterior)
  bmat, hmat = np.meshgrid(bvec, hvec)
  z = f(bmat, hmat)

  import pylab as plt  #type:ignore
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

  idxbest = z.argmax()
  bestb = bmat.ravel()[idxbest]
  besth = hmat.ravel()[idxbest]
  summary = []
  for x, t, h in zip(xs, ts, makeHalflives(bestb, besth, ts, clamp)):
    summary.append(f'{x}@{t:0.2f} {"🔥" if x<2 else ""} p={np.exp(-t/h):0.2f}')

  return dict(
      z=z,
      bvec=bvec,
      hvec=hvec,
      bmat=bmat,
      hmat=hmat,
      fig=fig,
      ax=ax,
      im=im,
      bestb=bestb,
      besth=besth,
      summary=summary,
  )


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

  if False:
    fracs = [0.7, 0.8, 0.9]
    subtrain = [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]
    reses = []
    priors = 'exp'
    clamp = True
    binomial = False
    for t in subtrain:
      title = f'Card {t.df.cid.iloc[0]}'
      print(f'\n## {title}, priors={priors}, clamp={clamp}, binomial={binomial}')

      res = ankiFitEasyHardMAP(
          t.results, t.dts_hours, priors=priors, clamp=clamp, binomial=binomial)
      res['ax'].set_title(title)
      reses.append(res)
      print(f'> best h={res["besth"]}, b={res["bestb"]}')

  if True:
    t = next(t for t in train if t.fractionCorrect > 0.9)
    priors = 'exp'
    clamp = True
    binomial = False

    reses = []
    for i in range(len(t.results)):
      title = f'{i+1}'
      res = ankiFitEasyHardMAP(
          t.results[:i + 1], t.dts_hours[:i + 1], priors=priors, clamp=clamp, binomial=binomial)
      plt.close()
      res['summary'].insert(0, title + f'h={res["besth"]:0.2f}, b={res["bestb"]:0.2f}')
      reses.append(res)
    from itertools import zip_longest
    with open('res.tsv', 'w') as fid:
      fid.write('\n'.join(
          ['\t'.join(z) for z in zip_longest(*[r['summary'] for r in reses], fillvalue='')]))
      fid.write(f'\n\nCard {t.df.cid.iloc[0]}')

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
## Card 1300038031922, priors=exp, clamp=True, binomial=False <==> ## Card 1300038031922, priors=exp, clamp=True, binomial=True
2@0.02  p=0.99 <==>  p=0.99
2@6.92  p=0.44 <==>  p=0.36
2@24.29  p=0.37 <==>  p=0.38
1@19.37 🔥 p=0.45 <==> 🔥 p=0.46
3@21.12  p=0.58 <==>  p=0.59
3@69.23  p=0.54 <==>  p=0.63
3@153.46  p=0.63 <==>  p=0.76
1@265.31 🔥 p=0.45 <==> 🔥 p=0.62
3@26.66  p=0.92 <==>  p=0.95
1@446.41 🔥 p=0.63 <==> 🔥 p=0.45
2@25.51  p=0.97 <==>  p=0.96
> best h=2.8813333333333335, b=2.9166666666666665 <==> > best h=1.7886666666666666, b=3.7383333333333333

## Card 1300038030806, priors=exp, clamp=True, binomial=False <==> ## Card 1300038030806, priors=exp, clamp=True, binomial=True
2@0.31  p=0.93 <==>  p=0.93
2@22.66  p=0.20 <==>  p=0.20
1@70.26 🔥 p=0.20 <==> 🔥 p=0.20
1@34.07 🔥 p=0.46 <==> 🔥 p=0.46
3@13.71  p=0.73 <==>  p=0.73
3@43.79  p=0.72 <==>  p=0.72
3@148.01  p=0.70 <==>  p=0.70
3@204.28  p=0.62 <==>  p=0.62
3@357.86  p=0.57 <==>  p=0.57
3@541.10  p=0.59 <==>  p=0.59
4@1031.98  p=0.72 <==>  p=0.72
1@2545.38 🔥 p=0.45 <==> 🔥 p=0.45
4@49.09  p=0.98 <==>  p=0.98
4@192.39  p=0.94 <==>  p=0.94
1@693.48 🔥 p=0.80 <==> 🔥 p=0.80
4@47.80  p=0.99 <==>  p=0.99
4@119.26  p=0.96 <==>  p=0.96
3@359.83  p=0.89 <==>  p=0.89
3@747.60  p=0.79 <==>  p=0.79
3@94.93  p=0.97 <==>  p=0.97
1@1847.51 🔥 p=0.56 <==> 🔥 p=0.56
3@27.00  p=0.99 <==>  p=0.99
3@96.87  p=0.97 <==>  p=0.97
4@236.92  p=0.93 <==>  p=0.93
3@809.26  p=0.78 <==>  p=0.78
3@713.24  p=0.80 <==>  p=0.80
> best h=4.520333333333332, b=3.11 <==> > best h=4.520333333333332, b=3.11

## Card 1300038030485, priors=exp, clamp=True, binomial=False <==> ## Card 1300038030485, priors=exp, clamp=True, binomial=True
3@23.38  p=0.12 <==>  p=0.11
3@118.24  p=0.10 <==>  p=0.09
2@22.42  p=0.64 <==>  p=0.63
2@76.28  p=0.72 <==>  p=0.72
2@192.44  p=0.58 <==>  p=0.58
2@119.20  p=0.71 <==>  p=0.72
3@347.87  p=0.79 <==>  p=0.79
2@338.48  p=0.80 <==>  p=0.80
2@420.03  p=0.76 <==>  p=0.75
2@841.76  p=0.57 <==>  p=0.57
1@964.46 🔥 p=0.53 <==> 🔥 p=0.52
4@15.33  p=0.99 <==>  p=0.99
4@118.73  p=0.92 <==>  p=0.92
3@314.67  p=0.81 <==>  p=0.81
1@668.11 🔥 p=0.64 <==> 🔥 p=0.64
4@26.48  p=0.98 <==>  p=0.98
3@108.05  p=0.93 <==>  p=0.93
4@128.55  p=0.92 <==>  p=0.92
4@411.86  p=0.76 <==>  p=0.76
4@1391.14  p=0.75 <==>  p=0.77
3@4992.25  p=0.80 <==>  p=0.78
> best h=10.927333333333333, b=4.608333333333333 <==> > best h=10.480333333333332, b=4.705
"""