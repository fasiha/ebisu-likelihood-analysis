from scipy.linalg import lstsq  #type:ignore
import typing
import scipy.optimize as opt  #type:ignore
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


def makeHalflives(b, h, xs, ts, left, right, boostRule):
  hs = [h]
  if boostRule:
    for x, t in zip(xs, ts):
      old = hs[-1]
      if x > 1:
        if boostRule == 'step':
          maxb = b
        elif boostRule == 'linear':
          maxb = b * (x - 1)
        elif boostRule == 'log':
          maxb = b * np.log(x)
        else:
          raise Exception('unknown boostRule')
        hs.append(old * clampLerp2(left * old, right * old, min(b, 1.0), maxb, t))
      else:
        hs.append(old)
  else:
    for x, t in zip(xs, ts):
      old = hs[-1]
      hs.append(old * clampLerp2(left * old, right * old, min(b, 1.0), b, t))
  return hs


def samplePeak(f: typing.Callable[[float], float],
               x0: float,
               xmin: float,
               xmax: float,
               ydown: float,
               nsamples: int,
               doubleIters: int = 5):
  ydown = abs(ydown)
  y0 = f(x0)
  samples = [(x0, y0)]
  x = x0
  for i in range(doubleIters):
    x = (x + xmin) / 2
    y = f(x)
    samples.append((x, y))
    if y < (y0 - ydown):
      break
  xleft = x
  x = x0
  for i in range(doubleIters):
    if xmax < np.inf:
      x = (xmax + x) / 2
    else:
      x *= 2
    y = f(x)
    samples.append((x, y))
    if y < (y0 - ydown):
      break
  xright = x
  xv = np.linspace(xleft, xright, nsamples + 2)  # 2 extra because we'll discard xleft and xright
  for x in xv[1:-1]:  # skip first and last , which we already have in `xs`
    y = f(x)
    samples.append((x, y))
  return samples


def ankiFitEasyHardMAP(xs: list[int],
                       ts: list[float],
                       binomial,
                       left,
                       right,
                       ah,
                       bh,
                       ab,
                       bb,
                       boostRule,
                       viz=False):
  from math import fsum

  LOG_HALF = -np.log(0.5)

  def posterior(b, h, extra=False):
    logb = np.log(b)
    logh = np.log(h)
    logprior = -bb * b - bh * h + (ab - 1) * logb + (ah - 1) * logh
    hs = makeHalflives(b, h, xs, ts, left, right, boostRule)

    if binomial:
      loglik = []
      for x, t, h in zip(xs, ts, hs):
        logp = -t / h * LOG_HALF
        if x == 1:
          loglik.append(np.log(-np.expm1(logp)))
        elif x == 3:
          loglik.append(logp)
        elif x == 2:  # hard
          n = 2
          k = 1
          loglik.append(binomln(n, k) + k * logp + (n - k) * np.log(-np.expm1(logp)))
        else:  # x>=4
          n = 2 + (x - 4)
          k = n
          loglik.append(binomln(n, k) + k * logp + (n - k) * np.log(-np.expm1(logp)))
    else:
      loglik = [
          -t / h * LOG_HALF if x > 1 else np.log(-np.expm1(-t / h * LOG_HALF))
          for x, t, h in zip(xs, ts, hs)
      ]
    logposterior = fsum(loglik + [logprior])
    if extra:
      return dict(logposterior=logposterior, loglikelihood=fsum(loglik), logprior=logprior)
    return logposterior

  if viz:
    bvec = np.linspace(0.5, 15, 301)
    hvec = np.linspace(0.1, 48, 301)
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
    vizdict = dict(fig=fig, ax=ax, im=im)
  else:
    vizdict = dict()

  MIN_BOOST = 1.0
  res = opt.shgo(lambda x: -posterior(*x), [(MIN_BOOST, 5), (6, 50)])
  print(res.message)
  bestb, besth = res.x

  if viz:
    mark = ax.plot([bestb], [besth], marker='o')

  if not viz:
    # same as above
    bvec = np.linspace(0.5, 15, 301)
    hvec = np.linspace(0.1, 48, 301)

  def clean(x, y):
    x = np.array(x)
    y = np.array(y)
    top = max(y)
    LIMIT = 10
    bot = top - LIMIT
    idx = y > bot
    return (x[idx], y[idx])

  fixBoostVaryHl = [posterior(bestb, h) for h in hvec]
  varyHl, fixBoostVaryHl = clean(hvec, fixBoostVaryHl)

  fixHlVaryBoost = [posterior(b, besth) for b in bvec]
  varyBoost, fixHlVaryBoost = clean(bvec, fixHlVaryBoost)

  boostSamples = []
  hlSamples = []
  posteriorSamples = []
  LIMIT = 3
  for b, post in samplePeak(lambda b: posterior(b, besth), bestb, MIN_BOOST, np.inf, LIMIT, 15):
    boostSamples.append(b)
    posteriorSamples.append(post)
    hlSamples.append(besth)
  for h, post in samplePeak(lambda h: posterior(bestb, h), besth, 0, np.inf, LIMIT, 15):
    hlSamples.append(h)
    posteriorSamples.append(post)
    boostSamples.append(bestb)

  bestloglikelihood = posterior(bestb, besth, True)['loglikelihood']
  summary = []
  halflives = makeHalflives(bestb, besth, xs, ts, left, right, boostRule)
  for x, t, h in zip(xs, ts, halflives):
    summary.append(
        f'{x}@{t:0.2f} {"🔥" if x<2 else ""} (halflife was {h:0.1f} hours) pRecall={np.exp(-t/h):0.2f}'
    )

  return dict(
      boostSamples=np.array(boostSamples),
      hlSamples=np.array(hlSamples),
      posteriorSamples=np.array(posteriorSamples),
      viz=vizdict,
      bestb=bestb,
      besth=besth,
      summary=summary,
      bestloglikelihood=bestloglikelihood,
      halflives=halflives,
      fixBoostVaryHl=fixBoostVaryHl,
      fixHlVaryBoost=fixHlVaryBoost,
      varyHl=varyHl,
      varyBoost=varyBoost,
      shgo=res,
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
  rescalec = lambda im, top: im.set_clim(im.get_clim()[1] - np.array([top, 0]))

  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = utils.traintest(df)
  # train = train[::10]  # further subdivide, for computational purposes
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  if False:
    x = np.linspace(0, 5, 501)
    mode = 1.5
    plt.figure()
    for b in [0.1, 0.5, 1.0, 2.0]:
      plt.plot(x, gammarv.pdf(x, mode * b + 1, scale=1 / b), label=f'b={b:0.2f}')
    plt.legend()

  MODE_BOOST = 1.5  # as in, "mean median mode"
  if False:
    bb = 1.0
    kws = dict(
        binomial=True,
        left=0.3,
        right=1,
        ah=1.0,
        bh=0.1,
        ab=MODE_BOOST * bb + 1,
        bb=bb,
    )
    resFail = ankiFitEasyHardMAP([1, 1, 1], [1., 3., 9.], **kws)
    resHard = ankiFitEasyHardMAP([2, 2, 2], [1., 3., 9.], **kws)
    resMedium = ankiFitEasyHardMAP([3, 3, 3], [1., 3., 9.], **kws)
    resEasy = ankiFitEasyHardMAP([4, 4, 4], [1., 3., 9.], **kws)
    from pprint import pprint
    for r in [resFail, resHard, resMedium, resEasy]:
      pprint(r)
    # Should be monotonically increasing:
    finalHls = [r['halflives'][-1] for r in [resFail, resHard, resMedium, resEasy]]
    if not np.all(np.diff(finalHls) > 0):
      print('final halflives should ideally be monotonic!')
      print('final halflives should ideally be monotonic!')
      print('final halflives should ideally be monotonic!')
      print(finalHls)

  if False:
    t = next(t for t in train if t.fractionCorrect > 0.95)
    binomial = True
    right = 1.0
    left = 0.3
    ah = 1.0
    bh = 0.2
    bb = 1.0
    ab = MODE_BOOST * bb + 1
    boostRule = 'step'
    for i in [0, 0.1, 0.5, 1, 5, 10]:
      xs = t.results[:-1] if i == 0 else list(t.results[:-1]) + [1]
      ts = t.dts_hours[:-1] if i == 0 else list(t.dts_hours[:-1]) + [t.dts_hours[-1] * i]
      res = ankiFitEasyHardMAP(
          xs,
          ts,
          binomial=binomial,
          left=left,
          right=right,
          ah=ah,
          bh=bh,
          ab=MODE_BOOST * bb + 1,
          bb=bb,
          boostRule=boostRule,
      )
      print(
          f'\n## binomial={binomial}, left={left}, right={right}, ah={ah}, bh={bh}, ab={ab}, bb={bb}, final={i}, boostRule={boostRule}'
      )
      print("\n".join(res['summary']))
      print(
          f'> best h={res["besth"]:0.2f}, b={res["bestb"]:0.2f}, final hl={res["halflives"][-1]:0.2f}, loglik={res["bestloglikelihood"]:0.2f}'
      )

  if True:
    fracs = [0.9, 0.95]
    subtrain = [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]

    # lens = [6, 7, 8, 9]
    # subtrain.extend([next(t for t in train if t.len == l) for l in lens])

    reses = []
    binomial = True
    right = 1.0
    left = 0.3
    ah = 1.0
    bh = 0.2
    bb = 3.0
    ab = MODE_BOOST * bb + 1
    boostRule = 'step'
    for t in subtrain:
      title = f'Card {t.df.cid.iloc[0]}'
      print(
          f'\n## {title},  binomial={binomial}, left={left}, right={right}, ah={ah}, bh={bh}, ab={ab}, bb={bb}, boostRule={boostRule}'
      )

      res = ankiFitEasyHardMAP(
          t.results,
          t.dts_hours,
          binomial=binomial,
          left=left,
          right=right,
          ah=ah,
          bh=bh,
          ab=ab,
          bb=bb,
          boostRule=boostRule,
          viz=False,
      )

      def errConst(x, y):
        return np.sum(((x - np.mean(x)) - (y - np.mean(y)))**2)

      def errAffine(x, y):
        "minimize `|y - a*x + b|` for a and b"
        vec = lambda v: np.array(v).ravel()[:, np.newaxis]
        A = np.hstack([vec(x), vec(np.ones_like(x))])
        sol = lstsq(A, y)
        return sol[1]  # error
        # equivalent to
        # err = np.dot(A, sol[0]) - y
        # return np.dot(err, err)

      # errAffine([1,2,3.0], [1,2,3.0])

      def fitter2d(x, y, fx, fy, modex, modey, logdomain=True):
        pdf = gammarv.logpdf if logdomain else gammarv.pdf
        xall = np.hstack([x, modex * np.ones_like(y)])
        yall = np.hstack([modey * np.ones_like(x), y])
        zall = np.hstack([fx, fy])
        if not logdomain:
          clean = lambda z: np.exp(z - np.max(z))
          zall = clean(zall)
          fx = clean(fx)  # just for plot
          fy = clean(fy)  # just for plot
        errfix = errConst if logdomain else errAffine

        def objective(v):
          alphax, alphay = v
          betax = (alphax - 1) / modex
          betay = (alphay - 1) / modey
          if logdomain:
            zfit = (pdf(xall, alphax, scale=1 / betax) + pdf(yall, alphay, scale=1 / betay))
          else:
            zfit = (pdf(xall, alphax, scale=1 / betax) * pdf(yall, alphay, scale=1 / betay))
          return errfix(zfit, zall)

        fin = opt.shgo(objective, [(1.01, 20.), (1.01, 20)])
        alphax, alphay = fin.x
        betax = (alphax - 1) / modex
        betay = (alphay - 1) / modey

        return dict(fin=fin, alphax=alphax, betax=betax, alphay=alphay, betay=betay)

      def fitsamples(xall, yall, zall):  # wls 4d
        weights = np.diag(np.exp(zall - np.max(zall)))**3
        A = np.vstack([np.log(xall), -xall, np.log(yall), -yall, np.ones_like(xall)]).T
        sol = lstsq(np.dot(weights, A), np.dot(weights, zall))
        t = sol[0]
        alphax = t[0] + 1
        betax = t[1]
        alphay = t[2] + 1
        betay = t[3]
        return dict(sol=sol, alphax=alphax, betax=betax, alphay=alphay, betay=betay)

      def fitterLin(x, y, fx, fy, modex, modey, ordinary=True):
        xall = np.hstack([x, modex * np.ones_like(y)])
        yall = np.hstack([modey * np.ones_like(x), y])
        zall = np.hstack([fx, fy])
        if ordinary:
          weights = np.eye(len(zall))
        else:  # weighted least squares: see note below
          weights = np.diag(np.exp(zall - np.max(zall)))

        A = np.vstack([np.log(xall), -xall, np.log(yall), -yall, np.ones_like(xall)]).T
        sol = lstsq(np.dot(weights, A), np.dot(weights, zall))
        t = sol[0]
        alphax = t[0] + 1
        betax = t[1]
        alphay = t[2] + 1
        betay = t[3]
        return dict(sol=sol, alphax=alphax, betax=betax, alphay=alphay, betay=betay)

      def fitterLin2d(x, y, fx, fy, modex, modey, ordinary=True):
        xall = np.hstack([x, modex * np.ones_like(y)])
        yall = np.hstack([modey * np.ones_like(x), y])
        zall = np.hstack([fx, fy])

        if ordinary:
          weights = np.eye(len(zall))
        else:  # weighted least squares: weight = exp(z) (technically sqrt(exp(z)))?)
          # move max to 0 because otherwise exp underflows
          weights = np.diag(np.exp(zall - np.max(zall)))

        A = np.vstack(
            [np.log(xall) - xall / modex,
             np.log(yall) - yall / modey,
             np.ones_like(xall)]).T
        sol = lstsq(np.dot(weights, A), np.dot(weights, zall))
        t = sol[0]
        alphax = t[0] + 1
        betax = (alphax - 1) / modex
        alphay = t[1] + 1
        betay = (alphay - 1) / modey
        return dict(sol=sol, alphax=alphax, betax=betax, alphay=alphay, betay=betay)

      def plotter(x, y, fx, fy, sols, labels):
        remmax = lambda v: v / np.max(v)
        fig, ax = plt.subplots(2)
        ax[0].plot(x, remmax(np.exp(fx)), label='post', linewidth=4)
        ax[1].plot(y, remmax(np.exp(fy)), label='post', linewidth=4)
        pdf = gammarv.pdf
        widths = [2, 1]
        styles = ['--', '-.', ':']

        for i, (sol, label) in enumerate(zip(sols, labels)):
          ax[0].plot(
              x,
              remmax(pdf(x, sol['alphax'], scale=1 / sol['betax'])),
              label=label,
              linewidth=widths[i % len(widths)],
              linestyle=styles[i % len(styles)])
          ax[1].plot(
              y,
              remmax(pdf(y, sol['alphay'], scale=1 / sol['betay'])),
              label=label,
              linewidth=widths[i % len(widths)],
              linestyle=styles[i % len(styles)])
        ax[0].legend()
        ax[1].legend()
        return fig, ax

      def cliponeside(x, fx, modex):
        xleft = modex - np.min(x[x < modex])
        xright = np.max(x[x > modex]) - modex
        xside = max(min(xleft, xright) * 2, 0.5)
        xidx = np.abs(x - modex) < xside
        return x[xidx], fx[xidx]

      def clipleftright(x, y, fx, fy, modex, modey):
        CLIP = False
        if not CLIP:
          print('NOT CLIPPING LEFT/RIGHT')
          x, y, fx, fy, modex, modey
        newx, newfx = cliponeside(x, fx, modex)
        newy, newfy = cliponeside(y, fy, modey)
        return newx, newy, newfx, newfy, modex, modey

      ressamples = fitsamples(res['boostSamples'], res['hlSamples'], res['posteriorSamples'])
      reslin = fitterLin(
          res['varyBoost'],
          res['varyHl'],
          res['fixHlVaryBoost'],
          res['fixBoostVaryHl'],
          res['bestb'],
          res['besth'],
          ordinary=False)
      reslin2d = fitterLin2d(res['varyBoost'], res['varyHl'], res['fixHlVaryBoost'],
                             res['fixBoostVaryHl'], res['bestb'], res['besth'])
      reslin2dw = fitterLin2d(
          res['varyBoost'],
          res['varyHl'],
          res['fixHlVaryBoost'],
          res['fixBoostVaryHl'],
          res['bestb'],
          res['besth'],
          ordinary=False)

      res2d = fitter2d(res['varyBoost'], res['varyHl'], res['fixHlVaryBoost'],
                       res['fixBoostVaryHl'], res['bestb'], res['besth'])
      res2dlin = fitter2d(
          res['varyBoost'],
          res['varyHl'],
          res['fixHlVaryBoost'],
          res['fixBoostVaryHl'],
          res['bestb'],
          res['besth'],
          logdomain=False)

      fig, ax = plotter(res['varyBoost'], res['varyHl'], res['fixHlVaryBoost'],
                        res['fixBoostVaryHl'], [reslin, reslin2d, reslin2dw, res2dlin, ressamples],
                        ['wls4d', 'ols2d', 'wls2d', 'shgolin', 'peaksample'])
      ax[0].set_xlabel('boost (unitless)')
      ax[1].set_xlabel('init hl (hours)')
      for a in ax:
        a.set_ylabel('normalized prob.')
      ax[0].set_title(
          f'Card {t.df.cid.iloc[0]}, {int(np.round(t.fractionCorrect * t.len))}/{t.len} pass')
      fig.tight_layout()

      if 'ax' in res['viz']:
        res['viz']['ax'].set_title(title)
      reses.append(res)
      if True:
        print("\n".join(res['summary']))
        print(
            f'> best h={res["besth"]:0.2f}, b={res["bestb"]:0.2f}, final hl={res["halflives"][-1]:0.2f}, loglik={res["bestloglikelihood"]:0.2f}'
        )

  if False:
    fracs = [0.7, 0.8, 0.9, 0.95]
    subtrain = [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]
    reses = []
    binomial = True
    right = 1.0
    left = 0.3
    ah = 1.0
    bh = 0.2
    bb = 1.0
    ab = MODE_BOOST * bb + 1
    boostRule = 'step'
    for t in subtrain:
      for boostRule in [False, 'step', 'linear', 'log']:
        title = f'Card {t.df.cid.iloc[0]}'
        print(
            f'\n## {title},  binomial={binomial}, left={left}, right={right}, ah={ah}, bh={bh}, ab={ab}, bb={bb}, boostRule={boostRule}'
        )

        res = ankiFitEasyHardMAP(
            t.results,
            t.dts_hours,
            binomial=binomial,
            left=left,
            right=right,
            ah=ah,
            bh=bh,
            ab=ab,
            bb=bb,
            boostRule=boostRule,
        )
        if 'ax' in res['viz']:
          res['viz']['ax'].set_title(title)
        reses.append(res)
        print("\n".join(res['summary']))
        print(
            f'> best h={res["besth"]:0.2f}, b={res["bestb"]:0.2f}, final hl={res["halflives"][-1]:0.2f}, loglik={res["bestloglikelihood"]:0.2f}'
        )

  if True:
    t = next(t for t in train if t.fractionCorrect > 0.9)
    priors = 'gamma'
    clamp = True
    binomial = False
    gridsearch = False
    ah = 1.0
    ab = MODE_BOOST * bb + 1
    boostRule = 'step'

    reses = []
    for i in range(len(t.results)):
      title = f'{i+1}'
      res = ankiFitEasyHardMAP(
          t.results[:i + 1],
          t.dts_hours[:i + 1],
          binomial=binomial,
          left=.3,
          right=1.0,
          ah=ah,
          bh=bh,
          ab=ab,
          bb=bb,
          boostRule=boostRule,
          viz=False,
      )
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
