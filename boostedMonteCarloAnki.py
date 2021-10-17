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


def makeHalflives(b, h, ts, left, right):
  hs = [h]
  for t in ts:
    old = hs[-1]
    hs.append(old * clampLerp2(left * old, right * old, min(b, 1.0), b, t))
  return hs


def ankiFitEasyHardMAP(xs: list[int],
                       ts: list[float],
                       binomial,
                       left,
                       right,
                       ah,
                       bh,
                       ab,
                       bb,
                       viz=False):
  from math import fsum

  LOG_HALF = -np.log(0.5)

  def posterior(b, h, extra=False):
    logb = np.log(b)
    logh = np.log(h)
    logprior = -bb * b - bh * h + (ab - 1) * logb + (ah - 1) * logh
    hs = makeHalflives(b, h, ts, left, right)

    if binomial:
      loglik = []
      for x, t, h in zip(xs, ts, hs):
        logp = -t / h * LOG_HALF
        if x == 1:
          loglik.append(np.log(-np.expm1(logp)))
        elif x == 3:
          loglik.append(logp)
        elif x == 2 or x == 4:
          n = 2
          k = 1 if x < 3 else 2
          loglik.append(binomln(n, k) + k * logp + (n - k) * np.log(-np.expm1(logp)))
        else:
          raise Exception('unknown result')
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
    viz = dict(fig=fig, ax=ax, im=im)
  else:
    viz = dict()

  MIN_BOOST = 1.0
  res = opt.shgo(lambda x: -posterior(*x), [(MIN_BOOST, 1 / bb * 5), (0.1, 1 / bh * 5)])
  print(res.message)
  bestb, besth = res.x

  bestloglikelihood = posterior(bestb, besth, True)['loglikelihood']
  summary = []
  halflives = makeHalflives(bestb, besth, ts, left, right)
  for x, t, h in zip(xs, ts, halflives):
    summary.append(f'{x}@{t:0.2f} {"🔥" if x<2 else ""} p={np.exp(-t/h):0.2f}, hl={h:0.1f}')

  return dict(
      viz=viz,
      bestb=bestb,
      besth=besth,
      summary=summary,
      bestloglikelihood=bestloglikelihood,
      halflives=halflives,
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

  if True:
    kws = dict(
        binomial=True,
        left=0.3,
        right=1,
        ah=1.0,
        bh=0.1,
        ab=1.0,
        bb=1.0,
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
    assert np.all(np.diff(finalHls) > 0), 'final halflives should be monotonic'

  if True:
    fracs = [0.7, 0.8, 0.9, 0.95]
    subtrain = [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]
    reses = []
    binomial = False
    right = 1.0
    left = 0.3
    ah = 1.0
    bh = 0.2
    ab = 2.0
    bb = 1.0
    for t in subtrain:
      for binomial in [True]:
        title = f'Card {t.df.cid.iloc[0]}'
        print(
            f'\n## {title},  binomial={binomial}, left={left}, right={right}, ah={ah}, bh={bh}, ab={ab}, bb={bb}'
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
            bb=bb)
        if 'ax' in res['viz']:
          res['viz']['ax'].set_title(title)
        reses.append(res)
        print("\n".join(res['summary']))
        print(
            f'> best h={res["besth"]:0.2f}, b={res["bestb"]:0.2f}, loglik={res["bestloglikelihood"]:0.2f}'
        )

  if False:
    t = next(t for t in train if t.fractionCorrect > 0.9)
    priors = 'gamma'
    clamp = True
    binomial = False
    gridsearch = False

    reses = []
    for i in range(len(t.results)):
      title = f'{i+1}'
      res = ankiFitEasyHardMAP(
          t.results[:i + 1],
          t.dts_hours[:i + 1],
          priors=priors,
          clamp=clamp,
          binomial=binomial,
          left=.8,
          right=1.0,
          bh=0.5,
          bb=1.0,
          gridsearch=gridsearch,
      )
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
# Base 2
## Card 1300038031922,  binomial=True, left=0.3, right=1.0, bh=0.1, bb=1.0, ah=0.1
Optimization terminated successfully.
2@0.02  p=1.00, hl=25.1
2@6.92  p=0.76, hl=25.1
2@24.29  p=0.38, hl=25.1
1@19.37 🔥 p=0.56, hl=33.4
3@21.12  p=0.57, hl=38.1
3@69.23  p=0.20, hl=42.9
3@153.46  p=0.07, hl=57.9
1@265.31 🔥 p=0.03, hl=78.0
3@26.66  p=0.78, hl=105.2
1@446.41 🔥 p=0.01, hl=105.2
2@25.51  p=0.84, hl=141.9
> best h=25.05, b=1.35, loglik=-14.98

## Card 1300038031922,  binomial=True, left=0.3, right=1.0, bh=0.2, bb=1.0, ah=0.2
Optimization terminated successfully.
2@0.02  p=1.00, hl=5.6
2@6.92  p=0.29, hl=5.6
2@24.29  p=0.08, hl=9.5
1@19.37 🔥 p=0.30, hl=15.9
3@21.12  p=0.46, hl=26.8
3@69.23  p=0.17, hl=39.6
3@153.46  p=0.10, hl=66.7
1@265.31 🔥 p=0.09, hl=112.4
3@26.66  p=0.87, hl=189.3
1@446.41 🔥 p=0.09, hl=189.3
2@25.51  p=0.92, hl=318.8
> best h=5.62, b=1.68, loglik=-13.91

# Base e
## Card 1300038031922,  binomial=True, left=0.3, right=1.0, bh=0.1, bb=1.0, ah=0.1
Optimization terminated successfully.
2@0.02  p=1.00, hl=6.9
2@6.92  p=0.37, hl=6.9
2@24.29  p=0.14, hl=12.2
1@19.37 🔥 p=0.41, hl=21.5
3@21.12  p=0.55, hl=35.5
3@69.23  p=0.23, hl=46.9
3@153.46  p=0.16, hl=82.7
1@265.31 🔥 p=0.16, hl=145.6
3@26.66  p=0.90, hl=256.5
1@446.41 🔥 p=0.18, hl=256.5
2@25.51  p=0.95, hl=451.8
> best h=6.92, b=1.76, loglik=-14.44

"""