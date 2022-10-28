"""
%load_ext autoreload
%autoreload 2
"""
import sys
import pylab as plt  # type: ignore

plt.ion()

from itertools import product
import ebisu3 as ebisu
import numpy as np
from copy import deepcopy
from scipy.stats import gamma as gammarv  # type: ignore


def re(actual: float, expected: float) -> float:
  e, a = np.array(expected), np.array(actual)
  return np.abs(a - e) / np.abs(e)


MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
weightedGammaEstimate = ebisu._weightedGammaEstimate
weightedMeanVarLogw = ebisu._weightedMeanVarLogw

initHlMean = 10  # hours
initHlBeta = 0.1
initHlPrior = (initHlBeta * initHlMean, initHlBeta)

boostMean = 1.5
boostBeta = 3.0
boostPrior = (boostBeta * boostMean, boostBeta)

now = ebisu._timeMs()
init = ebisu.initModel(initHlPrior=initHlPrior, boostPrior=boostPrior, now=now)

left = 0.3
# simulate a variety of 4-quiz trajectories:
# for fraction, result, lastNoisy in product([0.1, 0.5, 1.5, 9.5], [0, 1], [False, True]):
# for fraction, result, lastNoisy in product([.1], [0], [False]):
for fraction, result, lastNoisy in product([9.5], [0], [False]):
  upd = deepcopy(init)
  elapsedHours = fraction * initHlMean
  thisNow = now + elapsedHours * MILLISECONDS_PER_HOUR
  upd = ebisu.updateRecall(upd, result, now=thisNow)

  for nextResult, nextElapsed, nextTotal in zip(
      [1, 1, 1 if not lastNoisy else (0.2)],
      [elapsedHours * 3, elapsedHours * 5, elapsedHours * 7],
      [1, 2, 2 if not lastNoisy else 1],
  ):
    thisNow += nextElapsed * MILLISECONDS_PER_HOUR
    upd = ebisu.updateRecall(upd, nextResult, total=nextTotal, q0=0.05, now=thisNow)

  ### Full Ebisu update (max-likelihood to enhanced Monte Carlo proposal)
  # This many samples is probably WAY TOO MANY for practical purposes but
  # here I want to ascertain that this approach is correct as you crank up
  # the number of samples. If we have confidence that this estimator behaves
  # correctly, we can in practice use 1_000 or 10_000 samples and accept a
  # less accurate model but remain confident that the *means* of this posterior
  # are accurate.

  seed = np.random.randint(1, 1_000_000_000)
  # seed = 498076874 # causes mixture to fail initial fit…?
  # seed = 29907812  #np.random.randint(1, 100_000_000)
  # seed = 708572856  # fails?
  print(f'{seed=}')
  np.random.seed(seed=seed)  # for sanity when testing with Monte Carlo

  tmp: tuple[ebisu.Model, dict] = ebisu.updateRecallHistory(
      upd, left=left, size=1_000_000, debug=True)
  full, fullDebug = tmp
  bEbisuSamplesStats, hEbisuSamplesStats = fullDebug['stats']
  print('kish', fullDebug['kish'])
  # assert fullDebug['kish'] > 0.7

bstats, hstats = fullDebug["betterFit"]["stats"]
boost = dict(mean=bstats[0], m2=bstats[2])
inith = dict(mean=hstats[0], m2=hstats[2])

integralResults = {
    (.1, 0, False):
        dict(
            boost=dict(mean=1.56148031637861, m2=2.94182921460451),
            inith=dict(mean=6.48573020826124, m2=63.1383473633432))
}

intRes = integralResults.get((fraction, result, lastNoisy), None)
if intRes:
  boostInt = intRes['boost']
  inithInt = intRes['inith']
  print(
      f"b errs: mean={re(boost['mean'], boostInt['mean']):0.4g}, m2={re(boost['m2'], boostInt['m2']):0.4g}"
  )
  print(
      f"h errs: mean={re(inith['mean'], inithInt['mean']):0.4g}, m2={re(inith['m2'], inithInt['m2']):0.4g}"
  )

logw = fullDebug['betterFit']['logw']
ns = list(range(0, logw.size + 1, round(logw.size / 250)))[1:]
# ns = list(range(675000, 675125))
cumKish = [ebisu._kishLog(logw[:n]) for n in ns]

fig, ax = plt.subplots(2)
ax[0].plot(ns, cumKish, '.-')
ax[1].plot(logw)

w = np.exp(logw)
xs = fullDebug['betterFit']['xs']
ys = fullDebug['betterFit']['ys']


def _weightedGammaEstimateFAST(h, w):
  """
  See https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1067698046#Closed-form_estimators
  """
  wsum = np.sum(w)
  whsum = np.sum(w * h)
  t = np.sum(w * h * np.log(h)) / wsum - whsum / wsum * np.sum(w * np.log(h)) / wsum
  k = whsum / wsum / t
  return (k, 1 / t)


# cumXfit = [(_weightedGammaEstimateFAST(xs[:n], w[:n])) for n in ns]
# cumYfit = [(_weightedGammaEstimateFAST(ys[:n], w[:n])) for n in ns]

# fig2, ax2 = plt.subplots(4)
# ax2[0].plot(ns, [ebisu._gammaToMean(*res) for res in cumXfit])
# ax2[1].plot(ns, [ebisu._gammaToMean(*res) for res in cumYfit])
# ax2[2].plot(ns, [ebisu.gammaToVar(*res) for res in cumXfit])
# ax2[3].plot(ns, [ebisu.gammaToVar(*res) for res in cumYfit])

# fig3, ax3 = plt.subplots(tight_layout=True)
# hist = ax3.hist2d(xs, ys, 50)

# n = 675105
# ax3.plot(xs[n], ys[n], 'ro')

posterior2d = lambda b, h: ebisu._posterior(b, h, upd, 0.3, 1.0)
f = np.vectorize(posterior2d, [float])

# fig5, ax5 = plt.subplots(tight_layout=True)
# sc5 = ax5.scatter(xs, ys, c=logw, s=1, vmin=-10, vmax=2)
# # sc5 = ax5.scatter(xs[::10], ys[::10], c=logw[::10], s=1, vmin=-10, vmax=2)
# cm = plt.cm.get_cmap('RdYlBu')
# plt.colorbar(sc5)


def im(xv, yv, z, ax=plt):
  assert (yv.size, xv.size) == z.shape

  def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]

  return ax.imshow(
      z, aspect='auto', interpolation='none', extent=extents(xv) + extents(yv), origin='lower')


def set_clim(dynamicRange: int | float, gci):
  top = (gci).get_clim()[1]
  gci.set_clim((top - abs(dynamicRange), top))


def viz(m: ebisu.Model):
  bv = np.linspace(.1, 10, 150)
  hv = np.linspace(.1, 200, 250)
  bs, hs = np.meshgrid(bv, hv)
  post = np.vectorize(lambda b, h: ebisu._posterior(b, h, m, 0.3, 1.0), [float])(bs, hs)
  if fullDebug['origfit']:
    logprior = (
        gammarv.logpdf(bs, fullDebug['origfit']['alphax'],
                       scale=1 / fullDebug['origfit']['betax']) +
        gammarv.logpdf(hs, fullDebug['origfit']['alphay'], scale=1 / fullDebug['origfit']['betay']))
  else:
    logprior = (
        gammarv.logpdf(bs, full.prob.boostPrior[0], scale=1 / full.prob.boostPrior[1]) +
        gammarv.logpdf(hs, full.prob.initHlPrior[0], scale=1 / full.prob.initHlPrior[1]))

  fig, axs = plt.subplots(3, tight_layout=True)

  obj = im(bv, hv, post, axs[0])
  axs[0].set_title('log post')
  set_clim(10, obj)

  obj2 = im(bv, hv, logprior, axs[1])
  axs[1].set_title('log post gamma fit' if fullDebug['origfit'] else 'prior')
  set_clim(10, obj2)

  obj3 = im(bv, hv, post - logprior, axs[2])
  set_clim(20, obj3)

  return dict(objs=[obj, obj2, obj3], bv=bv, hv=hv, post=post, fig=fig, axs=axs)


v = viz(upd)
v["axs"][0].plot(fullDebug["bs"], fullDebug["hs"], 'r.', markersize=1)
v["axs"][0].plot(
    fullDebug["betterFit"]['xs'][::250], fullDebug["betterFit"]['ys'][::250], 'b.', markersize=1)

# plt.figure()
# plt.scatter(fullDebug["bs"], fullDebug["hs"], s=1, c=fullDebug['posteriors'])

f6, a6 = plt.subplots(2)
a6[0].plot(ns, [np.sum(xs[:n] * w[:n]) / np.sum(w[:n]) for n in ns])
a6[1].plot(ns, [np.sum(ys[:n] * w[:n]) / np.sum(w[:n]) for n in ns])

if intRes:
  a6[0].hlines(boostInt['mean'], ns[0], ns[-1], colors=['r'], linestyles='dotted')
  a6[1].hlines(inithInt['mean'], ns[0], ns[-1], colors=['r'], linestyles='dotted')