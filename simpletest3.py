"""
%load_ext autoreload
%autoreload 2
"""
import pylab as plt  # type: ignore

plt.ion()

from itertools import product
import ebisu3 as ebisu
import numpy as np
from copy import deepcopy
from scipy.stats import gamma as gammarv  # type: ignore


def relativeError(actual: float, expected: float) -> float:
  e, a = np.array(expected), np.array(actual)
  return np.abs(a - e) / np.abs(e)


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


def gammaToStats(a: float, b: float):
  mean = gammaToMean(a, b)
  var = gammaToVar(a, b)
  k = a
  t = 1 / b
  m2 = t**2 * k * (k + 1)  # second non-central moment
  return (mean, var, m2, np.sqrt(m2))


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
for fraction, result, lastNoisy in product([.1], [0], [False]):
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

seed = np.random.randint(1, 1_000_000_000)
# seed = 498076874 # causes mixture to fail initial fit…?
# seed = 29907812  #np.random.randint(1, 100_000_000)
# seed = 708572856  # fails?
# seed = 1234
print(f'{seed=}')
np.random.seed(seed=seed)  # for sanity when testing with Monte Carlo

fulls = []
for i in range(1):
  print(i)
  tmp: tuple[ebisu.Model, dict] = ebisu._updateRecallHistoryDebug(upd)
  tmp = (tmp[0], ebisu._enrichDebug(tmp[1]))
  print(f"{tmp[1]['kish']=}")
  bMom, hMom = [(stats[0]**2 / stats[1], stats[0] / stats[1])
                for stats in [tmp[1]['bEbisuSamplesStats'], tmp[1]['hEbisuSamplesStats']]]
  tmp[1]['bMom'] = bMom
  tmp[1]['hMom'] = hMom
  w = np.exp(tmp[1]['betterFit']['logw'])
  bMl = ebisu._weightedGammaEstimateMaxLik(tmp[1]['betterFit']['xs'], w)
  hMl = ebisu._weightedGammaEstimateMaxLik(tmp[1]['betterFit']['ys'], w)
  tmp[1]['bMl'] = bMl
  tmp[1]['hMl'] = hMl
  fulls.append(tmp)

fullToMean = lambda full: (gammaToMean(*full.prob.initHl), gammaToMean(*full.prob.boost))
means = [fullToMean(x[0]) for x in fulls]
print('mean', np.mean(means, axis=0))
print('std', np.std(means, axis=0))


def set_clim(dynamicRange: int | float, gci=None):
  top = (gci or plt.gci()).get_clim()[1]
  plt.clim((top - abs(dynamicRange), top))


"""
MIXTURE=false
mean [6.48339345 1.56056942]
std [0.11329404 0.02260367]

MIXTURE=True, unifWeight=1
weightPower=0.0
mean [6.49701033 1.55617411]
std [0.1301719  0.02569452]

WITH SHUFFLE

MIXTURE=True, unifWeight=0.1
weightPower=2
mean [6.48205908 1.56094228]
std [0.04942927 0.00702538]

MIXTURE=True, unifWeight=0.25
weightPower=2
mean [6.49051695 1.56257748]
std [0.05099051 0.00700811]

MIXTURE=True, unifWeight=0.5
weightPower=2
mean [6.49567787 1.56198629]
std [0.05496866 0.00890928]


MIXTURE=True, unifWeight=0.75
weightPower=2
mean [6.48921628 1.56269426]
std [0.07472302 0.01381985]


"""
# for size in [1e3, 10e3, 100e3, 1000e3]:
#   full, debug = ebisu.updateRecallHistory(upd, size=int(size), debug=True)
#   print(fullToMean(full))

import pickle
with open('finalres.pickle', 'rb') as fid:
  data = pickle.load(fid)


def recon(bs, hs, logps, w=0):
  fit = ebisu._fitJointToTwoGammas(bs, hs, logps, w)
  return (gammaToMean(fit['alphax'], fit['betax']), gammaToMean(fit['alphay'], fit['betay']))


import pylab as plt

plt.ion()

x = np.linspace(0, 25, 1001)
plt.figure()
plt.plot(
    x,
    gammarv.pdf(x, fulls[0][0].prob.initHl[0], scale=1 / fulls[0][0].prob.initHl[1]),
    label='mixed mode')
plt.plot(x, gammarv.pdf(x, tmp[1]['hMl'][0], scale=1 / tmp[1]['hMl'][1]), label='max lik')
plt.plot(x, gammarv.pdf(x, tmp[1]['hMom'][0], scale=1 / tmp[1]['hMom'][1]), label='mom')
plt.legend()
plt.ylim([0, .12])
if False:
  better = fulls[0][1]["betterFit"]

  plt.figure()
  plt.scatter(better["xs"], better["ys"], c=better["logw"])
  plt.colorbar()
  set_clim(10)

  plt.figure()
  plt.scatter(better["xs"][:5000], better["ys"][:5000], c=better["logw"][:5000])
  plt.colorbar()
  set_clim(10)

  plt.figure()
  plt.scatter(better["xs"][5000:], better["ys"][5000:], c=better["logw"][5000:])
  plt.colorbar()
  set_clim(10)
