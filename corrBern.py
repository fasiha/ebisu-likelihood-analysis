import matplotlib.pylab as plt  #type:ignore
from scipy.stats import bernoulli, beta as betarv, uniform  #type:ignore
from utils import sequentialImportanceResample, weightedMeanVar, meanVarToBeta
import numpy as np

alphaX = 2.0
betaX = 3.0
alphaY = 4.0
betaY = 2.0

n = 1_000_000

x = betarv.rvs(alphaX, betaX, size=n)
y = betarv.rvs(alphaY, betaY, size=n)
rho = uniform.rvs(size=n) * 2 - 1  # -1 to 1

xbin = [1, 0, 1, 0, 0]
ybin = [1, 1, 1, 0, 1]

# From https://stats.stackexchange.com/a/285008/
a = (1 - x) * (1 - y) + rho * np.sqrt(x * y * (1 - x) * (1 - x))
square = np.array([a, 1 - y - a, 1 - x - a, a + x + y - 1])  # 4 x n, also from above
valid = np.min(square, axis=0) >= 0

wx = np.ones_like(x)  # just x by itself
weight = np.ones_like(x)  # x, y together

weight[np.logical_not(valid)] = 0  # invalid has zero weight
idxs = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
for thisx, thisy in zip(xbin, ybin):
  idx = idxs[(thisx, thisy)]
  p = square[idx, valid]
  weight[valid] *= p
  wx *= (x**thisx) * (1 - x)**(1 - thisx)  # Bernoulli pdf

rhosir, _ = sequentialImportanceResample(rho, weight)
xpost = meanVarToBeta(*weightedMeanVar(weight, x))  # (4.6, 6.1)
x0post = meanVarToBeta(*weightedMeanVar(wx, x))  # (4, 6), i.e., standard conjugate update

plt.ion()

plt.figure()
plt.hist(rhosir, bins=100, density=True, alpha=0.5)