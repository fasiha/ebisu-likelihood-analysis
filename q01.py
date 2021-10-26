import ebisu

hl = 1.0
tnow = 100.0
new = ebisu.updateRecall((3.0, 3.0, hl), 1, 1, tnow, q0=0)
newQ0 = ebisu.updateRecall((3.0, 3.0, hl), 1, 1, tnow, q0=1e-2)
print(new)
print(newQ0)

import numpy as np
import pandas as pd  # type:ignore
from cmdstanpy import CmdStanModel  # type:ignore
import json

fits = []
for q0, t2 in zip([0.0, 0.01], [model[2] for model in [new, newQ0]]):
  data = dict(t0=1.0, alpha=3.0, beta=3.0, z=1, q1=1.0, q0=q0, t=100.0, t2=t2)
  with open('ebisu_data.json', 'w') as fid:
    json.dump(data, fid)
  model = CmdStanModel(stan_file="ebisu.stan")
  fit = model.sample(
      data='ebisu_data.json',
      chains=2,
      iter_warmup=10_000,
      iter_sampling=100_000,
  )
  fits.append(fit)
  print(fit.diagnose())

fitdfs = [
    pd.DataFrame({
        k: v.ravel()
        for k, v in fit.stan_variables().items()
        if 1 == len([s
                     for s in v.shape if s > 1])
    })
    for fit in fits
]


def _meanVarToBeta(mean, var) -> tuple[float, float]:
  """Fit a Beta distribution to a mean and variance."""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


alphabetas = [_meanVarToBeta(np.mean(fitdf.p2), np.var(fitdf.p2)) for fitdf in fitdfs]
print(alphabetas)

import numpy as np
import pylab as plt

plt.ion()
plt.style.use('ggplot')

tnows = np.logspace(0, 2)  # 1.0 to 100
q0ToNewHalflife = lambda q0: [
    ebisu.modelToPercentileDecay(ebisu.updateRecall((3.0, 3.0, hl), 1, 1, tnow, q0=q0))
    for tnow in tnows
]

plt.figure()
plt.plot(tnows, q0ToNewHalflife(1e-2), label='q0=1e-2')
plt.plot(tnows, q0ToNewHalflife(1e-3), linestyle='--', label='q0=1e-3')
plt.xlabel('tnow')
plt.ylabel('halflife after update')
plt.title('Behavior of update for q0')
axis = plt.axis()
plt.plot(tnows, q0ToNewHalflife(0), linestyle=':', label='q0=0')
plt.axis(axis)
plt.legend()
plt.savefig('q01.png', dpi=300)
plt.savefig('q01.svg')
