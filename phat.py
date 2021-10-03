from dataclasses import dataclass
from scipy.stats import gamma as gammarv  #type:ignore
import numpy as np
import matplotlib.pylab as plt  #type:ignore

plt.ion()

if __name__ == '__main__':
  from cmdstanpy import CmdStanModel  # type: ignore
  import pandas as pd  #type: ignore
  import json

  t3 = 1.5
  phat3 = 0.5
  hlhat3 = -t3 / np.log(phat3)

  with open('modeldata.json', 'w') as fid:
    json.dump({
        "x1": int(1),
        "x2": int(1),
        "t1": 0.3,
        "t2": 0.9,
        "hlhat3": hlhat3,
    }, fid)
  model = CmdStanModel(stan_file="phat.stan")
  fit = model.sample(data="modeldata.json", chains=2, iter_sampling=100_000)
  summarydf = fit.summary()

  print(fit.diagnose())
  fitdf = pd.DataFrame({
      k: v.ravel()
      for k, v in fit.stan_variables().items()
      if 1 == len([s for s in v.shape if s > 1])
  })
  # pd.plotting.scatter_matrix(fitdf.sample(1_000))
  print(f'phat3={phat3} MEDIAN\n{fitdf.median()}')
"""
phat3=0.1 MEDIAN
hla    4.378565
hlb    9.634560
hl0    0.592135
b      1.577485
hl1    0.919056
hl2    1.439010

phat3=0.5 MEDIAN
hla    5.135365
hlb    8.297425
hl0    0.741643
b      1.555440
hl1    1.134205
hl2    1.749020

phat3=0.9 MEDIAN
hla     4.716935
hlb     0.969299
hl0     4.763235 <<< wow, what a big jump!
b       1.484430
hl1     6.988150
hl2    10.318650
"""