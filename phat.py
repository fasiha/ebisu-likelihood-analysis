from cmdstanpy import CmdStanModel  #type:ignore
import json
import numpy as np

if __name__ == '__main__':
  # write data to disk
  data = {
      "x1": 1,
      "x2": 1,
      "x3": 0,
      "t1": 0.8,
      "t2": 2.2,
      "t3": 2.9,
      "p1": 0.5,
      "p2": 0.6,
      "p3": 0.1,
  }
  for i in range(1, 3 + 1):
    data[f'k{i}'] = round(data[f'p{i}'] * 10)

  with open('modeldata.json', 'w') as fid:
    json.dump(data, fid)

  model = CmdStanModel(stan_file="question_nohyper.stan")
  fit = model.sample(data="modeldata.json", chains=2, iter_sampling=100_000)

  import pandas as pd  #type:ignore

  fitdf = pd.DataFrame({
      k: v.ravel()
      for k, v in fit.stan_variables().items()
      if 1 == len([s for s in v.shape if s > 1])
  })
  # pd.plotting.scatter_matrix(fitdf.sample(1_000))
  print(f'phat3={data["p3"]} MEAN\n{fitdf.mean()}')

  viz = False
  if viz:
    import numpy as np
    from scipy.stats import gamma as gammarv
    import pylab as plt
    plt.ion()

    plt.figure()
    ax = fitdf.initialHalflife.hist(bins=50, density=True, label='Stan')
    x = np.linspace(*ax.get_xlim(), 1001)
    gfit = gammarv.fit(fitdf.initialHalflife, floc=0)
    ax.plot(x, gammarv.pdf(x, gfit[0], scale=gfit[2]), label='Fit')
    ax.legend()

    plt.figure()
    ax = fitdf.boost.hist(bins=50, density=True, label='Stan')
    x = np.linspace(*ax.get_xlim(), 1001)
    gfit = gammarv.fit(fitdf.boost, floc=0)
    ax.plot(x, gammarv.pdf(x, gfit[0], scale=gfit[2]), label='Fit')
    ax.legend()
"""
HYPERRRRR::

NO P3 MEAN
halflifeAlpha      3.762734
halflifeBeta       9.385637
initialHalflife    1.021932
boost              1.986307
hl1                1.999876
hl2                4.073356

phat3=0.1 MEAN
halflifeAlpha      3.753493
halflifeBeta       9.433539
initialHalflife    0.968110
boost              1.596542
hl1                1.512940
hl2                2.441117

phat3=0.9 MEAN
halflifeAlpha      3.780522
halflifeBeta       9.299784
initialHalflife    1.112161
boost              2.456105
hl1                2.689962
hl2                6.674616


NO HYPER:

no P3 MEAN
initialHalflife    0.975524
boost              2.012956
hl1                1.936751
hl2                3.997737

phat3=0.1 MEAN
initialHalflife    0.928966
boost              1.625546
hl1                1.480586
hl2                2.433432

phat3=0.9 MEAN
initialHalflife    1.054815
boost              2.493186
hl1                2.593793
hl2                6.536977
"""
