import numpy as np
from json import dump, load
from cmdstanpy import CmdStanModel


def init(datafile, T):
    delta = [20., 20. * 2, 20. * 4, 20. * 8]
    quiz = [0, 1, 0, 1]
    assert T <= len(delta)
    data = {
        "T": T,
        "quiz": quiz[:T],
        "delta": delta[:T],
        "time": np.cumsum(np.hstack([[0], delta[:T - 1]])).tolist()
    }
    dump(data, open(datafile, 'w'))


datafile = 'data.json'
# init(datafile, 4)

model = CmdStanModel(stan_file="model.stan")
fit = model.sample(data=datafile,
                   chains=2,
                   adapt_delta=0.9999,
                   iter_sampling=10_000)

summarydf = fit.summary()

print(fit.diagnose())

data = load(open(datafile, 'r'))
# hlmedian = summarydf.loc[[c for c in summarydf.index if c.startswith("hl[")],
#                          '50%'].values
# pRecall = np.exp(-np.array(data['delta']) / hlmedian)

import pandas as pd
import pylab as plt

plt.ion()

fitdf = pd.DataFrame({
    k: v.ravel()
    for k, v in fit.stan_variables().items()
    if 1 == len([s for s in v.shape if s > 1])
})
pd.plotting.scatter_matrix(fitdf)

from scipy.stats import lognorm

pdf = lambda x, mu, s: lognorm.pdf(x, s=s, scale=np.exp(mu)) / np.exp(mu)


def _meanVarToBeta(mean, var):
    """Fit a Beta distribution to a mean and variance."""
    # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
    tmp = mean * (1 - mean) / var - 1
    alpha = mean * tmp
    beta = (1 - mean) * tmp
    return alpha, beta


hlmean = np.median(fit.stan_variables()['hl'], axis=0)
n = np.arange(len(hlmean))
t = np.cumsum(data['delta'])  # different than data['time]!
ok = np.array(data['quiz'], dtype=bool)
notok = np.logical_not(ok)
plt.figure()
plt.plot(n, hlmean)
plt.plot(n[ok], hlmean[ok], 'bo')
plt.plot(n[notok], hlmean[notok], 'rx')
