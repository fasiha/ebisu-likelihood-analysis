import numpy as np
from json import dump, load
from cmdstanpy import CmdStanModel


def init(datafile):
    data = {"T": 4, "quiz": [1, 1, 0, 1], "delta": [50, 50, 100, 100]}
    dump(data, open(datafile, 'w'))


datafile = 'data.json'
model = CmdStanModel(stan_file="model.stan")
fit = model.sample(data=datafile, adapt_delta=0.99, iter_sampling=10_000)

fitdf = fit.summary()

print(fit.diagnose())

data = load(open(datafile, 'r'))
hlmedian = fitdf.loc[[c for c in fitdf.index if c.startswith("hl[")],
                     '50%'].values
pRecall = np.exp(-np.array(data['delta']) / hlmedian)