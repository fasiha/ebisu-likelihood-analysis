from cmdstanpy import CmdStanModel
from json import dump


def init(datafile):
    data = {"T": 4, "quiz": [1, 1, 0, 1], "delta": [50, 50, 100, 100]}
    dump(data, open(datafile, 'w'))


datafile = 'data.json'
model = CmdStanModel(stan_file="model.stan")
fit = model.sample(data=datafile, adapt_delta=0.9, iter_sampling=10_000)

print(fit.summary())

print(fit.diagnose())
fit.stan_variables()['hl']
