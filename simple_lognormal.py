model = r"""
parameters {
    real<lower=0> x;
    real<lower=0> y;
    real<lower=0> z;
}
model {
  x ~ lognormal(log(2), 1.6);
  y ~ lognormal(log(2), 0.25);
  z ~ exponential(4);
}
"""

import stan

posterior = stan.build(model, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=10_000)

import numpy as np
import pandas as pd
import pylab as plt
plt.ion()

fitdf = fit.to_frame()
fitdf = fitdf[[c for c in fitdf.columns if not c.endswith("_")]]

pd.plotting.scatter_matrix(fitdf)