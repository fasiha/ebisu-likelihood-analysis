import pylab as plt
plt.ion()

model = r"""
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real<lower=0> maxval;
    real minval;
    real mid;
    real<lower=0> steepness;
    real<lower=0> sigma;
}
model {
  y ~ normal(minval + (maxval - minval) * inv_logit(steepness * 1e-3 * (x - mid)), sigma);
  steepness ~ normal(6, 20);
  sigma ~ normal(0, 10);
}
"""

import stan
import numpy as np
from scipy.special import expit
minval = -2
maxval = 3.5
mid = 0.3 * 365 * 24
# derivative (slope) at midpoint: steepness / 4
steepness = 5 / (0.5 * 365 * 24)
sigma = 0.1 / 5

x = np.linspace(0, 2 * 365 * 24)
y = minval + (maxval - minval) * expit(
    steepness * (x - mid)) + np.random.randn(len(x)) * sigma

plt.plot(x, y)

data = {"N": len(x), "x": x, "y": y}
posterior = stan.build(model, data=data, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=1_000)

fitdf = fit.to_frame()
fitdf = fitdf[[c for c in fitdf.columns if not c.endswith("_")]]

import pandas as pd
pd.plotting.scatter_matrix(fitdf)
