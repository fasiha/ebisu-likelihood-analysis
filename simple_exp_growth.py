model = r"""
data {
    int T;
    int<lower=0,upper=1> quiz[T];
    vector[T] delta;
    vector[T] time;
}
parameters {
    real<lower=0> learnRate;
    real<lower=0> initHl;
}
transformed parameters {
    vector[T] hl = initHl * exp(learnRate / 1000 * time);
}
model {
    learnRate ~ exponential(1.5);
    initHl ~ normal(20, 20);

    for (t in 1:T)
        quiz[t] ~ bernoulli(exp(-delta[t] / hl[t]));
}
"""

import numpy as np
import stan

initHl = 12  # hours
learnRate = 0.001  # halflife hours per studied hours
N = 50

RNG = np.random.RandomState(123)
t = np.sort(RNG.rand(N) * (365 * 24))  # hours
hl = initHl * np.exp(learnRate * t)

delta = np.hstack([t[0], np.diff(t)])
pRecall = np.exp(-delta / hl)

from scipy.stats import bernoulli
quiz = bernoulli.rvs(pRecall)

data = {"T": N, "time": t, "delta": delta, "quiz": quiz}
posterior = stan.build(model, data=data, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=1000)

import pylab as plt
plt.ion()

fig, axs = plt.subplots(3)
axs[0].plot(t, hl, '.')
axs[1].hist(fit['initHl'].ravel(), 30)
axs[2].hist(fit['learnRate'].ravel(), 30)
