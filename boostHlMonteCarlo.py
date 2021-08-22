from dataclasses import dataclass
from scipy.stats import gamma as gammarv  #type:ignore
import numpy as np
import matplotlib.pylab as plt  #type:ignore
from scipy.stats import multinomial

plt.ion()


@dataclass
class Review:
  result: bool
  studyGap: float


def ebisuPosteriorGammaFit(samplesHalflife: np.ndarray, review: Review) -> tuple[float, float]:
  pRecall = 2.0**(-review.studyGap / samplesHalflife)
  weights = pRecall if review.result else (1 - pRecall)
  sumWeights = np.sum(weights)
  weightedMean = np.sum(weights * pRecall) / sumWeights
  weightedVar = np.sum(weights * (pRecall - weightedMean)**2) / sumWeights
  # https://www.wolframalpha.com/input/?i=solve+m%3Da%2Fb%2C+v%3Da%2Fb%5E2+for+a%2C+b
  newA = weightedMean**2 / weightedVar
  newB = weightedMean / weightedVar
  return (newA, newB)


def ebisuPosteriorSamples(samplesHalflife: np.ndarray, review: Review) -> np.ndarray:
  pRecall = 2.0**(-review.studyGap / samplesHalflife)
  weights = pRecall if review.result else (1 - pRecall)
  return sequentialImportanceResample(samplesHalflife, weights)[0]


def sequentialImportanceResample(particles: np.ndarray,
                                 weights: np.ndarray,
                                 N=None) -> tuple[np.ndarray, np.ndarray]:
  if N is None:
    N = len(particles)
  draw: np.ndarray = multinomial(N, weights / np.sum(weights))
  # each element of `draw` is an integer, the number of times the particle at that index should appear in the output

  # this isn't going to be fast FIXME
  newParticles = np.hstack(
      [np.ones(repeat) * particle for repeat, particle in zip(draw, particles)])
  newWeights = np.ones(N)
  return (newParticles, newWeights)


if __name__ == '__main__':
  modeBoost = 1.4
  bBoost = 10.0

  modeHl = 0.25  # hours
  bHl = 10.0

  def generateRandomGamma(mode: float, beta: float) -> np.ndarray:
    return gammarv.rvs(beta * mode + 1, scale=1 / beta, size=100_000)

  b = generateRandomGamma(modeBoost, bBoost)
  hl = generateRandomGamma(modeHl, bHl)

  if False:
    plt.figure()
    plt.hist(b, bins=50, alpha=0.5, density=True, label='boost')
    plt.hist(hl, bins=50, alpha=0.5, density=True, label='hl')

    plt.legend()
    plt.grid()

  reviews = [Review(result=True, studyGap=0.5), Review(result=False, studyGap=1.5)]

  from cmdstanpy import CmdStanModel  # type: ignore
  import pandas as pd  #type: ignore
  import json

  with open('modeldata.json', 'w') as fid:
    json.dump(
        {
            "x1": int(reviews[0].result),
            "x2": int(reviews[1].result),
            "t1": reviews[0].studyGap,
            "t2": reviews[1].studyGap
        }, fid)
  modelCode = """
  data {
    int<lower=0,upper=1> x1;
    real t1;
    int<lower=0,upper=1> x2;
    real t2;
  }
  parameters {
    real<lower=0> hl0;
    real<lower=0> b;
  }
  transformed parameters {
    real<lower=0> hl1 = hl0 * b;
    real<lower=0> hl2 = hl1 * b;
  }
  model {
    x1 ~ bernoulli(exp(-t1 / hl0));
    x2 ~ bernoulli(exp(-t2 / hl1));
    hl0 ~ gamma(10 * 0.25 + 1, 10.0);
    b ~ gamma(10 * 1.4 + 1, 10.0);
  }
  """
  with open('model.stan', 'w') as fid:
    fid.write(modelCode)
  model = CmdStanModel(stan_file="model.stan")
  fit = model.sample(data="modeldata.json", chains=2, iter_sampling=100_000)
  summarydf = fit.summary()

  print(fit.diagnose())
  fitdf = pd.DataFrame({
      k: v.ravel()
      for k, v in fit.stan_variables().items()
      if 1 == len([s for s in v.shape if s > 1])
  })
  pd.plotting.scatter_matrix(fitdf)

  if True:
    aFit, _, bFitRecip = gammarv.fit(fitdf.hl2, floc=0)
    plt.figure()
    pdf, bins, _ = plt.hist(fitdf.hl2, bins=100, alpha=0.5, density=True, label='empirical')
    xpdf = np.linspace(bins[0], bins[-1], num=5000)
    plt.plot(xpdf, gammarv.pdf(xpdf, aFit, scale=bFitRecip), label='fit')
    plt.title("Posterior isn't really Gamma is it?")
    plt.legend()
    plt.grid()
"""
np.median(fitdf.values,axis=0)
Out[114]: array([0.5838555, 1.637895 , 0.9425555, 1.5424   ])
Out[117]: array([0.4245475, 1.44468  , 0.608157 , 0.880836 ])
"""