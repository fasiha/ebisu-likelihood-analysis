
functions {
  real clampLerp(real x1, real x2, real y1, real y2, real x) {
    real mu = (x - x1) / (x2 - x1);
    real y = (y1 * (1 - mu) + y2 * mu);
    return fmin(y2, fmax(y1, y));
  }
  int success(int nSuccess, int nTotal) {
    return 2 * nSuccess >= nTotal;
  }
}
data {
  // quiz history
  int<lower=0> T;
  array[T] int successes;
  array[T] int totals;
  array[T] real<lower=0> t;
  
  // algorithm parameters
  real left;
  real right;
  real<lower=0> alphaHl;
  real<lower=0> betaHl;
  real<lower=0> alphaBoost;
  real<lower=0> betaBoost;
}
parameters {
  real<lower=0> hl0;
  real<lower=0> boost;
}
transformed parameters {
  array[T] real<lower=0> hl;
  hl[1] = hl0; // halflife for quiz 1
  for (n in 2:T) {
    real thisBoost = success(successes[n-1], totals[n-1]) ? clampLerp(left * hl[n-1], right * hl[n-1], 1.0, fmax(boost, 1.0), t[n-1]) : 1.0;
    hl[n] = thisBoost * hl[n-1];
  }

  array[T] real<lower=0, upper=1> prob;
  for (n in 1:T) {
    prob[n] = exp(-t[n] / hl[n] * log2());
  }
}
model {
  hl0 ~ gamma(alphaHl, betaHl);
  boost ~ gamma(alphaBoost, betaBoost);

  for (n in 1:T) {
    successes[n] ~ binomial(totals[n], prob[n]);
  }
}
