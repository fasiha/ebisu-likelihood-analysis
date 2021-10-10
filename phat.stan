
data {
  int<lower=0,upper=1> x1;
  int<lower=0,upper=1> x2;
  int<lower=0,upper=1> x3;
  int<lower=0> k1;
  int<lower=0> k2;
  int<lower=0> k3;
  real t1;
  real t2;
  real t3;
}
parameters {
  real<lower=0> initialHalflife;
  real<lower=0> boost;
}
transformed parameters {
  real<lower=0> hl1 = initialHalflife * boost;
  real<lower=0> hl2 = hl1 * boost;
}
model {
  initialHalflife ~ gamma(10 * 0.25 + 1, 10.0);

  boost ~ gamma(10 * 1.4 + 1, 10.0);

  k1 ~ binomial(10, exp(-t1 / initialHalflife));
  k2 ~ binomial(10, exp(-t2 / hl1));
  k3 ~ binomial(10, exp(-t3 / hl2));
}
