data {
  real<lower=0> t0;
  real<lower=0> alpha;
  real<lower=0> beta;
  int<lower=0, upper=1> z;
  real<lower=0, upper=1> q1;
  real<lower=0, upper=1> q0;
  real<lower=0> t;
  real<lower=0> t2;
}
parameters {
  real<lower=0, upper=1> p0;

  // We WANT this:
  // `int<lower=0, upper=1> x;`
  // But we can't have it: https://mc-stan.org/docs/2_28/stan-users-guide/change-point.html
  // So we marginalize over x.
}
transformed parameters {
  real<lower=0, upper=1> p = pow(p0, t / t0); // Precall at t
  real<lower=0, upper=1> p2 = pow(p, t2 / t); // Precall at t2
}
model {
  p0 ~ beta(alpha, beta); // Precall at t0

  // Again, we WANT the following:
  // `x ~ bernoulli(p);`
  // `z ~ bernoulli(x ? q1 : q0);`
  // But we can't so we had to marginalize:
  target += log_mix(p, bernoulli_lpmf(z | q1), bernoulli_lpmf(z | q0));
  // log_mix is VERY handy: https://mc-stan.org/docs/2_28/functions-reference/composed-functions.html
}
