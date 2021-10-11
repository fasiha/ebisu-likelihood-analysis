
functions {
  real clampLerp(real x1, real x2, real y1, real y2, real x) {
    real mu = (x - x1) / (x2 - x1);
    return x < x1 ? y1 : x > x2 ? y2 : (y1 * (1 - mu) + y2 * mu);
  }
}
data {
  int<lower=0> T;
  array[T] int x;
  array[T] real<lower=0> t;
}
parameters {
  real<lower=0> hl0;
  real<lower=0> boost;
}
transformed parameters {
  array[T] real<lower=0> hl;
  hl[1] = hl0; // halflife for quiz 1
  for (n in 2:T){
    hl[n] = hl[n-1] * clampLerp(0.8 * hl[n-1], hl[n-1], fmin(1.0, boost), boost, t[n-1]);
  }

  array[T] real<lower=0, upper=1> prob;
  for (n in 1:T) {
    prob[n] = exp(-t[n] / hl[n]);
  }
}
model {
  hl0 ~ gamma(10 * 0.25 + 1, 10.0);
  boost ~ gamma(10 * 1.4 + 1, 10.0);

  for (n in 1:T) {
    if (x[n]==1 || x[n]==3) { // fail or pass
      int xx = x[n] > 1;
      xx ~ bernoulli(prob[n]);
    } else { // hard or easy
      int xx = x[n] < 3 ? 1 : 2;
      xx ~ binomial(2, prob[n]);
    }
  }
}
