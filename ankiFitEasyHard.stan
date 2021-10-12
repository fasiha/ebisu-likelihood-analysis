
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
  // real<lower=0, upper=1> clampLeft;
  // real<lower=0> clampWidth;
}
transformed parameters {
  array[T] real<lower=0> hl;
  hl[1] = hl0; // halflife for quiz 1
  for (n in 2:T){
    hl[n] = hl[n-1] * clampLerp(0.8 * hl[n-1], (0.8 + 0.2) * hl[n-1], fmin(1.0, boost), boost, t[n-1]);
  }

  array[T] real<lower=0, upper=1> prob;
  for (n in 1:T) {
    prob[n] = exp(-t[n] / hl[n]);
  }
}
model {
  // clampLeft ~ beta(1, 1);
  // clampWidth ~ exponential(0.5);
  hl0 ~ gamma(10 * 0.25 + 1, 10.0);
  boost ~ gamma(10 * 1.4 + 1, 10.0);

  for (n in 1:T) {
    if (x[n]<=4) { // ALL
    // if (x[n]==1 || x[n]==3) { // fail or pass
      int xx = x[n] > 1;
      xx ~ bernoulli(prob[n]);
    } else { // hard or easy
      int xx = 2;
      xx ~ binomial(x[n] < 3 ? 3 : 2, prob[n]);
    }
  }
}
