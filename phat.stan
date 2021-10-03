  data {
    int<lower=0,upper=1> x1;
    real t1;
    int<lower=0,upper=1> x2;
    real t2;
    real hlhat3; // -t3 / log2(phat3), instead of {x3, t3}
  }
  parameters {
    real<lower=0> hla;
    real<lower=0> hlb;
    
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
    
    // See https://mc-stan.org/docs/2_27/reference-manual/increment-log-prob-section.html
    // hlhat3 ~ hl2; // <-- we WANT to do this but we can't I don't think so:
    target += gamma_lupdf(hlhat3 | hla, hlb);

    hla ~ normal(10 * 0.25 + 1, 1.0);
    hlb ~ normal(10.0, 1.0);
    hl0 ~ gamma(hla, hlb);
    b ~ gamma(10 * 1.4 + 1, 10.0);
  }
