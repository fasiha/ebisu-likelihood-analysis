data {
    int T;
    int<lower=0,upper=1> quiz[T];
    vector[T] delta;
}
parameters {
    real<lower=0> initHl;
    // real scalemean;
    real<lower=0> scalesigma;

    real<lower=0> scale[T];
}
transformed parameters {
    real<lower=0> hl[T];
    real<lower=0> hl0 = 50 * initHl;
    hl[1] = hl0 * scale[1];
    for (t in 2:T)
        hl[t] = hl[t-1] * scale[t];
    real mu[T];
    real scalemean = log(2);
    mu[1] = scalemean * delta[1] / hl0;
    for (t in 2:T)
        mu[t] = scalemean * delta[t] / hl[t-1];
}
model {
    for (t in 1:T)
        quiz[t] ~ bernoulli(exp(-delta[t] / (hl[t])));

    scale ~ lognormal(mu, scalesigma);

    // scalemean ~ normal(log(2), 1);
    initHl ~ gamma(2, 0.5);
    scalesigma ~ exponential(4);
}
