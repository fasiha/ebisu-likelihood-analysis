data {
    int T;
    int<lower=0,upper=1> quiz[T];
    vector[T] delta;
    vector[T] time;
}
parameters {
    real<lower=0> maxval;
    real<lower=0> minval;
    real mid;
    real<lower=0> steepness;
}
transformed parameters {
    vector[T] hl = minval + maxval * inv_logit(steepness * 1e-3 * (time - mid));
}
model {
    maxval ~ normal(time[T], time[T]*10);
    mid ~ normal(time[T] / 2, time[T] * 2);
    minval ~ normal(20, 20);

    steepness ~ normal(6, 20);


    for (t in 1:T)
        quiz[t] ~ bernoulli(exp(-delta[t] / hl[t]));
}
