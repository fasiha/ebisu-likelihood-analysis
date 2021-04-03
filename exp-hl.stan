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
    vector[T] hl = initHl + exp(learnRate / 1000 * time);
    // alternate: initHl * exp(...)
    // Similar results
}
model {
    learnRate ~ exponential(1.5);
    initHl ~ normal(20, 20);

    for (t in 1:T)
        quiz[t] ~ bernoulli(exp(-delta[t] / hl[t]));
}
