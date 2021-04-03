data {
    int T;
    int<lower=0,upper=1> quiz[T];
    vector[T] delta;
}
parameters {
    real<lower=0> hl;
}
model {
    for (t in 1:T)
        quiz[t] ~ bernoulli(exp(-delta[t] / hl));
}
