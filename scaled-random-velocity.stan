data {
    int T;
    int<lower=0,upper=1> quiz[T];
    vector[T] delta;
}
parameters {
    real<lower=0> initHl;

    vector[T] process;
    // real<lower=0> sigmaprocess;
}
transformed parameters {
    real<lower=0> hl[T];
    hl[1] = 50 * initHl;
    for (t in 2:T)
        hl[t] = hl[t-1] * exp2(process[t] * delta[t]/hl[t-1]);
}
model {
    for (t in 1:T)
        quiz[t] ~ bernoulli(exp(-delta[t] / (hl[t])));
    
    // sigmaprocess ~ normal(1, 1);
    process ~ normal(0, .1);
    initHl ~ normal(1, 1);
}
