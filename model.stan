data {
    int T;
    int<lower=0,upper=1> quiz[T];
    vector[T] delta;
}
parameters {
    vector[T] loghl;
    vector[T] hldot;
    real<lower=0> sigmaLoghl;
    real<lower=0> sigmaHldot;
}
transformed parameters {
    vector[T] hl;
    for (t in 1:T)
        hl[t] = exp(loghl[t]) + delta[t] * hldot[t];
}
model {
    sigmaLoghl ~ normal(0, 2.3); // log(10) = 2.3
    sigmaHldot ~ normal(0, 10);
    for (t in 1:T)
        quiz[t] ~ bernoulli(exp(-delta[t] / hl[t]));
    
    loghl[1] ~ normal(0, 4);
    hldot[1] ~ normal(0, 10);
    for (t in 2:T)
        loghl[t] ~ normal(loghl[t-1], sigmaLoghl);
    for (t in 2:T)
        hldot[t] ~ normal(hldot[t-1], sigmaHldot);
}
