data {
    int T;
    int<lower=0,upper=1> quiz[T];
    vector[T] delta;
    vector[T] time;
}
parameters {
    real<lower=0> initHl;
    vector<lower=0>[T-1] process;
    real<lower=0> processMean;
    real<lower=0> processStd;
}
transformed parameters {
    vector[T] hl;
    hl[1] = 20 * initHl;
    for (t in 2:T)
        hl[t] = hl[t-1] * fmin(fmax(pow(process[t-1], delta[t-1] / hl[t-1]), 0.5), 3);

    vector[T] p = exp(-delta ./ hl);
}
model {
    initHl ~ exponential(1.0);
    processMean ~ normal(2, 2);
    processStd ~ exponential(1.0);
    process ~ normal(processMean, processStd);
    quiz ~ bernoulli(p);
}
