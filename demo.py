from email.utils import parsedate_tz, mktime_tz
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import ebisu
import typing

from utils import split_by, partition_by


def likelihood(initialModel,
               tnows,
               results,
               baseBoost: typing.Union[None, float] = None):
    model = initialModel
    logProbabilities = []
    for (tnow, result) in zip(tnows, results):
        logPredictedRecall = ebisu.predictRecall(model, tnow)
        # Bernoulli trial's probability mass function: p if result=True, else 1-p, except in log
        logProbabilities.append(logPredictedRecall if result else np.
                                log(-np.expm1(logPredictedRecall)))
        newModel = ebisu.updateRecall(model, result, 1, tnow)
        if baseBoost is None:
            model = newModel
        else:
            hlBoost = ebisu.modelToPercentileDecay(
                newModel) / ebisu.modelToPercentileDecay(model)
            boostedModel = ebisu.rescaleHalflife(newModel, hlBoost * baseBoost)
            model = boostedModel

    # return joint probability assuming independent trials: prod(prob) or sum(logProb)
    return sum(logProbabilities)


def dfToVariables(df):
    g = df.copy().sort_values('timestamp')
    g.timestamp -= g.iloc[0].timestamp
    hour_per_second = 1 / 3600
    # drop the first
    tnows_hours = np.diff(g.timestamp.values) * hour_per_second
    results = g.ease.values[1:] > 1

    ret = []
    for partition in partition_by(lambda tnow_res: tnow_res[1],
                                  zip(tnows_hours, results)):
        # `partition` is all True or all False
        first_result = partition[0][1]
        if first_result or len(partition) <= 1:
            ret.extend(partition)
        else:
            # failure, and more than one of them in a row. Group them within a time period
            splits = split_by(lambda v, vs: (v[0] - vs[0][0]) >= 0.5,
                              partition)
            # Then for each group of timed-clusters, pick the last one
            ret.extend([split[-1] for split in splits])
    tnows_hours, results = zip(*ret)

    return tnows_hours, results, g


def dfToLikelihood(df, default_a, *args, **kwargs):
    tnows_hours, results, g = dfToVariables(df[df.cardId == cid])

    hl = np.logspace(0, 4, 100)
    lik = [
        likelihood([default_a, default_a, h], tnows_hours, results, *args,
                   **kwargs) for h in hl
    ]
    best_hl = hl[np.argmax(lik)]
    return best_hl, hl, lik, g, tnows_hours, results


if __name__ == '__main__':
    df = pd.read_csv("fuzzy-anki.csv", parse_dates=True)
    # via https://stackoverflow.com/a/31905585
    df['timestamp'] = df.dateString.map(
        lambda s: mktime_tz(parsedate_tz(s.replace("GMT", ""))))

    import pylab as plt
    plt.ion()

    if 0 == 1:
        results = []
        cardId_group = df.groupby('cardId')
        from tqdm import tqdm
        for cardId, group in tqdm(cardId_group):
            failrate = (group.ease <= 1).sum() / len(group)
            if failrate == 0: continue
            best_hl, hl, lik, gdf = dfToLikelihood(group, 2.)
            results.append(
                dict(cardId=cardId,
                     best_hl=best_hl,
                     failrate=(gdf.ease.iloc[1:] <= 1).mean(),
                     tot=len(gdf)))
        rdf = pd.DataFrame(results).sort_values('best_hl')

        rdf.plot.scatter(x='failrate', y='best_hl')
        plt.gca().set_yscale('log')

    cid = 1300038030580.0  # 90% pass rate, 30 quizzes
    cid = 1300038030510.0  # 85% pass rate, 20 quizzes

    boosts = [None, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    liks = []
    for baseBoost in boosts:
        best_hl, hl, lik, g, tnows_hours, results = dfToLikelihood(
            df[df.cardId == cid], 2.0, baseBoost=baseBoost)
        liks.append(lik)
        print(f'done with baseBoost={baseBoost}')

    plt.figure()
    plt.semilogx(hl, np.array(liks).T)
    plt.grid()
    plt.xlabel('initial halflife (hours)')
    plt.ylabel('log likelihood')
    plt.legend([f'boost={b}' for b in boosts])