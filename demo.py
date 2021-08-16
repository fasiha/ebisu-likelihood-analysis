from email.utils import parsedate_tz, mktime_tz
import pandas as pd
import numpy as np
import ebisu
import typing

from utils import split_by, partition_by

Model = tuple[float, float, float]
Updater = typing.Callable[[Model, int, float], Model]


def likelihood(initialModel: Model,
               tnows: list[float],
               results: list[int],
               update: Updater,
               verbose: bool = False) -> float:
    model = initialModel
    logProbabilities: list[float] = []
    for (tnow, result) in zip(tnows, results):
        boolResult = result > 1  # 1=fail in Anki, 2=hard, 3=normal, 4=easy

        logPredictedRecall = ebisu.predictRecall(model, tnow)
        # Bernoulli trial's probability mass function: p if result=True, else 1-p, except in log
        logProbabilities.append(logPredictedRecall if boolResult else np.
                                log(-np.expm1(logPredictedRecall)))
        model = update(model, result, tnow)
        if verbose:
            print(f'model={model}')

    if verbose:
        print(f'logProbabilities={logProbabilities}')
    # return joint probability assuming independent trials: prod(prob) or sum(logProb)
    return sum(logProbabilities)


def dfToVariables(df):
    g = df.copy().sort_values('timestamp')
    g.timestamp -= g.iloc[0].timestamp
    hour_per_second = 1 / 3600
    # drop the first
    tnows_hours: list[float] = np.diff(g.timestamp.values) * hour_per_second
    results: list[int] = g.ease.values[1:]

    ret = []
    for partition in partition_by(lambda tnow_res: tnow_res[1] > 1,
                                  zip(tnows_hours, results)):
        # `partition` is all True or all False
        first_result = partition[0][1] > 1
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


def dfToLikelihood(df, default_a: float, updater: Updater):
    tnows_hours, results, g = dfToVariables(df[df.cardId == cid])

    hl = np.logspace(0, 3, 50)
    lik = [
        likelihood((default_a, default_a, h), tnows_hours, results, updater)
        for h in hl
    ]
    best_hl: float = hl[np.argmax(lik)]
    return best_hl, hl, lik, g, tnows_hours, results


def boostedUpdateModel(model: Model,
                       tnow: float,
                       result: int,
                       baseBoost: float,
                       verbose: bool = False) -> Model:
    extra = baseBoost - 1  # e.g., 0.4 if baseBoost is 1.4
    assert extra >= 0
    baseBoosts = [
        0, 1.0, baseBoost - extra / 2, baseBoost, baseBoost + extra / 2
    ]

    boolResult = result > 1
    newModel = ebisu.updateRecall(model, boolResult, 1, tnow)
    b = baseBoosts[result]
    boostedModel = ebisu.rescaleHalflife(newModel, b)
    if verbose:
        hlBoost: float = ebisu.modelToPercentileDecay(
            newModel) / ebisu.modelToPercentileDecay(model)
        print(f'result={result}, hlBoost={hlBoost}, baseBoost={b}')
    return boostedModel


def traintest(inputDf):
    l = []
    for key, df in inputDf.copy().groupby('cardId'):
        if len(df) < 5:
            continue
        _tnows_hours, results, _ = dfToVariables(df)
        l.append({
            'df': df,
            'len': len(results),
            'key': key,
            'pctCorrect': np.mean(np.array(results) > 1)
        })
    l.sort(key=lambda d: d['pctCorrect'])
    return l


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
    cid = 1354715369763.0  # 67%, 31 quizzes

    boosts = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    liks = []
    for baseBoost in boosts:
        if baseBoost is None:
            updater: Updater = lambda model, result, tnow: ebisu.updateRecall(
                model, result > 1, 1, tnow)
        else:
            updater: Updater = (  # type: ignore
                lambda model, result, tnow: boostedUpdateModel(
                    model, tnow, result, baseBoost))
        best_hl, hl, lik, g, tnows_hours, results = dfToLikelihood(
            df[df.cardId == cid], 2.0, updater)
        liks.append(lik)
        print(f'done with baseBoost={baseBoost}')

    plt.figure()
    plt.semilogx(hl, np.array(liks).T)
    plt.grid()
    plt.xlabel('initial halflife (hours)')
    plt.ylabel('log likelihood')
    plt.legend([f'boost={b}' for b in boosts])

    print(
        np.array([[*max(zip(v, hl), key=lambda xh: xh[0]), baseBoost]
                  for (v, baseBoost) in zip(liks, boosts)]))
    # likelihood([2., 2., 4.], tnows_hours, results, lambda m, r, t: boostedUpdateModel(m, t, r, 1.9, True), True)

    # groups = traintest(df)
    # [{'len':g['len'], 'pct':g['pctCorrect'], 'cid':g['key']} for g in groups[:50]]
