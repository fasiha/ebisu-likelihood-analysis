from tqdm import tqdm  #type:ignore
from email.utils import parsedate_tz, mktime_tz
import pandas as pd  # type: ignore
import numpy as np
import ebisu  # type: ignore
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
    hour_per_millisecond = 1 / 3600e3
    # drop the first
    tnows_hours: list[float] = np.diff(
        g.timestamp.values.astype('datetime64[ms]')).astype(
            'timedelta64[ms]').astype(float) * hour_per_millisecond
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
    tnows_hours, results, g = dfToVariables(df)

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
    allGroups = []
    for key, df in inputDf.copy().groupby('cid'):
        if len(df) < 5:
            continue

        _tnows_hours, results, _ = dfToVariables(df)
        fractionCorrect = np.mean(np.array(results) > 1)
        if len(results) < 5 or fractionCorrect < 0.67:
            continue

        allGroups.append({
            'df': df,
            'len': len(results),
            'key': key,
            'fractionCorrect': fractionCorrect
        })
    allGroups.sort(key=lambda d: d['fractionCorrect'])
    trainGroups = [
        group for group in allGroups[::3] if group['fractionCorrect'] < 1.0
    ]
    return trainGroups, allGroups


def likelihoodHelper(tnows_hours: list[float], results: list[int],
                     initAlphaBeta: float, initHl: float, baseBoost: float):
    model = (initAlphaBeta, initAlphaBeta, initHl)
    updater: Updater = lambda model, result, tnow: boostedUpdateModel(
        model, tnow, result, baseBoost)
    return likelihood(model, tnows_hours, results, updater)


if __name__ == '__main__':
    sqlReviewsWithCards = 'select revlog.*, notes.flds from revlog inner join notes on revlog.cid = notes.id'
    sqlAllReviews = sqlReviewsWithCards.replace(' inner join ',
                                                ' left outer join ')
    SQL_TO_USE = sqlReviewsWithCards
    # Sometimes you delete cards because you were testing reviews, or something?
    # So here you might want to just look at reviews for which the matching cards
    # still exist in the deck.

    import sqlite3
    con = sqlite3.connect('collection.anki2')
    df = pd.read_sql(SQL_TO_USE, con)
    # WILL ONLY LOAD REVIEWS FOR CARDS THAT STILL EXIST
    # Change "inner join" to "left outer join" above to look at ALL reviews
    con.close()
    df['timestamp'] = df.id.astype('datetime64[ms]')
    print(f'loaded SQL data, {len(df)} rows')

    train, _ = traintest(df)
    train = train[::10]  # further subdivide
    print(f'split flashcards into train/test, {len(train)} cards in train set')

    import pylab as plt  # type: ignore
    plt.ion()

    hl = np.logspace(0, 3, 50)
    boost = np.logspace(0, np.log10(2), 5)
    hls, boosts = np.meshgrid(hl, boost)
    likelihoodsPerGroup = []
    for group in tqdm(train):
        tnows_hours, results, _ = dfToVariables(group['df'])
        initAlphaBeta = 2.0
        curriedLikelihood = lambda *args: likelihoodHelper(
            tnows_hours, results, initAlphaBeta, *args)

        liks = np.vectorize(curriedLikelihood)(hls, boosts)
        likelihoodsPerGroup.append(liks)

    exampleFig, exampleAxs = plt.subplots(3, 3)
    for idx, (ax, lik, group) in enumerate(
            zip(exampleAxs.ravel(), likelihoodsPerGroup, train)):
        ax.semilogx(hl, lik.T)
        ax.set_xlabel('init halflife (hours)')
        ax.set_ylabel('log lik.')
        ax.grid()
        ax.set_title(
            f'{group["len"]} reviews, {group["fractionCorrect"]*100:0.1f}% correct'
        )
        if idx == 0:
            ax.legend([f'boost={b:0.2f}' for b in boost])
    exampleFig.tight_layout()

    fig, [ax1, ax2] = plt.subplots(2, 1)
    ax1.plot(boost,
             np.vstack([np.max(x, axis=1) for x in likelihoodsPerGroup]).T)
    ax1.set_xlabel('baseBoost')
    ax1.set_ylabel('max log lik.')
    # fig.legend([f'{int(group["key"])}' for group in train])
    ax1.set_title('Likelihood, max over all init halflife')

    ax2.semilogx(hl, sum(likelihoodsPerGroup).T)
    ax2.set_xlabel('initial halflife')
    ax2.set_ylabel('max likelihood')
    ax2.set_title('Likelihood, summed over all train cards')
    ax2.legend([f'boost={b:0.2f}' for b in boost])

    fig.tight_layout()
