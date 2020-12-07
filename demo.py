from email.utils import parsedate_tz, mktime_tz
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import ebisu

from typing import Callable, TypeVar
from collections.abc import Iterable
T = TypeVar('T')


def split_by(split_pred: Callable[[T, list[T]], bool],
             lst: Iterable[T]) -> list[list[T]]:
    "Allows each element to decide if it wants to be in previous partition"
    lst = iter(lst)
    try:
        x = next(lst)
    except StopIteration:  # empty iterable (list, zip, etc.)
        return []
    ret: list[list[T]] = []
    ret.append([x])
    for x in lst:
        if split_pred(x, ret[-1]):
            ret.append([x])
        else:
            ret[-1].append(x)
    return ret


def partition_by(f: Callable[[T], bool], lst: Iterable[T]) -> list[list[T]]:
    "See https://clojuredocs.org/clojure.core/partition-by"
    lst = iter(lst)
    try:
        x = next(lst)
    except StopIteration:  # empty iterable (list, zip, etc.)
        return []
    ret: list[list[T]] = []
    ret.append([x])
    y = f(x)
    for x in lst:
        newy = f(x)
        if y == newy:
            ret[-1].append(x)
        else:
            ret.append([x])
        y = newy
    return ret


def likelihood(initialModel, tnows, results):
    model = initialModel
    logProbabilities = []
    for (tnow, result) in zip(tnows, results):
        logPredictedRecall = ebisu.predictRecall(model, tnow)
        # Bernoulli trial's probability mass function: p if result=True, else 1-p, except in log
        logProbabilities.append(logPredictedRecall if result else np.
                                log(-np.expm1(logPredictedRecall)))
        model = ebisu.updateRecall(model, result, 1, tnow)

    # return joint probability assuming independent trials: prod(prob) or sum(logProb)
    return sum(logProbabilities)


def dfToLielihood(df, default_a):
    g = df.copy().sort_values('timestamp')
    g.timestamp -= g.iloc[0].timestamp
    hour_per_second = 1 / 3600
    # drop the first
    tnows_hours = g.timestamp.values[1:] * hour_per_second
    results = g.ease.values[1:] > 1

    ret = []
    for partition in partition_by(lambda tnow_res: tnow_res[1],
                                  zip(tnows_hours, results)):
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
    hl = np.logspace(1, 5)
    lik = [likelihood([default_a,default_a, h], tnows_hours, results) for h in hl]
    best_hl = hl[np.argmax(lik)]
    return best_hl, hl, lik, g


df = pd.read_csv("fuzzy-anki.csv", parse_dates=True)
# via https://stackoverflow.com/a/31905585
df['timestamp'] = df.dateString.map(
    lambda s: mktime_tz(parsedate_tz(s.replace("GMT", ""))))

results = []
cardId_group = df.groupby('cardId')
from tqdm import tqdm
for cardId, group in tqdm(cardId_group):
    failrate = (group.ease <= 1).sum() / len(group)
    if failrate == 0: continue
    best_hl, hl, lik, gdf = dfToLielihood(group, 2.)
    results.append(
        dict(cardId=cardId, best_hl=best_hl, failrate=(gdf.ease.iloc[1:]<=1).mean(),
             tot=len(gdf)))
rdf = pd.DataFrame(results).sort_values('best_hl')

import pylab as plt
plt.ion()

rdf.plot.scatter(x='failrate', y='best_hl')
plt.gca().set_yscale('log')

best_hl, hl, lik, g = dfToLielihood(df[df.cardId == 1300038030580.0],4)
best_hl, hl, lik, g = dfToLielihood(df[df.cardId == 1300038030510.0],32)

plt.figure()
plt.semilogx(hl, lik)
plt.grid()