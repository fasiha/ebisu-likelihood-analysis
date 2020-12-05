from email.utils import parsedate_tz, mktime_tz
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import ebisu

df = pd.read_csv("fuzzy-anki.csv", parse_dates=True)
# via https://stackoverflow.com/a/31905585
df['timestamp'] = df.dateString.map(
    lambda s: mktime_tz(parsedate_tz(s.replace("GMT", ""))))

# cardId_group = df.groupby('cardId')
# for cardId, group in cardId_group:
#     print(group)
#     break


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


def dfToLielihood(df):
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
    print(ret)
    tnows_hours, results = zip(*ret)
    # print(likelihood([2., 2., 1.], tnows_hours, results))
    # print(likelihood([2., 2., 10000.], tnows_hours, results))
    # print(likelihood([2., 2., 1000000.], tnows_hours, results))
    # options = dict(disp=True, maxiter=10_000)
    # tol = 1e-11
    # res = minimize(
    #     lambda x: -likelihood([x[0], x[0], x[1]], tnows_hours, results),
    #     [2., 10e3],
    #     tol=tol,
    #     options=options)
    # print(res)
    # res = minimize(
    #     lambda x: -likelihood([x[0], x[0], 10e3], tnows_hours, results), [2.],
    #     tol=tol,
    #     options=options)
    # print(res)
    # res = minimize(lambda x: -likelihood([2., 2., x[0]], tnows_hours, results),
    #                [10e3],
    #                tol=tol,
    #                options=options)
    # print(res)

    hl = np.logspace(1, 5)
    a = 4.
    lik = [likelihood([a,a, h], tnows_hours, results) for h in hl]
    best_hl = hl[np.argmax(lik)]
    return best_hl, hl, lik


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


best_hl, hl, lik = dfToLielihood(df[df.cardId == 1300038030580.0])

import pylab as plt
plt.ion()
plt.semilogx(hl, lik)
plt.grid()