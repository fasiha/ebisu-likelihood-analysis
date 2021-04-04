import stan
from email.utils import parsedate_tz, mktime_tz
import pandas as pd
import numpy as np
from utils import split_by, partition_by


def fails(df):
    cardIdToFailrate = dict()
    cardId_group = df.groupby('cardId')
    for cardId, group in (cardId_group):
        failrate = (group.ease <= 1).sum() / len(group)
        cardIdToFailrate[cardId] = failrate
    return cardIdToFailrate
    # print(np.sort(list(cardIdToFailrate.values())))


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

    return tnows_hours, results


def fitVariables(deltas, results, viz=False, msg=''):
    t = np.cumsum(deltas)
    data = {
        "T": len(results),
        "quiz": [int(x) for x in results],
        "delta": deltas,
        "time": t
    }

    model = open('model.stan', 'r').read()
    posterior = stan.build(model, data=data, random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=10_000)
    print(fit)
    fitdf = fit.to_frame()
    fitdf = fitdf[[c for c in fitdf.columns if not c.endswith("_")]]

    if viz:
        import pylab as plt
        plt.ion()

        import pandas as pd
        dfplot = fitdf[[c for c in fitdf.columns if "." not in c]]

        p = pd.plotting.scatter_matrix(dfplot)

        if len(msg):
            p[0, 0].get_figure().suptitle(msg)

        p[0, 0].get_figure().set_tight_layout(True)
    return fitdf, fit


def fitCardid(df, cid, viz=True):
    deltas, results = dfToVariables(df[df.cardId == cid])
    months = sum(deltas) / 24 / 365 * 12
    percentRight = 100 * np.mean(results)
    return fitVariables(
        deltas,
        results,
        viz=viz,
        msg=
        f'card {int(cid)}: {len(results)} quizzes, {percentRight:.3}%, {months:.1f} months'
    )


df = pd.read_csv("fuzzy-anki.csv", parse_dates=True)
# via https://stackoverflow.com/a/31905585
df['timestamp'] = df.dateString.map(
    lambda s: mktime_tz(parsedate_tz(s.replace("GMT", ""))))


def demo():
    model = open('model.stan', 'r').read()
    posterior = stan.build(model,
                           data={
                               "T": 4,
                               "quiz": [1, 1, 0, 1],
                               "delta": [50, 50, 100, 100]
                           },
                           random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=10_000)
    fitdf = fit.to_frame()
    fitdf = fitdf[[c for c in fitdf.columns if not c.endswith("_")]]


# 85% pass rate, 20 quizzes
fitdf, fit = fitCardid(df, 1300038030510.0, viz=False)
deltas, results = dfToVariables(df[df.cardId == 1300038030510.0])

# fitCardid(df, 1300038030580.0)  # 90% pass rate, 30 quizzes


def analyzeDf(df):
    cardId_group = df.groupby('cardId')
    cids = []
    lens = []
    passRates = []
    times = []
    for cardId, group in cardId_group:
        if len(group) < 4: continue
        deltas, results = dfToVariables(group)
        passRates.append(np.mean(results))
        lens.append(len(group))
        times.append(np.sum(deltas) / 24 / 365 * 12)
        cids.append(cardId)

    summary = pd.DataFrame(zip(cids, lens, passRates, times),
                           columns='cardid,len,passRate,months'.split(','))
    pd.plotting.scatter_matrix(summary[summary.columns[1:]])
    return summary


# summary = analyzeDf(df)
# [fitCardid(df, c) for c in summary[summary['len'] > 40]['cardid']]
