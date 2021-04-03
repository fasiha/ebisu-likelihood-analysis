import stan
from email.utils import parsedate_tz, mktime_tz
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import ebisu
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

    tnows_hours = np.diff(tnows_hours)
    results = results[1:]

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

    if viz:
        import pylab as plt
        plt.ion()

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist2d(fit['initHl'].ravel(),
                         fit['learnRate'].ravel(),
                         bins=30)
        axs[0, 0].set_xlabel('init halflife')
        axs[0, 0].set_ylabel('learnRate')

        axs[1, 0].hist(fit['initHl'].ravel(), 30)
        axs[1, 0].set_xlabel('init halflife')

        axs[0, 1].hist(fit['learnRate'].ravel(), 30, orientation='horizontal')
        axs[0, 1].set_ylabel('learnRate')

        if len(msg):
            fig.suptitle(msg)

        plt.tight_layout()


def fitCardid(df, cid):
    deltas, results = dfToVariables(df[df.cardId == cid])
    months = sum(deltas) / 24 / 365 * 12
    percentRight = 100 * np.mean(results)
    fitVariables(
        deltas,
        results,
        viz=True,
        msg=
        f'card {int(cid)}: {len(results)} quizzes, {percentRight:3}%, {months:.1f} months'
    )


df = pd.read_csv("fuzzy-anki.csv", parse_dates=True)
# via https://stackoverflow.com/a/31905585
df['timestamp'] = df.dateString.map(
    lambda s: mktime_tz(parsedate_tz(s.replace("GMT", ""))))

fitCardid(df, 1300038030510.0)  # 85% pass rate, 20 quizzes
fitCardid(df, 1300038030580.0)  # 90% pass rate, 30 quizzes
