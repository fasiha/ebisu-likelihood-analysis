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

    tnows_hours = np.diff(tnows_hours)
    results = results[1:]

    return tnows_hours, results


df = pd.read_csv("fuzzy-anki.csv", parse_dates=True)
# via https://stackoverflow.com/a/31905585
df['timestamp'] = df.dateString.map(
    lambda s: mktime_tz(parsedate_tz(s.replace("GMT", ""))))

cid = 1300038030580.0  # 90% pass rate, 30 quizzes
cid = 1300038030510.0  # 85% pass rate, 20 quizzes
tnows_hours, results = dfToLielihood(df[df.cardId == cid], 2)
t = np.cumsum(tnows_hours)

with open('model.stan', 'r') as fid:
    model = fid.read()

import stan
data = {
    "T": len(tnows_hours),
    "quiz": [int(x) for x in results],
    "delta": tnows_hours,
    "time": t
}
posterior = stan.build(model, data=data, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=10_000)

import pylab as plt
plt.ion()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(fit['initHl'].T, fit['learnRate'].T, '.')
axs[0, 0].set_xlabel('init halflife')
axs[0, 0].set_ylabel('learnRate')

axs[1, 0].hist(fit['initHl'].ravel(), 30)
axs[1, 0].set_xlabel('init halflife')

axs[0, 1].hist(fit['learnRate'].ravel(), 30, orientation='horizontal')
axs[0, 1].set_ylabel('learnRate')

plt.tight_layout()
