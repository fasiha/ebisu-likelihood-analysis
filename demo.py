"""
Setup suggestion:

```console
python -m venv likelihood-demo
source likelihood-demo/bin/activate
python -m pip install tqdm pandas numpy ebisu matplotlib
```

This installs a virtual environment via `venv` so you don't pollute your system, then
installs some dependencies.

Then, copy an Anki database, `collection.anki2` to this directory and then run
```console
python demo.py
```
This will generate some plots and save them.

I personally tend to install ipython and run it:
```
python -m pip install ipython
ipython
```
and the run the script there: `%run demo.py`, so I can interact with plots, but that's just me.
"""

from tqdm import tqdm  #type:ignore
import pandas as pd  # type: ignore
import numpy as np
import ebisu  # type: ignore
import typing

from utils import split_by, partition_by

Model = tuple[float, float, float]
Updater = typing.Callable[[Model, int, float], Model]


def likelihood(initialModel: Model,
               dts: list[float],
               results: list[int],
               update: Updater,
               verbose: bool = False) -> float:
  """Compute likelihood of a set of reviews given an initial Ebisu model and update function

  This is the core function for this likelihood analysis: given a set of reviews, `dts` being
  a list of hours since last review (so the first element is the number of hours after the card
  is learned) and `results` being an Anki result:
  - 1=fail
  - 2=hard
  - 3=normal
  - 4=easy,

  as well as an `initialModel` and a model `update` function, reduce all these into a single number:
  the probabilistic likelihood that this model could have generated the reviews seen. This is a
  powerful way to evaluate both `initialModel` and `update`.

  N.B. The *log* likelihood will be returned. Bigger is better (more likely model/update).

  "Likelihood" (or log likelihood) is basically a product (sum) of each actual quiz result's
  predicted outcome. The more surprising quiz results are to the model, the lower the likelihood
  is. A perfect model/updater will assign probabilities of 1.0 to each quiz, yielding a likelihood
  of 1.0. In log, `log(1) = 0` so the best log likelihood possible is 0.
  """
  model = initialModel
  logProbabilities: list[float] = []
  for (dt, result) in zip(dts, results):
    boolResult = result > 1  # 1=fail in Anki, 2=hard, 3=normal, 4=easy

    logPredictedRecall = ebisu.predictRecall(model, dt)
    # Bernoulli trial's probability mass function: p if result=True, else 1-p, except in log
    logProbabilities.append(
        logPredictedRecall if boolResult else np.log(-np.expm1(logPredictedRecall)))
    model = update(model, result, dt)
    if verbose:
      print(f'model={model}')

  if verbose:
    print(f'logProbabilities={logProbabilities}')
  # return joint probability assuming independent trials: prod(prob) or sum(logProb)
  return sum(logProbabilities)


def likelihoodHelper(dts_hours: list[float], results: list[int], initAlphaBeta: float,
                     initHl: float, baseBoost: float):
  """Convert the parameters we're interested in to a call to the `likelihood` function

  The `likelihood` function above is quite generic: it works for any model/update function. We
  may be interested in varying the following:
  - inital alpha=beta
  - initial halflife
  - the basic boost to apply (whatever that might mean)

  This helper is a thin wrapper to `likelihood`.
  """
  model = (initAlphaBeta, initAlphaBeta, initHl)
  updater: Updater = lambda model, result, df: boostedUpdateModel(model, df, result, baseBoost)
  return likelihood(model, dts_hours, results, updater)


def dfToVariables(df):
  """Convert a Pandas dataframe of an Anki SQLite database to a list of delta-times and results

  Given a dataframe containing all reviews of a single card, we want to get a simple list of hours
  before each review as well as the result of that review (1=fail, 2=hard, 3=normal, 4=easy).

  This function does some extra work to find successive failed reviews and combine them into a
  SINGLE failed review if they happened close enough. This is because sometimes in my data, I have
  a failure, and then Anki quizzed me a couple of minutes ago and I accidentally clicked "fail"
  again. This erroneous data can really mess up Ebisu so I want to try and combine these runs of
  two or more successive failures that happen within an interval (say, a half-hour), into a single
  failure.
  """
  g = df.copy().sort_values('timestamp')
  hour_per_millisecond = 1 / 3600e3
  # drop the first
  dts_hours: list[float] = np.diff(g.timestamp.values.astype('datetime64[ms]')).astype(
      'timedelta64[ms]').astype(float) * hour_per_millisecond
  results: list[int] = g.ease.values[1:]

  ret = []
  for partition in partition_by(lambda dt_res: dt_res[1] > 1, zip(dts_hours, results)):
    # `partition_by` splits up the list of `(dt, result)` tuples into sub-lists where each
    # `partition` is all successes or all failures.
    # SO: `partition[0][1]` is "the first reivew's result" (1 through 4).
    first_result = partition[0][1] > 1  # boolean
    if first_result or len(partition) <= 1:
      # either this chunk of reviews are all successes or there's only one failure
      ret.extend(partition)
    else:
      # failure, and more than one of them in a row. Group them within a time period
      GROUP_FAILURE_TIME_HOURS = 0.5
      splits = split_by(lambda v, vs: (v[0] - vs[0][0]) >= GROUP_FAILURE_TIME_HOURS, partition)
      # Then for each group of timed-clusters, pick the last one
      ret.extend([split[-1] for split in splits])
  dts_hours, results = zip(*ret)

  return dts_hours, results, g


def boostedUpdateModel(model: Model,
                       dt: float,
                       result: int,
                       baseBoost: float,
                       verbose: bool = False) -> Model:
  """Early proposed alternative to Ebisu's updateModel: apply a boost after each update

  The basic idea here is that, `ebisu.updateModel` returns the posterior on recall probability at
  the initial point in time that the prior was created, without moving it to the time of the quiz.
  For example, if we learn a fact at midnight and model its recall probability with `[a, b, t]` 
  (`Beta(a, b)` distribution at time `t` hours after midnight), and then have a quiz at 1am, and
  call `ebisu.updateRecall`, the returned model is a posterior is still from the point of view of
  when the fact was first learned. It still only applies to quizzes that might happen sometime
  after midnight even though an hour has passed and that means the strengh of the memory has
  changed.

  We need to have SOME way of moving the posterior past midnight.

  This function presents one stupid way to do that.

  It just scales the posterior halflife by some fixed factor 😅.

  That's what `baseBoost` does. With `baseBoost >= 1`, the posterior halflife will be rescaled
  by `baseBoost` factor for normal quiz results (result=2), a bit more for easy, a bit less for
  hard.

  There's a million ways to improve this update mechanism but the point here is to test the
  likelihood mechanism. The simple mechanism taken here is explicitly modeled on Anki.
  """
  extra = baseBoost - 1  # e.g., 0.4 if baseBoost is 1.4
  assert extra >= 0
  baseBoosts = [0, 1.0, baseBoost - extra / 2, baseBoost, baseBoost + extra / 2]

  boolResult = result > 1
  newModel = ebisu.updateRecall(model, boolResult, 1, dt)
  b = baseBoosts[result]
  boostedModel = ebisu.rescaleHalflife(newModel, b)
  if verbose:
    # `hlBoost` is how much Ebisu already boosted the halflife
    # hlBoost: float = ebisu.modelToPercentileDecay(newModel) / ebisu.modelToPercentileDecay(model)
    print(f'result={result}, boost={b}')
  return boostedModel


def traintest(inputDf):
  """Split a Pandas dataframe of all reviews into a train set and a test set
  
  Groups all reviews into those belonging to the same card. Throws out cards with too few reviews
  or too few correct reviews. Splits the resulting cards into a small train set and a larger test
  set. We'll only look at the training set when designing our new update algorithm.

  The training set will only have cards with less than perfect reviews (since it's impossible for
  a meaningful likelihood to be computed for all passing reviews).
  """
  allGroups = []
  for key, df in inputDf.copy().groupby('cid'):
    if len(df) < 5:
      continue

    _dts_hours, results, _ = dfToVariables(df)
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
  trainGroups = [group for group in allGroups[::3] if group['fractionCorrect'] < 1.0]
  return trainGroups, allGroups


def sqliteToDf(filename: str, reviewsWithCardsOnly=True):
  sqlReviewsWithCards = 'select revlog.*, notes.flds from revlog inner join notes on revlog.cid = notes.id'
  sqlAllReviews = sqlReviewsWithCards.replace(' inner join ', ' left outer join ')
  SQL_TO_USE = sqlReviewsWithCards if reviewsWithCardsOnly else sqlAllReviews
  # Sometimes you delete cards because you were testing reviews, or something?
  # So here you might want to just look at reviews for which the matching cards
  # still exist in the deck.

  import sqlite3
  con = sqlite3.connect(filename)
  df = pd.read_sql(SQL_TO_USE, con)
  con.close()
  df['timestamp'] = df.id.astype('datetime64[ms]')
  return df


if __name__ == '__main__':
  df = sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, _ = traintest(df)
  train = train[::10]  # further subdivide, for computational purposes
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  import pylab as plt  # type: ignore
  plt.ion()

  # vary initial halflife and baseBoost amount
  hl = np.logspace(0, 3, 50)
  boost = np.logspace(0, np.log10(2), 5)

  hls, boosts = np.meshgrid(hl, boost)
  likelihoodsPerGroup = []
  for group in tqdm(train):
    dts_hours, results, _ = dfToVariables(group['df'])
    initAlphaBeta = 2.0
    curriedLikelihood = lambda *args: likelihoodHelper(dts_hours, results, initAlphaBeta, *args)

    liks = np.vectorize(curriedLikelihood)(hls, boosts)
    likelihoodsPerGroup.append(liks)

  # Show some example results
  exampleFig, exampleAxs = plt.subplots(3, 3)
  for idx, (ax, lik, group) in enumerate(zip(exampleAxs.ravel(), likelihoodsPerGroup, train)):
    ax.semilogx(hl, lik.T)
    ax.set_xlabel('init halflife (hours)')
    ax.set_ylabel('log lik.')
    ax.grid()
    ax.set_title(f'{group["len"]} reviews, {group["fractionCorrect"]*100:0.1f}% correct')
    if idx == 0:
      ax.legend([f'boost={b:0.2f}' for b in boost])
  exampleFig.tight_layout()
  plt.savefig("examples.png", dpi=300)
  plt.savefig("examples.svg")

  # Aggregate all results
  fig, [ax1, ax2] = plt.subplots(2, 1)
  ax1.plot(boost, np.vstack([np.max(x, axis=1) for x in likelihoodsPerGroup]).T)
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
  plt.savefig("results.png", dpi=300)
  plt.savefig("results.svg")
