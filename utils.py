import pandas as pd  # type: ignore
from dataclasses import dataclass
from scipy.stats import multinomial  #type:ignore
import numpy as np
from typing import Callable, TypeVar
from collections.abc import Iterable
import typing

T = TypeVar('T')


def weightedMean(w: np.ndarray, x: np.ndarray) -> float:
  return np.sum(w * x) / np.sum(w)


def weightedMeanVar(w: np.ndarray, x: np.ndarray) -> tuple[float, float]:
  mean = weightedMean(w, x)
  var = np.sum(w * (x - mean)**2) / np.sum(w)
  return (mean, var)


def meanVarToBeta(mean, var) -> tuple[float, float]:
  """Fit a Beta distribution to a mean and variance."""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


def clampLerpFloat(x1: float, x2: float, y1: float, y2: float, x: float):
  mu = (x - x1) / (x2 - x1)  # will be >=0 and <=1
  # branchless: hoping it's faster (cache misses, etc.) than the equivalent:
  # `y1 if (x < x1) else y2 if (x > x2) else (y1 * (1 - mu) + y2 * mu)`
  return (x < x1) * y1 + (x > x2) * y2 + (x1 <= x <= x2) * (y1 * (1 - mu) + y2 * mu)


def sequentialImportanceResample(particles: np.ndarray,
                                 weights: np.ndarray,
                                 N=None) -> tuple[np.ndarray, np.ndarray]:
  if N is None:
    N = len(particles)
  draw: np.ndarray = multinomial.rvs(N, weights / np.sum(weights))
  # each element of `draw` is an integer, the number of times the particle at that index should appear in the output

  # this isn't going to be fast FIXME
  newParticles = np.hstack(
      [np.ones(repeat) * particle for repeat, particle in zip(draw, particles)])
  newWeights = np.ones(N)
  return (newParticles, newWeights)


def split_by(split_pred: Callable[[T, list[T]], bool], lst: Iterable[T]) -> list[list[T]]:
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


def traintest(inputDf, minQuiz=5, minFractionCorrect=0.67):
  """Split an Anki Pandas dataframe of all reviews into a train set and a test set
  
  Groups all reviews into those belonging to the same card. Throws out cards with too few reviews
  or too few correct reviews. Splits the resulting cards into a small train set and a larger test
  set. We'll only look at the training set when designing our new update algorithm.

  The training set will only have cards with less than perfect reviews (since it's impossible for
  a meaningful likelihood to be computed for all passing reviews).
  """
  allGroups: list[Card] = []
  for key, df in inputDf.groupby('cid'):
    if len(df) < minQuiz:
      continue

    # "Group chunks should be treated as immutable, and changes to a group chunk may produce
    # unexpected results"
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#transformation
    sortedDf = df.copy().sort_values('timestamp')
    dts_hours, results, absts_hours = dfToVariables(sortedDf)
    fractionCorrect = np.mean(np.array(results) > 1)
    if len(results) < minQuiz or fractionCorrect < minFractionCorrect:
      continue

    allGroups.append(
        Card(
            df=sortedDf,
            len=len(results),
            key=key,
            fractionCorrect=fractionCorrect,
            dts_hours=dts_hours,
            results=results,
            absts_hours=absts_hours,
        ))
  allGroups.sort(key=lambda d: d.fractionCorrect)
  trainGroups = [group for group in allGroups[::3] if group.fractionCorrect < 1.0]
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


def dfToVariables(g):
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
  assert g.timestamp.is_monotonic_increasing
  hour_per_millisecond = 1 / 3600e3
  # drop the first quiz (that's our "learning" case)
  # delta time between this quiz and the previous quiz/study
  dts_hours: list[float] = np.diff(g.timestamp.values.astype('datetime64[ms]')).astype(
      'timedelta64[ms]').astype(float) * hour_per_millisecond
  # 1-4: results of quiz
  results: list[int] = g.ease.values[1:]
  # absolute time of this quiz
  ts_hours = g.timestamp.values[1:].astype('datetime64[ms]').astype('timedelta64[ms]').astype(
      float) * hour_per_millisecond

  ret = []
  for partition in partition_by(lambda dt_res: dt_res[1] > 1, zip(dts_hours, results, ts_hours)):
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
  dts_hours, results, ts_hours = zip(*ret)

  return dts_hours, results, ts_hours


@dataclass
class Card:
  df: pd.DataFrame
  len: int
  key: int
  fractionCorrect: float
  dts_hours: list[float]
  results: list[int]
  absts_hours: list[float]
