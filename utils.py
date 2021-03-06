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
  "Allows each element to decide if it wants to not be in previous partition"
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
    if len(results) < minQuiz:
      continue

    fractionCorrect = np.mean(np.array(results) > 1)
    if fractionCorrect < minFractionCorrect:
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
  # 1-4: results of quiz
  results: list[int] = g.ease.values
  # absolute time of this quiz
  ts_hours: list[float] = g.timestamp.values.astype('datetime64[ms]').astype(
      'timedelta64[ms]').astype(float) * hour_per_millisecond

  ret = []
  HOUR_WINDOW = 4.0
  for group in split_by(lambda rev, revs: (rev[1] - revs[-1][1]) > HOUR_WINDOW,
                        zip(results, ts_hours)):
    # each `group` contains a list of reviews that all happened in the same HOUR_WINDOW window.
    # Anki only does daily-ish reviews so reviews this close together must have happened due to a failure.
    # We want to clean up those.

    # See https://stackoverflow.com/a/8534381
    failure = next(((result, t) for result, t in group if result == 1), None)

    if failure:
      # In general, this group will be [fail, pass] or maybe [pass, fail, pass] (if I accidentally clicked "pass" at first).
      # Add only the first failure in this HOUR_WINDOW-group of reviews
      ret.append(failure)
    else:
      # add all reviews in this HOUR_WINDOW-group because all were successes.
      # In reality this group will most likely be 1-element long
      ret.extend(group)

  results, ts_hours = zip(*ret)
  # drop the first quiz (that's our "learning" case)
  # delta time between this quiz and the previous quiz/study
  dts_hours = np.diff(ts_hours).tolist()
  ts_hours = ts_hours[1:]
  results = results[1:]

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
