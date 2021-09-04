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
