"""
run as 
$ python -m nose test_ebisu3.py
or
$ python test_ebisu3.py
"""

import ebisu3 as ebisu
import unittest
import numpy as np

MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


class TestEbisu(unittest.TestCase):

  def test_gamma_update(self):
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    a, b = initHlPrior

    for fraction in [0.1, 0.5, 1., 2., 10.]:
      t = initHlMean * fraction

      for noisy in [0, 0.1, 0.3, 0.7, 0.9, 1]:
        z = noisy >= 0.5
        q1 = noisy if z else 1 - noisy
        q0 = 1 - q1
        updated = ebisu._gammaUpdateNoisy(a, b, t, q1, q0, z)

        msg = f'q1={q1}, q0={q0}, z={z}, noisy={noisy}, u={updated}'
        if z:
          self.assertTrue(updated.mean >= initHlMean, msg)
        else:
          self.assertTrue(updated.mean <= initHlMean, msg)

      for result in [0, 1]:
        updated = ebisu._gammaUpdateBinomial(a, b, t, result, 1)
        if result:
          self.assertTrue(updated.mean >= initHlMean)
        else:
          self.assertTrue(updated.mean <= initHlMean)

  def test_simple(self):
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    boostMean = 1.5
    boostBeta = 3.0
    boostPrior = (boostBeta * boostMean, boostBeta)

    nowMs = ebisu._timeMs()

    currentHalflife = initHlMean

    init = ebisu.Model(
        elapseds=[],
        results=[],
        startStrengths=[],
        initHalflifePrior=initHlPrior,
        boostPrior=boostPrior,
        startTime=nowMs,
        currentHalflife=currentHalflife,
        logStrength=0.0,
    )

    updateds: list[ebisu.Model] = []
    for fraction in [0.1, 0.5, 1.0, 2.0, 10.0]:
      for result in [0, 1]:
        elapsedHours = fraction * initHlMean
        updated = ebisu.simpleUpdateRecall(
            init,
            elapsedHours,
            result,
            total=1,
            now=nowMs + elapsedHours * MILLISECONDS_PER_HOUR,
            reinforcement=1.0)

        msg = f'result={result}, fraction={fraction} => currHl={updated.currentHalflife}'
        if result:
          self.assertTrue(updated.currentHalflife >= initHlMean, msg)
        else:
          self.assertTrue(updated.currentHalflife <= initHlMean, msg)

        updateds.append(updated)
    # print(updateds)


if __name__ == '__main__':
  import os
  # get just this file's module name: no `.py` and no path
  name = os.path.basename(__file__).replace(".py", "")
  unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromName(name))