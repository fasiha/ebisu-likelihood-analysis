"""
run as 
$ python -m nose test_ebisu3.py
or
$ python test_ebisu3.py
"""

import ebisu3 as ebisu
import unittest

MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


class TestEbisu(unittest.TestCase):

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

    elapsedHours = 1.0  # hours

    print('init', init)
    for elapsedHours in [1.0, 6.5, 10, 20]:
      updated = ebisu.simpleUpdateRecall(
          init,
          elapsedHours,
          1,
          total=1,
          now=nowMs + elapsedHours * MILLISECONDS_PER_HOUR,
          reinforcement=1.0)
      print('posterior inithl mean', ebisu._gammaToMean(*updated.initHalflifePrior))
      print('currhl / inithl',
            updated.currentHalflife / ebisu._gammaToMean(*updated.initHalflifePrior))
      print('updated', updated)

    self.assertTrue(updated, "yay")


if __name__ == '__main__':
  import os
  # get just this file's module name: no `.py` and no path
  name = os.path.basename(__file__).replace(".py", "")
  unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromName(name))