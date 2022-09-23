import ebisu3 as ebisu
import utils

initHlMean = 10  # hours
initHlBeta = 0.1
initHlPrior = (initHlBeta * initHlMean, initHlBeta)

boostMean = 1.5
boostBeta = 3.0
boostPrior = (boostBeta * boostMean, boostBeta)

model = ebisu.initModel(initHlPrior, boostPrior)
model = ebisu.updateRecall(model, 15.0, 1)
model = ebisu.updateRecall(model, 35.0, 0)
model = ebisu.updateRecall(model, 25.0, 1)
model = ebisu.updateRecall(model, 45.0, 0.9)


def summarize(big: ebisu.Model, small: ebisu.Model):
  for bigsmall in ['big', 'small']:
    for rv in ['initHl', 'boost']:
      this = (big if bigsmall == 'big' else small)
      post = this.prob.initHl if rv == "initHl" else this.prob.boost
      mean = ebisu._gammaToMean(*post)
      print(f'{bigsmall} | {rv} | {post} | {mean:0.3f} | {this.pred.currentHalflifeHours:0.3f}')


summarize(
    ebisu.updateRecallHistory(model, size=100_000), ebisu.updateRecallHistory(model, size=1_000))

df = utils.sqliteToDf('collection.anki2', True)
print(f'loaded SQL data, {len(df)} rows')
train, TEST_TRAIN = utils.traintest(df)
t = next(t for t in train if t.fractionCorrect > 0.8)
real = ebisu.initModel(initHlPrior, boostPrior)
for x, t in zip(t.results, t.dts_hours):
  real = ebisu.updateRecall(real, t, int(x >= 2))

summarize(
    ebisu.updateRecallHistory(real, size=100_000), ebisu.updateRecallHistory(real, size=1_000))
