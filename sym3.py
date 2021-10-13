import mpmath as mp  #type:ignore
import sympy as s  #type:ignore

pp = s.pprint


def besseliSimp(expr):
  """
  Sympy doesn't simplify `BesselI[-v, b] - BesselI[v, b]` to `(2 BesselK[-v, b] Sin[Pi v])/Pi` so do it here
  """
  query = lambda arg: "besseli" in str(arg) and arg.func == s.Add and len(arg.args) == 2

  founds = expr.find(query)
  if len(founds) != 1:
    return expr
  found = list(founds)[0]
  if len(found.args) != 2:
    return expr
  pos, neg = found.args
  if (pos.func == s.Mul and neg.func == s.besseli):
    pos, neg = neg, pos  # flipped
  elif (neg.func == s.Mul and pos.func == s.besseli):
    pass  # good
  else:
    return expr
  negabs = -neg

  if not len(negabs.args) == 2 and len(pos.args) == 2:
    return expr
  if not negabs.args[0] == -pos.args[0] and negabs.args[1] == pos.args[1]:
    return expr

  # hooray!
  first = abs(pos.args[0])
  second = pos.args[1]

  orig = s.besseli(-first, second) - s.besseli(first, second)
  better = 2 * s.sin(s.pi * first) / s.pi * s.besselk(-first, second)

  return expr.replace(orig, better).replace(-orig, -better).simplify()


h, ah, bh, b, ab, bb, *t = s.symbols('h a_h b_h b a_b b_b t:3', real=True, positive=True)
valsGamma = {
    ab: 10 * 1.4 + 1,
    bb: 10.0,
    ah: 10 * .25 + 1,
    bh: 10.0,
    t[0]: 0.9,
    t[1]: 3.3,
    t[2]: 14.5
}
valsExp = {bb: 1.0, bh: 0.5, t[0]: 0.9, t[1]: 3.3, t[2]: 14.5}

priorhGamma = h**(ah - 1) * s.exp(-bh * h)
priorbGamma = b**(ab - 1) * s.exp(-bb * b)
priorhExp = bh * s.exp(-bh * h)
priorbExp = bb * s.exp(-bb * b)

priors = 'gamma'
# priors = 'exp'
if priors == 'gamma':
  prior = priorhGamma * priorbGamma
elif priors == 'exp':
  prior = priorhExp * priorbExp
lik = s.exp(-t[2] / (h * b**2)) * s.exp(-t[1] / (h * b)) * s.exp(-t[0] / h)

print(priors)
if priors == 'gamma':
  # print((prior * lik).subs(valsGamma))
  den = mp.quadgl(
      lambda b, h: b**14.0 * h**2.5 * mp.exp(-10.0 * b - 0.9 / h - 10.0 * h - 14.5 /
                                             (b**2 * h) - 3.3 / (b * h)), [0, mp.inf], [0, mp.inf],
      error=True)
  eh = mp.quadgl(
      lambda b, h: h * b**14.0 * h**2.5 * mp.exp(-10.0 * b - 0.9 / h - 10.0 * h - 14.5 /
                                                 (b**2 * h) - 3.3 / (b * h)), [0, mp.inf],
      [0, mp.inf],
      error=True)
  eb = mp.quadgl(
      lambda b, h: b * b**14.0 * h**2.5 * mp.exp(-10.0 * b - 0.9 / h - 10.0 * h - 14.5 /
                                                 (b**2 * h) - 3.3 / (b * h)), [0, mp.inf],
      [0, mp.inf],
      error=True)
  print(dict(den=den, eb=eb, eh=eh))
  print(dict(eb=eb[0] / den[0], eh=eh[0] / den[0]))
elif priors == 'exp':
  # print((prior * lik).subs(valsExp))
  den = mp.quadgl(
      lambda b, h: 0.5 * mp.exp(-1.0 * b - 0.9 / h - 0.5 * h - 14.5 / (b**2 * h) - 3.3 / (b * h)),
      [0, mp.inf], [0, mp.inf],
      error=True)
  eb = mp.quadgl(
      lambda b, h: b * 0.5 * mp.exp(-1.0 * b - 0.9 / h - 0.5 * h - 14.5 / (b**2 * h) - 3.3 /
                                    (b * h)), [0, mp.inf], [0, mp.inf],
      error=True)
  eh = mp.quadgl(
      lambda b, h: h * 0.5 * mp.exp(-1.0 * b - 0.9 / h - 0.5 * h - 14.5 / (b**2 * h) - 3.3 /
                                    (b * h)), [0, mp.inf], [0, mp.inf],
      error=True)
  print(dict(den=den, eb=eb, eh=eh))
  print(dict(eb=eb[0] / den[0], eh=eh[0] / den[0]))

expectationH = prior * lik * h
expectationB = prior * lik * b

# denB = s.integrate(prior * lik, (b, 0, s.oo)).simplify()
# denH = s.integrate(prior * lik, (h, 0, s.oo)).simplify()
