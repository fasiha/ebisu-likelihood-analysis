# Ebisu v3 playground
See the [Ebisu v3 RFC](https://github.com/fasiha/ebisu/issues/58).

## Ebisu v3 and Stan
[Stan](https://mc-stan.org/) is, for our purposes, a Monte Carlo solver. Given a Bayesian model and some data, it computes the model's parameters by very intelligently drawing thousands of samples from the probability distribution of the unknowns after applying all the data. Since Ebisu is a Bayesian model, where for v3 the unknowns are the initial halflife and the ideal boost factor, we can write its model in the Stan format, [ebisu3_binomial.stan](./ebisu3_binomial.stan), and ask Stan to do all the heavy lifting. Then we can compare Stan's estimate of our unknowns to the estimates Ebisu v3 computes.

So.

See [ebisu3_stan.py](./ebisu3_stan.py) for a quick script that
1. loads your `collection.anki2` Anki database of flashcard history (if you export your Anki deck to an apkg file and unzip that, this collection.anki2 file, which is a SQLite database file, will be inside), cleans the reviews, converts timestamps to elapsed hours, etc.,
2. finds a few cards (e.g., a card that you got "right" >85% of reviews),
3. applies Ebisu v3 and Stan to each.

To run this, you'll need the following setup (for macOS/Unix):
```bash
python -m pip install cmdstanpy scipy matplotlib pandas   # install dependencies
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()" # build cmdstan
```

Of course [ebisu3_stan.py](./ebisu3_stan.py) can be adapted to use another source of flashcard data (not Anki). It also demonstrates how to run just Ebisu v3.

When I run `python ebisu3_stan.py`, it currently loads four flashcards, printing the following output:
```
loaded SQL data, 16623 rows
split flashcards into train/test, 190 cards in train set
```
followed by some Stan output and a few Markdown lists with numbers—in the below, the units of `init hl` are hours, and `boost` is unitless.

- Card 1300038030806
  - Stan
    - hl0   mean=20.68, std=12.12
    - boost mean=2.597, std=0.5506
  - Ebisu v3
    - hl0   mean=20.84, std=11.79
    - boost mean=2.589, std=0.5318
  - percent difference
    - hl0 mean: 0.747%
    - hl0 std : -2.71%
    - boost mean: -0.317%
    - boost std : -3.42%
- Card 1300038030504
  - Stan
    - hl0   mean=25.2, std=11.27
    - boost mean=2.571, std=0.6613
  - Ebisu v3
    - hl0   mean=25.13, std=10.49
    - boost mean=2.565, std=0.639
  - percent difference
    - hl0 mean: -0.298%
    - hl0 std : -6.97%
    - boost mean: -0.263%
    - boost std : -3.37%
- Card 1300038030485
  - Stan
    - hl0   mean=28.78, std=13.18
    - boost mean=2.846, std=0.6973
  - Ebisu v3
    - hl0   mean=28.77, std=12.39
    - boost mean=2.845, std=0.678
  - percent difference
    - hl0 mean: -0.0369%
    - hl0 std : -5.96%
    - boost mean: -0.0465%
    - boost std : -2.77%
- Card 1300038030542
  - Stan
    - hl0   mean=26.79, std=12.87
    - boost mean=2.916, std=0.6967
  - Ebisu v3
    - hl0   mean=26.69, std=12.08
    - boost mean=2.922, std=0.6837
  - percent difference
    - hl0 mean: -0.374%
    - hl0 std : -6.15%
    - boost mean: 0.193%
    - boost std : -1.86%


As you can see, Ebisu and Stan agree quite well on the final estimate of boost's mean and standard deviation, as well as on the initial halflife's mean, but it appears that Ebisu consistently underestimates the initial halflife's standard deviation compared to Stan. This discrepancy doesn't go away even if I crank up the number of samples (the accuracy) of both methods. I'm investigating why this happens.

> The reason we don't just use Stan to fit the data instead of all this custom Python code Ebisu has is: although Stan translates its model into C++ and compiles that down to super-optimized binary code, it's style of Monte Carlo sampling (called Markov chain Monte Carlo or MCMC) is orders of magnitude slower than Ebisu v3's implementation. We use every scrap of information about this specific model to make an estimator that's *pretty fast* (leveraging linear solvers and importance sampling), while Stan as a general purpose solver doesn't know all the mathematical tricks that pertain to our model.

## Likelihood analysis
See https://github.com/fasiha/ebisu/issues/35#issuecomment-899252582 for context and details.

```console
git clone https://github.com/fasiha/ebisu-likelihood-analysis
cd ebisu-likelihood-analysis

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
