# Ebisu v3 playground

## Ebisu v3 and Stan
[Stan](https://mc-stan.org/) is, for our purposes, a Monte Carlo solver. Given a Bayesian model and some data, it computes the model's parameters by very intelligently drawing thousands of samples from the probability distribution of the unknowns after applying all the data. Since Ebisu is a Bayesian model, where for v3 the unknowns are the initial halflife and the ideal boost factor, we can write its model in the Stan format, [ebisu3_binomial.stan](./ebisu3_binomial.stan), and ask Stan to do all the heavy lifting. Then we can compare Stan's estimate of our unknowns to the estimates Ebisu v3 computes.

So.

See [ebisu3_stan.py](./ebisu3_stan.py) for a quick script that
1. loads your `collection.anki2` Anki database of flashcard history and cleans the reviews,
2. finds a few cards (e.g., a card that you got "right" >85% of reviews),
3. applies Ebisu v3 and Stan to each.

To run this, you'll need to `python -m pip install cmdstanpy scipy matplotlib pandas` first.

Of course [ebisu3_stan.py](./ebisu3_stan.py) can be adapted to use another source of flashcard data (not Anki). It also demonstrates how to run just Ebisu v3.

When I run `python ebisu3_stan.py`, it currently loads four flashcards, printing the following output:
```
loaded SQL data, 16623 rows
split flashcards into train/test, 190 cards in train set
```
followed by some Stan output and a few Markdown tables:

Card 1300038030806:

| variable | Ebisu mean | Ebisu std | Stan mean | Stan std |
|----------|-----------|----------|------------|-----------|
| init hl  | 20.87 |  11.51 |  20.67 | 11.99 |
| boost    | 2.589 | 0.532 | 2.598 | 0.5515 |


Card 1300038030504:

| variable | Ebisu mean | Ebisu std | Stan mean | Stan std |
|----------|-----------|----------|------------|-----------|
| init hl  | 24.96 |  10.24 |  25.18 | 11.19 |
| boost    | 2.569 | 0.6346 | 2.563 | 0.6609 |


Card 1300038030485:

| variable | Ebisu mean | Ebisu std | Stan mean | Stan std |
|----------|-----------|----------|------------|-----------|
| init hl  | 28.7 |  12.53 |  28.86 | 13.33 |
| boost    | 2.844 | 0.6812 | 2.842 | 0.6979 |


Card 1300038030542:

| variable | Ebisu mean | Ebisu std | Stan mean | Stan std |
|----------|-----------|----------|------------|-----------|
| init hl  | 26.53 |  11.64 |  26.57 | 12.61 |
| boost    | 2.93 | 0.6813 | 2.923 | 0.6932 |

As you can see, Ebisu and Stan agree quite well on the final estimate of boost's mean and standard deviation, as well as on the initial halflife's mean, but it appears that Ebisu consistently underestimates the initial halflife's standard deviation compared to Stan. This discrepancy doesn't go away even if I crank up the number of samples (the accuracy) of both methods. I'm investigating why this happens.

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
