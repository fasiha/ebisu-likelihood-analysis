# See https://github.com/fasiha/ebisu/issues/35#issuecomment-899252582 for context and details.

## Setup suggestion

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
