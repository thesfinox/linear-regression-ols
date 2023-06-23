# Simple Linear Regression

A simple example of Ordinary Least Squares (OLS) regression using the [statsmodels](https://www.statsmodels.org/stable/index.html) library.

## Installation

Use `python -m venv venv/` and `pip install -r requirements.txt`.

## Usage

You can control the **no. of samples** in the simulation, the **slope** of the simulation, the fit of the **intercept**.

e.g. 1:

```python
python linear.py --slope 3.4 --samples 100
```

e.g. 2:

```python
python linear.py --slope 6.5 --samples 15 --no-intercept
```

N.B.: you can use `python linear.py --help` for info.
