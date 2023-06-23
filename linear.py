# -*- coding: utf-8 -*-
"""
OLS example

A small example of linear regression using OLS regression.
"""
import argparse
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.nonparametric.smoothers_lowess import lowess

__author__ = 'Riccardo Finotello'
__email__ = 'riccardo.finotello@cea.fr'
__description__ = 'A small example of linear regression using OLS regression.'
__epilog__ = 'For bug reports and info: ' + __author__ + ' <' + __email__ + '>'


def main(args):

    # Simulate data with a linear relationship
    rnd = np.random.RandomState(args.seed)
    X = rnd.normal(size=(100, ))  # 100 samples (data)
    b = 3.5  # slope
    z = rnd.normal(size=(100, ))  # 100 samples (noise)
    y = b*X + z  # 100 samples (target)

    # Create a pandas dataframe
    features = ['x']
    label = ['y']
    df = pd.DataFrame(np.stack((X, y), axis=1), columns=features + label)

    # Fit the model
    # formula = 'y ~ x -1'  # y = b * x + z (do not fit the intercept)
    formula = 'y ~ x'  # y = b * x + z
    model = smf.ols(formula=formula, data=df)
    results = model.fit()

    print(results.summary())

    pred = results.get_prediction()
    infl = results.get_influence()

    # Complete the dataframe with the predictions
    df['y_true'] = b * df['x']
    df['y_pred'] = pred.predicted_mean
    df['obs_ci_upper'] = pred.summary_frame()['obs_ci_upper']
    df['obs_ci_lower'] = pred.summary_frame()['obs_ci_lower']
    df['mean_ci_upper'] = pred.summary_frame()['mean_ci_upper']
    df['mean_ci_lower'] = pred.summary_frame()['mean_ci_lower']
    df['residuals'] = results.resid
    df['residual_norm'] = infl.resid_studentized_internal
    df['leverage'] = infl.hat_matrix_diag
    df['cooks_d'] = infl.cooks_distance[0]

    # Sort by x for the plot
    df = df.sort_values(by='x')

    # Plot the results
    mpl.use('agg')
    mpl.style.use('ggplot')
    fig, ax = plt.subplots(ncols=3, figsize=(18, 6))

    # Plot the data
    ax[0].plot(df['x'], df['y'], 'bo', label='data')
    ax[0].plot(df['x'], df['y_pred'], 'r-', label='OLS')
    ax[0].plot(df['x'], df['y_true'], 'g-', label=rf'true: $y = {b} \cdot x$')
    ax[0].fill_between(df['x'],
                       df['obs_ci_lower'],
                       df['obs_ci_upper'],
                       color='r',
                       alpha=0.1)
    ax[0].fill_between(df['x'],
                       df['mean_ci_lower'],
                       df['mean_ci_upper'],
                       color='g',
                       alpha=0.1)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend(loc='best')
    ax[0].set_title('Linear regression')

    # Plot the residuals with the lowess curve
    x_low, y_low = lowess(df['residuals'], df['y_pred']).T
    r2 = np.corrcoef(df['y_pred'], df['residuals'])[0, 1]**2
    ax[1].plot(df['y_pred'], df['residuals'], 'bo', label='residuals')
    ax[1].plot(x_low, y_low, 'r-', label='lowess')
    ax[1].set_xlabel('predictions')
    ax[1].set_ylabel('residuals')
    ax[1].legend(loc='best')
    ax[1].set_title(rf'Residual plot ($R^2 = {r2:.2f}$)')

    # Check if the residuals are normally distributed using a Q-Q plot
    x = np.linspace(df['residual_norm'].min(), df['residual_norm'].max(), 100)
    qq = np.linspace(0.01, 0.99, 100)
    the = norm.ppf(qq)
    exp = np.percentile(df['residual_norm'], qq * 100)
    r2 = np.corrcoef(the, exp)[0, 1]**2
    ax[2].plot(the, exp, 'bo')
    ax[2].plot(x, x, 'r--')
    ax[2].set_xlabel('Theoretical quantiles')
    ax[2].set_ylabel('Sample quantiles')
    ax[2].set_title(rf'Q-Q plot ($R^2 = {r2:.2f}$))')

    plt.tight_layout()
    plt.savefig('linear.png')
    plt.close(fig)

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__description__,
                                     epilog=__epilog__)
    parser.add_argument('-b',
                        '--slope',
                        type=float,
                        default=3.5,
                        help='Slope of the linear relationship')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)
