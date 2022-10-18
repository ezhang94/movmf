"""Plotting utility functions"""
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import matplotlib.transforms as transforms

def draw_confidence_ellipse(mu, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    """Draw covariance confidence ellipse of 2D Gaussian given mean and covariance

    Modified from
      https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    Instead of passing in data points x and y, I directly pass in the mean
    and covariance matrix, because I have those already computed.

    Parameters
        mu[d]: Mean of data
        cov[d,d]: Covariance of data
        ax (matplotlib.axes.Axes): The axes object to draw the ellipse into.
        n_std (float or list): Number of standard deviations to plot,m via the 
            size of the ellipse's radiuses. If a sequence of values are provided,
            then plot ellipses for each value.
    
    Returns
        matplotlib.patches.Ellipse

    Other parameters
        kwargs : ~matplotlib.patches.Patch properties
    """

    assert mu.size == cov.shape[0] == cov.shape[1] == 2
    pearson = cov[0, 1]/onp.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using special case to obtain the eigenvalues of this two-dimensional dataset.
    # See derivation in Ex 4-3 at https://online.stat.psu.edu/stat505/lesson/4/4.5
    ell_radius_x = onp.sqrt(1 + pearson)
    ell_radius_y = onp.sqrt(1 - pearson)

    mean_x = mu[0]
    mean_y = mu[1]

    # Standard deviation is the square root of the respective diagonal variance
    # element. Multiply the desired number of standard deviations to scale axis.
    if isinstance(n_std, (int, float)):
        n_std = [n_std]

    for std in n_std:
        scale_x = onp.sqrt(cov[0, 0]) * std
        scale_y = onp.sqrt(cov[1, 1]) * std

        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                    facecolor=facecolor, **kwargs)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
    return