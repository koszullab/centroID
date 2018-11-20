__author__ = "hervemn"

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.stats import poisson
from scipy.stats import kurtosis
import scipy.ndimage.measurements as meas
from scipy.ndimage.interpolation import zoom as Zoom
from scipy.ndimage import center_of_mass
import scipy as scp
from violin_plot import violin_plot


from numpy import unravel_index
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2
    )


def moments(data, tickX):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.meshgrid(tickX, tickX)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(np.argmin(abs(y - tickX)))]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(np.argmin(abs(x - tickX))), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data, tickX):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data, tickX)
    grid = np.meshgrid(tickX, tickX)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*grid) - data)
    #    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
    #                                 data)
    p, success = optimize.leastsq(errorfunction, params)

    print("Fit centromere position (bp) = ", p[2])
    print("Fit std  x= ", p[3])
    return p


def gauss(x, *p):
    """
    :param x:
    :param p:
    :return:
    """
    A, B, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + B


def gauss_simple(x, *p):
    """
    :param x:
    :param p:
    :return:
    """
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def gaussian_fit_restricted(data_raw,):
    data = np.copy(data_raw)
    data_smooth_b = gaussian_filter(data, 10)
    B_e_x = data_smooth_b.mean()
    data_smooth_s = gaussian_filter(data, 1)
    data[data < B_e_x] = B_e_x
    data = data - B_e_x
    data[data < 0] = 0
    bin_centres_x = np.array(list(range(0, len(data_raw))))
    sub_bin_centres_x = np.linspace(0, len(data_raw), 2 * len(data_raw))
    A_e_x = data.max()
    Mu_e_x = np.argmax(gaussian_filter(data_smooth_s, 1))
    Sigma_e_x = np.abs(10)
    p0_x = np.array([Mu_e_x, Sigma_e_x])
    opt_fun = lambda x, *p: A_e_x * np.exp(-(x - p[0]) ** 2 / (2. * p[1] ** 2))
    try:
        coeff_x, var_matrix_x = curve_fit(opt_fun, bin_centres_x, data, p0=p0_x)
    except RuntimeError:
        print("optimal parameters not found")
        coeff_x = p0_x
    data_fit = opt_fun(sub_bin_centres_x, *coeff_x)
    print("Fit centromere position (bp) = ", coeff_x[0])
    print("Fit std  x= ", coeff_x[1])
    return coeff_x, data_fit, sub_bin_centres_x, data


def gmm_fit_kb(data_raw, init_sigma):
    data = np.copy(data_raw)
    data_smooth_b = gaussian_filter(data, 10)
    B_e_x = data_smooth_b.mean()
    data_smooth_s = gaussian_filter(data, 1)
    sort_indexes = data_smooth_s.argsort()
    data[data < B_e_x] = B_e_x
    data = data - B_e_x
    data[data < 0] = 0
    bin_centres_x = np.array(list(range(0, len(data_raw))))
    sub_bin_centres_x = np.linspace(0, len(data_raw), 2 * len(data_raw))

    Mu_e_x_1 = sort_indexes[0]
    A_e_x_1 = data[Mu_e_x_1]
    Sigma_e_x_1 = np.abs(init_sigma)

    Mu_e_x_2 = sort_indexes[1]
    A_e_x_2 = data[Mu_e_x_2]
    Sigma_e_x_2 = np.abs(init_sigma)

    p0_x = np.array([A_e_x_1, Mu_e_x_1, Sigma_e_x_1, A_e_x_2, Mu_e_x_2, Sigma_e_x_2])
    opt_fun = lambda x, *p: p[0] * np.exp(-(x - p[1]) ** 2 / (2. * p[2] ** 2)) + p[
        3
    ] * np.exp(-(x - p[4]) ** 2 / (2. * p[5] ** 2))
    try:
        coeff_x, var_matrix_x = curve_fit(opt_fun, bin_centres_x, data, p0=p0_x)
    except RuntimeError:
        print("optimal parameters not found")
        coeff_x = p0_x
    data_fit = opt_fun(sub_bin_centres_x, *coeff_x)
    print("Fit centromere position (bp) = ", (coeff_x[1] + coeff_x[4]) / 2.)
    print("Fit std 1 = ", coeff_x[2])
    print("Fit std 2 = ", coeff_x[5])
    return coeff_x, data_fit, sub_bin_centres_x, data


def center_of_mass_kb(data, tick, end_tick):
    max_tick = len(tick) - 1
    out = center_of_mass(data)
    index = out[0]
    pos_int = int(index)
    pos_float = index - pos_int
    pre_pos = tick[pos_int]
    if pos_int == max_tick:
        sub_pos = pre_pos + float(end_tick[-1] - pre_pos) * pos_float
    else:
        sub_pos = pre_pos + float(tick[pos_int + 1] - pre_pos) * pos_float
    return sub_pos


def gaussian_fit_kb(data_raw, tickX, sub_tickX, end_tick):
    data = np.copy(data_raw)
    index_extrem = np.any([tickX < 20000, tickX > tickX[-1] - 20000])
    if tickX[-1] - tickX[0] > 70000:
        data[index_extrem] = 0
    bin_centres_x = tickX
    A_e_x = data.max()
    B_e_x = data.min()
    Mu_e_x = tickX[np.argmax(data)]
    #    Mu_e_x = tickX[np.argmax(gaussian_filter(data, 1))]
    Sigma_e_x = np.abs(5000)
    p0_x = [A_e_x, B_e_x, Mu_e_x, Sigma_e_x]
    problem = False
    try:
        coeff_x, var_matrix_x = curve_fit(gauss, bin_centres_x, data, p0=p0_x)
    except RuntimeError:
        print("optimal parameters not found")
        problem = True
        coeff_x = p0_x
    # p0_x = [Mu_e_x, Sigma_e_x, A_e_x]
    # opt_fun = lambda x, mu, sigma, A : gauss(x, A, B_e_x, mu, sigma)
    # coeff_x, var_matrix_x = curve_fit(opt_fun, bin_centres_x, data, p0=p0_x)
    # data_fit = opt_fun(sub_tickX, coeff_x[0], coeff_x[1], coeff_x[2])
    data_fit = gauss(sub_tickX, *coeff_x)
    if coeff_x[2] > tickX[-1] or coeff_x[2] < tickX[0]:
        print(" centromere position out of bound")
        coeff_x[2] = Mu_e_x
        problem = True
    if problem:
        coeff_x[2] = center_of_mass_kb(data, tickX, end_tick)
    #    print 'Fit centromere position (bp) = ', coeff_x[2]
    #    print 'Fit std  x= ', coeff_x[3]
    return coeff_x, data_fit


def gaussian_fit_pxl(data_raw, tickX, sub_tickX):
    data = np.copy(data_raw)
    index_extrem = np.any([tickX < 20000, tickX > tickX[-1] - 20000])
    if tickX[-1] - tickX[0] > 70000:
        data[index_extrem] = 0
    bin_centres_x = list(range(0, len(tickX)))
    sub_bin_centres_x = np.linspace(0, len(sub_tickX), len(sub_tickX))
    A_e_x = data.max()
    B_e_x = data.min()
    Mu_e_x = np.argmax(gaussian_filter(data, 1))
    Sigma_e_x = np.abs(5)
    p0_x = [A_e_x, B_e_x, Mu_e_x, Sigma_e_x]
    try:
        coeff_x, var_matrix_x = curve_fit(gauss, bin_centres_x, data, p0=p0_x)
    except RuntimeError:
        print("optimal parameters not found")
        coeff_x = p0_x
    data_fit = gauss(sub_bin_centres_x, *coeff_x)
    #    print 'Fit centromere position (pixel) = ', coeff_x[2]
    #    print 'Fit std  x= ', coeff_x[3]
    return coeff_x, data_fit
