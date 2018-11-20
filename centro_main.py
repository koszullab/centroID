__author__ = "hervemn"

# coding: utf-8
import math, sys, time
import pp
import pyramid as pyr
import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import sys, socket
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.stats import poisson
from scipy.stats import kurtosis
import scipy.ndimage.measurements as meas
from scipy.ndimage.interpolation import zoom as Zoom
from scipy.ndimage import center_of_mass
import scipy as scp
import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_laplace
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from numpy import unravel_index
from violin_plot import violin_plot
import parrallel_bootstrap as pb
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


def gaussian_fit_kb(data_raw, tickX, sub_tickX):
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
    try:
        coeff_x, var_matrix_x = curve_fit(gauss, bin_centres_x, data, p0=p0_x)
    except RuntimeError:
        print("optimal parameters not found")
        coeff_x = p0_x
    # p0_x = [Mu_e_x, Sigma_e_x, A_e_x]
    # opt_fun = lambda x, mu, sigma, A : gauss(x, A, B_e_x, mu, sigma)
    # coeff_x, var_matrix_x = curve_fit(opt_fun, bin_centres_x, data, p0=p0_x)
    # data_fit = opt_fun(sub_tickX, coeff_x[0], coeff_x[1], coeff_x[2])
    data_fit = gauss(sub_tickX, *coeff_x)
    if coeff_x[2] > tickX[-1] or coeff_x[2] < tickX[0]:
        print(" centromere position out of bound")
        coeff_x[2] = Mu_e_x
    print("Fit centromere position (bp) = ", coeff_x[2])
    print("Fit std  x= ", coeff_x[3])
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
    print("Fit centromere position (pixel) = ", coeff_x[2])
    print("Fit std  x= ", coeff_x[3])
    return coeff_x, data_fit


class level:
    def __init__(self, level, output_folder, is_max_level):
        self.level = level
        self.output_folder = os.path.join(output_folder, str(level))
        #        if not (os.path.exists(self.output_folder)):
        #            os.mkdir(self.output_folder)
        self.collect_bootstrap = dict()
        self.all_localization = dict()
        self.is_max_level = is_max_level

    def load_data(self, pyramid):
        """
        :param genome: genome
        """
        print("loading data from level = ", self.level)
        self.im_init = np.array(pyramid.data[str(self.level)], dtype=np.int32)
        self.n_frags = self.im_init.shape[0]
        # outIm_raw = Image.fromarray(self.im_init)
        # outIm_raw.save(os.path.join(self.output_folder, "raw_image.tiff"))
        self.dict_contigs = dict()
        ContFrags = pyramid.spec_level[str(self.level)]["contigs_dict"]
        coord_cont = dict()
        self.trans_free_im_init = np.empty_like(self.im_init)
        np.copyto(self.trans_free_im_init, self.im_init)
        self.distri_frag = []
        for id_cont in ContFrags:
            if not (id_cont == 17):
                self.dict_contigs[id_cont] = dict()
                cont_frag = ContFrags[id_cont]
                self.collect_bootstrap[id_cont] = dict()
                self.collect_bootstrap[id_cont]["pxl"] = []
                self.collect_bootstrap[id_cont]["kb"] = []
                coord_cont[id_cont] = []
                name_contig = cont_frag[0].init_contig
                tick_kb = []
                tick = list(range(0, len(cont_frag)))

                end_frags_kb = []
                print("loading fragments ...")
                print("n frags = ", len(cont_frag))
                for id_f in range(0, len(cont_frag)):
                    f = cont_frag[id_f]
                    lkb = f.length_kb
                    self.distri_frag.append(lkb)
                    start = f.start_pos
                    end = f.end_pos
                    tick_kb.append(start + lkb / 2.)
                    end_frags_kb.append(end)
                #                    tick_kb.append(end)
                #                    tick_kb.append(start)
                sub_tick_kb = np.linspace(0, tick_kb[-1], len(tick_kb) * 10)
                for f in cont_frag:
                    coord_cont[id_cont].append(f.np_id_abs - 1)
                self.dict_contigs[id_cont]["intra_coord"] = coord_cont[id_cont]
                self.dict_contigs[id_cont]["frags"] = cont_frag
                self.dict_contigs[id_cont]["name"] = name_contig
                self.dict_contigs[id_cont]["tick_kb"] = tick_kb
                self.dict_contigs[id_cont]["end_frags_kb"] = end_frags_kb
                self.dict_contigs[id_cont]["sub_tick_kb"] = sub_tick_kb
                self.dict_contigs[id_cont]["tick"] = tick
            #######################
        print("trans free matrix ...")
        total_trans = 0
        n_tot = 0
        for id_cont in range(1, len(self.dict_contigs) + 1):
            print("current chrom  = ", id_cont)
            if not (id_cont == 17):
                full = self.trans_free_im_init[coord_cont[id_cont], :]
                intra = self.trans_free_im_init[
                    np.ix_(coord_cont[id_cont], coord_cont[id_cont])
                ]
                total_trans += full.sum()
                total_trans -= intra.sum()
                n_tot += (full.shape[0] * full.shape[1]) - intra.shape[0] * intra.shape[
                    1
                ]
                # self.trans_free_im_init[np.ix_(coord_cont[id_cont], coord_cont[id_cont])] = -1

        self.mean_value_trans = total_trans / np.float32(n_tot)
        print("computing mean trans value ... = ", self.mean_value_trans)
        # self.mean_value_trans = self.trans_free_im_init[self.trans_free_im_init > -1].mean()
        print("putting random values ...")
        for id_cont in self.dict_contigs:
            if not (id_cont == 17):
                print("chromosome id = ", id_cont)
                if self.is_max_level:
                    mat_tmp = self.trans_free_im_init[
                        np.ix_(coord_cont[id_cont], coord_cont[id_cont])
                    ]
                    mat_tmp[:, :] = self.mean_value_trans
                    mat_tmp = np.random.poisson(mat_tmp)
                    self.trans_free_im_init[
                        np.ix_(coord_cont[id_cont], coord_cont[id_cont])
                    ] = mat_tmp
                else:
                    self.trans_free_im_init[
                        np.ix_(coord_cont[id_cont], coord_cont[id_cont])
                    ] = self.mean_value_trans

        self.distri_frag = np.array(self.distri_frag)
        self.init_data()
        self.n_contigs = len(self.dict_contigs)

    def init_data(self,):
        print("init data ", self.level)
        tmp = np.empty_like(self.im_init)
        np.copyto(tmp, self.im_init)
        tmp_trans_free = np.empty_like(self.trans_free_im_init)
        np.copyto(tmp_trans_free, self.trans_free_im_init)

        tmp = np.float32(tmp)
        tmp_trans_free = np.float32(tmp_trans_free)
        #        self.im_curr = self.perform_scn(tmp,10)
        #        self.im_trans_free_curr = self.perform_scn(tmp_trans_free,10)
        self.im_curr = tmp
        self.im_trans_free_curr = tmp_trans_free

        #        self.im_curr = self.im_curr / self.im_curr.mean()
        #        self.im_trans_free_curr = self.im_trans_free_curr / self.im_trans_free_curr.mean()
        #        factor_my = 200

        #        factor_my = 1/self.im_trans_free_curr.mean()
        #        self.im_curr = self.im_curr * factor_my
        #        self.im_trans_free_curr = self.im_trans_free_curr * factor_my

        if self.is_max_level:
            self.corr_im = np.corrcoef(self.im_trans_free_curr)
        else:
            self.corr_im = self.im_trans_free_curr

        self.extract_matrix_contig()

    def perform_scn(self, mat, n_iter):
        """
        :param mat: initial matrix
        :param n iter: number of iterations
        return: normalized matrix
        """
        new_mat = np.empty_like(mat)
        np.copyto(new_mat, mat)
        #    print " is there any nan : ", np.any(np.isnan(new_mat.sum(axis=1)))
        #    print " is there any inf : ", np.any(np.isinf(new_mat.sum(axis=1)))
        new_mat = np.float32(new_mat)
        for k in range(0, n_iter):
            print("iteration = ", k)
            v_norm = new_mat.sum(axis=1)
            #            print "minimum norm = ", v_norm.min()
            new_mat = new_mat.T / v_norm

        #        print " is there any nan : ", np.any(np.isnan(new_mat.sum(axis=1)))
        #        print " is there any inf : ", np.any(np.isinf(new_mat.sum(axis=1)))

        return new_mat

    def update_data_scn(self,):
        print("init data for scn.. level = ", self.level)
        tmp = np.random.poisson(self.im_init)
        tmp_trans_free = np.random.poisson(self.trans_free_im_init)

        tmp = np.float32(tmp)
        tmp_trans_free = np.float32(tmp_trans_free)
        self.im_curr = self.perform_scn(tmp, 10)
        self.im_trans_free_curr = self.perform_scn(tmp_trans_free, 10)

        #        self.im_curr = self.im_curr * self.im_init.sum(axis=1)
        #        self.im_trans_free_curr = self.im_trans_free_curr * self.trans_free_im_init.sum(axis=1)

        if self.is_max_level:
            self.corr_im = np.corrcoef(self.im_trans_free_curr)
        else:
            self.corr_im = self.im_trans_free_curr
        self.extract_matrix_contig()

    def update_data(self,):
        print("init data .. level = ", self.level)
        tmp = np.random.poisson(self.im_init)
        tmp_trans_free = np.random.poisson(self.trans_free_im_init)

        tmp = np.float32(tmp)
        tmp_trans_free = np.float32(tmp_trans_free)
        self.im_curr = tmp
        self.im_trans_free_curr = tmp_trans_free

        #        self.im_curr = self.im_curr * self.im_init.sum(axis=1)
        #        self.im_trans_free_curr = self.im_trans_free_curr * self.trans_free_im_init.sum(axis=1)

        if self.is_max_level:
            self.corr_im = np.corrcoef(self.im_trans_free_curr)
        else:
            self.corr_im = self.im_trans_free_curr
        self.extract_matrix_contig()

    def dist_bp_2_frag(self, dist_bp):
        mean_length_bp = self.distri_frag.mean()
        output_pxl = int(dist_bp / mean_length_bp)
        return output_pxl

    def extract_matrix_contig(self):
        """

        """
        self.all_data = dict()
        print("sub matrix extraction ...")
        for id_cont in self.dict_contigs:
            print("id chrom = ", id_cont)
            if not (id_cont == 17):
                self.all_data[id_cont] = dict()
                coord_intra = self.dict_contigs[id_cont]["intra_coord"]
                full_raw_matrix = self.im_trans_free_curr[coord_intra, :]
                intra_raw_matrix = self.im_curr[np.ix_(coord_intra, coord_intra)]

                inter_raw_matrix = full_raw_matrix
                #                inter_raw_matrix[:, coord_intra] = 0

                full_corr_matrix = self.corr_im[coord_intra, :]
                intra_corr_matrix = self.corr_im[np.ix_(coord_intra, coord_intra)]
                inter_corr_matrix = full_corr_matrix

                self.all_data[id_cont]["full_raw_matrix"] = full_raw_matrix
                self.all_data[id_cont]["intra_raw_matrix"] = intra_raw_matrix
                self.all_data[id_cont]["inter_raw_matrix"] = inter_raw_matrix

                self.all_data[id_cont]["full_corr_matrix"] = full_corr_matrix
                self.all_data[id_cont]["intra_corr_matrix"] = intra_corr_matrix
                self.all_data[id_cont]["inter_corr_matrix"] = inter_corr_matrix

        for id_cont_X in self.dict_contigs:
            if not (id_cont_X == 17):
                self.all_data[id_cont_X]["sub_matrices"] = dict()
                full_corr_matrix = self.all_data[id_cont_X]["full_corr_matrix"]
                full_raw_matrix = self.all_data[id_cont_X]["full_raw_matrix"]
                for id_cont_Y in self.dict_contigs:
                    if not (id_cont_Y == 17):
                        coord_cont_Y = self.dict_contigs[id_cont_Y]["intra_coord"]
                        sub_mat_corr = full_corr_matrix[:, coord_cont_Y]
                        sub_mat_raw = full_raw_matrix[:, coord_cont_Y]
                        self.all_data[id_cont_X]["sub_matrices"][id_cont_Y] = dict()
                        self.all_data[id_cont_X]["sub_matrices"][id_cont_Y][
                            "raw"
                        ] = sub_mat_raw
                        self.all_data[id_cont_X]["sub_matrices"][id_cont_Y][
                            "corr"
                        ] = sub_mat_corr

    def eco_extract_matrix_contig(self):
        """
        :param index_contig: index of the contig in self.ContigS.current_contig
        """
        self.all_data = dict()
        print("sub matrix extraction ...")
        for id_cont in self.dict_contigs:
            print("id chrom = ", id_cont)
            if not (id_cont == 17):
                self.all_data[id_cont] = dict()
                coord_intra = self.dict_contigs[id_cont]["intra_coord"]
                full_raw_matrix = np.copy(self.im_trans_free_curr[coord_intra, :])
                #                full_raw_matrix = np.copy(self.im_trans_free_curr[coord_intra, :])
                intra_raw_matrix = np.copy(
                    self.im_curr[np.ix_(coord_intra, coord_intra)]
                )

                inter_raw_matrix = np.copy(full_raw_matrix)
                inter_raw_matrix[:, coord_intra] = 0

                full_corr_matrix = np.copy(self.corr_im[coord_intra, :])
                intra_corr_matrix = np.copy(
                    self.corr_im[np.ix_(coord_intra, coord_intra)]
                )
                inter_corr_matrix = np.copy(full_corr_matrix)

                self.all_data[id_cont]["full_raw_matrix"] = full_raw_matrix
                self.all_data[id_cont]["intra_raw_matrix"] = intra_raw_matrix
                self.all_data[id_cont]["inter_raw_matrix"] = inter_raw_matrix

                self.all_data[id_cont]["full_corr_matrix"] = full_corr_matrix
                self.all_data[id_cont]["intra_corr_matrix"] = intra_corr_matrix
                self.all_data[id_cont]["inter_corr_matrix"] = inter_corr_matrix

        for id_cont_X in self.dict_contigs:
            if not (id_cont_X == 17):
                self.all_data[id_cont_X]["sub_matrices"] = dict()
                full_corr_matrix = self.all_data[id_cont_X]["full_corr_matrix"]
                full_raw_matrix = self.all_data[id_cont_X]["full_raw_matrix"]
                for id_cont_Y in self.dict_contigs:
                    if not (id_cont_Y == 17):
                        coord_cont_Y = self.dict_contigs[id_cont_Y]["intra_coord"]
                        sub_mat_corr = full_corr_matrix[:, coord_cont_Y]
                        sub_mat_raw = full_raw_matrix[:, coord_cont_Y]
                        self.all_data[id_cont_X]["sub_matrices"][id_cont_Y] = dict()
                        self.all_data[id_cont_X]["sub_matrices"][id_cont_Y][
                            "raw"
                        ] = sub_mat_raw
                        self.all_data[id_cont_X]["sub_matrices"][id_cont_Y][
                            "corr"
                        ] = sub_mat_corr


class analysis:
    def __init__(self, pyramid_hic, output_folder, n_levels, start):
        self.pyramid = pyramid_hic
        self.level = dict()
        self.max_level = n_levels - 1
        # start = 1
        self.all_res_folder = os.path.join(output_folder, "all_res")
        if not (os.path.exists(self.all_res_folder)):
            os.mkdir(self.all_res_folder)

        for lev in range(start, n_levels):
            is_max_level = lev == self.max_level
            self.level[lev] = level(lev, output_folder, is_max_level)

        for lev in list(self.level.keys()):
            self.level[lev].load_data(self.pyramid)

        self.ground_truth = {
            1: (151465, 151582),
            2: (238207, 238323),
            3: (114385, 114501),
            4: (449711, 449821),
            5: (151987, 152104),
            6: (148510, 148627),
            7: (496920, 497038),
            8: (105586, 105703),
            9: (355629, 355745),
            10: (436307, 436425),
            11: (440129, 440246),
            12: (150828, 150947),
            13: (268031, 268149),
            14: (628758, 628875),
            15: (326584, 326702),
            16: (555957, 556073),
        }

    def bp_coord_2_pxl_coord(self, pos_bp, id_chrom, id_level):
        """


        :param id_chrom:
        :param pos_bp: genomic coordinate
        :param id_level: pyramid level
        """
        level_pyr = self.level[id_level]
        chrom = level_pyr.dict_contigs[id_chrom]
        end_frags_kb = np.array(chrom["end_frags_kb"])
        try:
            pos_pxl = np.where(pos_bp < end_frags_kb)[0][0]
        except IndexError:
            print("position out of chromosome")
            pos_pxl = len(end_frags_kb) - 1
        return pos_pxl

    #
    # def compute_distribution_contacts(self):
    #     """
    #
    #     """
    #     lev0 = self.level[0]
    #     list_chrom = lev0.dict_contigs.keys()
    #     for id_chrom in xrange(1, len(list_chrom) + 1):

    def bootstrap(self, id_level, delta_bp, size_filter_bp, N):
        """
        id_level
        """
        level_pyr = self.level[id_level]
        for i in range(0, N):
            print("##########################")
            print("bootstrap iteration = ", i)
            level_pyr.update_data_scn()
            self.refine_localization_4_boot(id_level, delta_bp, size_filter_bp)
            print("##########################")

    def single_bootstrap(self, id_level, delta_bp, size_filter_bp):
        """
        id_level
        """
        level_pyr = self.level[id_level]
        level_pyr.update_data_scn()
        output = self.refine_localization_4_boot_multi_proc(
            id_level, delta_bp, size_filter_bp
        )
        return output

    def analyse_bootstrap(self, id_level, ground_truth):
        level_pyr = self.level[id_level]
        n_contigs = level_pyr.n_contigs
        if n_contigs == 17:
            n_contigs = n_contigs - 1
        if np.floor(np.sqrt(n_contigs)) == np.sqrt(n_contigs):
            plusOne = 0
        else:
            plusOne = 1
        sizX = int(np.round(np.sqrt(n_contigs)) + plusOne)
        sizY = int(np.floor(np.sqrt(n_contigs)))
        fig = plt.figure(1, figsize=(10, 10))
        gs0 = gridspec.GridSpec(sizX, sizY)
        gs0.update(wspace=0.5, hspace=0.5)
        for id_chr in list(level_pyr.dict_contigs.keys()):
            pos_chr = np.int32(id_chr)
            ind_x = int((pos_chr - 1) / sizX)
            ind_y = (pos_chr - 1) - ind_x * sizX
            curr_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs0[ind_y, ind_x]
            )
            self.plot_results(level_pyr, id_chr, fig, curr_gs, ground_truth)
        plt.draw()
        fig.savefig(
            os.path.join(
                level_pyr.output_folder, "boot_strap_analysis_" + str(id_level) + ".pdf"
            )
        )
        plt.close()

    def old_plot_results(self, level_pyr, id_chr, fig, curr_gs, ground_truth):
        name_chrom = level_pyr.dict_contigs[id_chr]["name"]
        im_corr = level_pyr.all_data[id_chr]["sub_matrices"][id_chr]["corr"]
        axImCorr = plt.Subplot(fig, curr_gs[1, 0])
        fig.add_subplot(axImCorr)
        axBStrapCorr = plt.Subplot(fig, curr_gs[0, 0])
        fig.add_subplot(axBStrapCorr)
        tick_X = level_pyr.dict_contigs[id_chr]["tick_kb"]
        pos_bootstrap = level_pyr.collect_bootstrap[id_chr]["kb"]
        pos_bootstrap = np.array(pos_bootstrap)
        nbins = len(pos_bootstrap)
        axBStrapCorr.hist(
            pos_bootstrap, facecolor="b", edgecolor="b", linewidth=0, alpha=0.5
        )
        if ground_truth:
            ground_truth = self.ground_truth[id_chr]
            axBStrapCorr.axvspan(
                ground_truth[0], ground_truth[1], edgecolor="r", facecolor="r", alpha=1
            )
            pos_centro = (ground_truth[0] + ground_truth[1]) / 2.
            axImCorr.plot(pos_centro, pos_centro, "*b", markersize=7)

        pos_centro_boot = pos_bootstrap.mean()
        axImCorr.plot(pos_centro_boot, pos_centro_boot, "*w", markersize=7)
        # plt.setp(rect_ld, facecolor='b', edgecolor='b', linewidth=0, alpha=0.7)
        max_bound = tick_X[-1]
        min_bound = tick_X[0]

        axBStrapCorr.set_xlim([ground_truth[0] - 20000, ground_truth[1] + 20000])

        axImCorr.imshow(
            im_corr,
            interpolation="nearest",
            extent=[min_bound, max_bound, min_bound, max_bound],
            origin="lower",
        )

        axImCorr.set_xlim([min_bound, max_bound])
        axImCorr.set_ylim([min_bound, max_bound])
        # plt.draw()

    def refine_localization_4_boot(self, id_level, delta_bp, size_filter_bp):
        """

        :param id_level:
        :param delta:
        """

        level_pyr = self.level[id_level]
        is_max_level = level_pyr.is_max_level
        delta = level_pyr.dist_bp_2_frag(delta_bp)
        print("level pyramid = ", id_level)
        print("delta = ", delta)
        size_filter = level_pyr.dist_bp_2_frag(size_filter_bp)

        for chrom_id_i in self.list_id_chrom:
            data_i = level_pyr.all_data[chrom_id_i]["sub_matrices"]
            pre_pos_bp_i = self.detection[chrom_id_i]["pre_detect_pos"]
            pre_pos_pxl_i = self.bp_coord_2_pxl_coord(
                pre_pos_bp_i, chrom_id_i, id_level
            )
            cumul_mat = np.zeros((delta * 2 + 1, delta * 2 + 1))
            print("shape cumul mat = ", cumul_mat.shape)
            print("pos centro i = ", pre_pos_pxl_i)
            for chrom_id_j in self.list_id_chrom:
                if chrom_id_i != chrom_id_j:
                    pre_pos_bp_j = self.detection[chrom_id_j]["pre_detect_pos"]
                    pre_pos_pxl_j = self.bp_coord_2_pxl_coord(
                        pre_pos_bp_j, chrom_id_j, id_level
                    )
                    raw_sub = data_i[chrom_id_j]["raw"]
                    print("shape raw sup = ", raw_sub.shape)
                    print("pos centro j = ", pre_pos_pxl_j)

                    tmp = raw_sub[
                        pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1,
                        pre_pos_pxl_j - delta : pre_pos_pxl_j + delta + 1,
                    ]
                    print("shape tmp =", tmp.shape)
                    cumul_mat = cumul_mat + tmp

            base_contact = cumul_mat.min() + cumul_mat.std()
            # base_contact = cumul_mat.min() + cumul_mat.std()
            cumul_mat = cumul_mat / base_contact
            cumul_mat = gaussian_filter(
                cumul_mat, size_filter, mode="constant", cval=0.0
            )
            #                cumul_mat = gaussian_filter(cumul_mat, size_filter, mode="constant", cval=0.0)

            self.detection[chrom_id_i][id_level]["cumul_mat"] = cumul_mat

            tick_kb = np.array(level_pyr.dict_contigs[chrom_id_i]["tick_kb"])
            end_tick_kb = np.array(level_pyr.dict_contigs[chrom_id_i]["end_frags_kb"])

            sub_tick_kb = np.array(level_pyr.dict_contigs[chrom_id_i]["sub_tick_kb"])
            tick = tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]
            sub_tick = sub_tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]
            end_tick = end_tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]

            subTick = sub_tick_kb[
                np.all([sub_tick_kb >= tick[0], sub_tick_kb <= tick[-1]], axis=0)
            ]

            #            data_fit = cumul_mat.max(axis=1)
            #                data_fit = cumul_mat.max((1)) * cumul_mat.std((1))
            #            pos_max = np.argmax(data_fit)
            ########################################################################################################################
            #            tmp_corr = np.corrcoef(cumul_mat)
            #            mon_delta = 9
            #            pos_kb, out_position_kb, score, data_fit_kb, sub_pixel_pos_pxl,out_position_pxl, \
            #            data_fit_pxl = self.estimate_location_bloc_corr(tmp_corr, mon_delta, tick, sub_tick)
            ########################################################################################################################
            #            if not is_max_level:
            #                print "pos max = ", pos_max
            #                if pos_max>3:
            #                    mon_delta = 3
            #                else:
            #                    mon_delta = 2
            #                    print "modif delta "
            #                le_zero = pos_max - mon_delta
            #                sous_bloc  = cumul_mat[pos_max - mon_delta: pos_max + mon_delta + 1, pos_max - mon_delta: pos_max + mon_delta + 1]
            #                print "shape sous bloc = ", sous_bloc.shape
            #                pox, poy = center_of_mass(sous_bloc)
            #                pos_pxl = le_zero + pox
            #                print "pos pxl = ", pos_pxl
            #                floor_pxl = int(np.floor(pos_pxl))
            #                dec_kb = pos_pxl - floor_pxl
            #                if floor_pxl < len(tick) - 1:
            #                    inter_dist = tick[floor_pxl + 1] - tick[floor_pxl]
            #                else:
            #                    inter_dist = end_tick[floor_pxl] - tick[floor_pxl]
            #                sub_pix_loc = (inter_dist) * dec_kb
            #                pos_kb = tick[floor_pxl] + sub_pix_loc
            #            else:
            #                data_to_fit = cumul_mat.sum((1)) * cumul_mat.std((1))
            #                data_to_fit = (cumul_mat /cumul_mat.max()) * tick.mean()
            #                coeff_x, data_fit = gaussian_fit_kb(data_to_fit, tick, subTick)
            #                pos_kb = coeff_x[2]
            ########################################################################################################################
            cumul_mat = cumul_mat - gaussian_filter(
                cumul_mat, delta, mode="constant", cval=0.0
            )
            cumul_mat[cumul_mat < 0] = 0
            cumul_mat = gaussian_filter(
                cumul_mat, size_filter, mode="constant", cval=0.0
            )
            cumul_mat_thresh = np.copy(cumul_mat)
            thresh = 0.35
            cumul_mat_thresh[cumul_mat_thresh < cumul_mat_thresh.max() * thresh] = (
                cumul_mat_thresh.max() * thresh
            )
            data_to_fit = cumul_mat_thresh.max(axis=1)
            #            data_to_fit = cumul_mat_thresh.max(axis=1) * cumul_mat_thresh.std(axis=1)
            coeff_x, data_fit = gaussian_fit_kb(data_to_fit, tick, subTick)
            #            data_to_fit = cumul_mat_thresh
            #            coeff_x = fitgaussian(data_to_fit,tick)
            pos_kb = coeff_x[2]
            ########################################################################################################################
            #            cumul_mat = cumul_mat - gaussian_filter(cumul_mat,delta, mode="constant", cval=0.0)
            #            cumul_mat[cumul_mat<0] = 0
            #            cumul_mat = gaussian_filter(cumul_mat,1,mode="constant", cval=0.0)
            #            cumul_mat_thresh = np.copy(cumul_mat)
            #            cumul_mat_thresh[cumul_mat_thresh<cumul_mat_thresh.max()*0.65] = 0
            #            data_to_fit = np.corrcoef(cumul_mat_thresh)
            #            mon_delta = 9
            #            pos_kb, out_position_kb, score, data_fit_kb, sub_pixel_pos_pxl,out_position_pxl, \
            #            data_fit_pxl = self.estimate_location_bloc_corr(data_to_fit, mon_delta, tick, sub_tick)
            ########################################################################################################################
            #                tick_kb_spe = np.arange(0,len(tick))
            #                coeff_x = fitgaussian(cumul_mat,tick_kb_spe)
            #                pre_pos_kb = coeff_x[2]
            #
            #                floor_kb = int(np.floor(pre_pos_kb))
            #                dec_kb = pre_pos_kb - floor_kb
            #                sub_pix_loc = (end_tick[floor_kb] - tick[floor_kb]) * dec_kb
            #                pos_kb = tick[floor_kb] + sub_pix_loc
            ########################################################################################################################
            self.detection[chrom_id_i][id_level]["bootstrap"].append(pos_kb)
            self.detection[chrom_id_i][id_level]["cumul_mat"] = cumul_mat

    def refine_localization_4_boot_multi_proc(self, id_level, delta_bp, size_filter_bp):
        """

        :param id_level:
        :param delta:
        """

        level_pyr = self.level[id_level]
        is_max_level = level_pyr.is_max_level
        delta = level_pyr.dist_bp_2_frag(delta_bp)
        print("level pyramid = ", id_level)
        print("delta = ", delta)
        size_filter = level_pyr.dist_bp_2_frag(size_filter_bp)
        detection = dict()
        for chrom_id_i in self.list_id_chrom:
            detection[chrom_id_i] = dict()
            data_i = level_pyr.all_data[chrom_id_i]["sub_matrices"]
            pre_pos_bp_i = self.detection[chrom_id_i]["pre_detect_pos"]
            pre_pos_pxl_i = self.bp_coord_2_pxl_coord(
                pre_pos_bp_i, chrom_id_i, id_level
            )
            cumul_mat = np.zeros((delta * 2 + 1, delta * 2 + 1))
            print("shape cumul mat = ", cumul_mat.shape)
            print("pos centro i = ", pre_pos_pxl_i)
            for chrom_id_j in self.list_id_chrom:
                if chrom_id_i != chrom_id_j:
                    pre_pos_bp_j = self.detection[chrom_id_j]["pre_detect_pos"]
                    pre_pos_pxl_j = self.bp_coord_2_pxl_coord(
                        pre_pos_bp_j, chrom_id_j, id_level
                    )
                    raw_sub = data_i[chrom_id_j]["raw"]
                    print("shape raw sup = ", raw_sub.shape)
                    print("pos centro j = ", pre_pos_pxl_j)

                    tmp = raw_sub[
                        pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1,
                        pre_pos_pxl_j - delta : pre_pos_pxl_j + delta + 1,
                    ]
                    print("shape tmp =", tmp.shape)
                    cumul_mat = cumul_mat + tmp

            base_contact = cumul_mat.min() + cumul_mat.std()
            # base_contact = cumul_mat.min() + cumul_mat.std()
            cumul_mat = cumul_mat / base_contact
            cumul_mat = gaussian_filter(
                cumul_mat, size_filter, mode="constant", cval=0.0
            )
            #                cumul_mat = gaussian_filter(cumul_mat, size_filter, mode="constant", cval=0.0)

            tick_kb = np.array(level_pyr.dict_contigs[chrom_id_i]["tick_kb"])
            end_tick_kb = np.array(level_pyr.dict_contigs[chrom_id_i]["end_frags_kb"])

            sub_tick_kb = np.array(level_pyr.dict_contigs[chrom_id_i]["sub_tick_kb"])
            tick = tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]
            sub_tick = sub_tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]
            end_tick = end_tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]

            subTick = sub_tick_kb[
                np.all([sub_tick_kb >= tick[0], sub_tick_kb <= tick[-1]], axis=0)
            ]

            #            data_fit = cumul_mat.max(axis=1)
            #                data_fit = cumul_mat.max((1)) * cumul_mat.std((1))
            #            pos_max = np.argmax(data_fit)
            ########################################################################################################################
            tmp_corr = np.corrcoef(cumul_mat)
            mon_delta = 9
            pos_kb, out_position_kb, score, data_fit_kb, sub_pixel_pos_pxl, out_position_pxl, data_fit_pxl = self.estimate_location_bloc_corr(
                tmp_corr, mon_delta, tick, sub_tick
            )
            ########################################################################################################################
            #            if not is_max_level:
            #                print "pos max = ", pos_max
            #                if pos_max>3:
            #                    mon_delta = 3
            #                else:
            #                    mon_delta = 2
            #                    print "modif delta "
            #                le_zero = pos_max - mon_delta
            #                sous_bloc  = cumul_mat[pos_max - mon_delta: pos_max + mon_delta + 1, pos_max - mon_delta: pos_max + mon_delta + 1]
            #                print "shape sous bloc = ", sous_bloc.shape
            #                pox, poy = center_of_mass(sous_bloc)
            #                pos_pxl = le_zero + pox
            #                print "pos pxl = ", pos_pxl
            #                floor_pxl = int(np.floor(pos_pxl))
            #                dec_kb = pos_pxl - floor_pxl
            #                if floor_pxl < len(tick) - 1:
            #                    inter_dist = tick[floor_pxl + 1] - tick[floor_pxl]
            #                else:
            #                    inter_dist = end_tick[floor_pxl] - tick[floor_pxl]
            #                sub_pix_loc = (inter_dist) * dec_kb
            #                pos_kb = tick[floor_pxl] + sub_pix_loc
            ########################################################################################################################
            #            else:
            #                data_to_fit = cumul_mat.sum((1)) * cumul_mat.std((1))
            #                coeff_x, data_fit = gaussian_fit_kb(data_to_fit, tick, subTick)
            #                data_to_fit = (cumul_mat /cumul_mat.max()) * tick.mean()
            #                coeff_x = fitgaussian(data_to_fit,tick)
            #                pos_kb = coeff_x[2]
            ########################################################################################################################
            #                tick_kb_spe = np.arange(0,len(tick))
            #                coeff_x = fitgaussian(cumul_mat,tick_kb_spe)
            #                pre_pos_kb = coeff_x[2]
            #
            #                floor_kb = int(np.floor(pre_pos_kb))
            #                dec_kb = pre_pos_kb - floor_kb
            #                sub_pix_loc = (end_tick[floor_kb] - tick[floor_kb]) * dec_kb
            #                pos_kb = tick[floor_kb] + sub_pix_loc
            ########################################################################################################################
            detection[chrom_id_i] = pos_kb
            return detection

    def post_detection(self, id_level, delta, N_samples, size_filter, is_cerevisiae):

        level_pyr = self.level[id_level]
        for id_chrom_i in list(level_pyr.dict_contigs.keys()):
            name_contig = level_pyr.dict_contigs[id_chrom_i]["name"]
            f, axarr = plt.subplots(2, 2, figsize=(10, 10))
            cumul_mat = np.zeros((delta * 2 + 1, delta * 2 + 1))
            data_i = level_pyr.all_data[id_chrom_i]["sub_matrices"]
            imCorr = data_i[id_chrom_i]["corr"]
            pos_centro_i = round(
                level_pyr.all_localization[id_chrom_i]["pxl"][id_chrom_i][1].mean()
            )
            axImCorr = axarr[0, 0]
            axImCCorr = axarr[0, 1]
            axBootStrap = axarr[1, 1]
            axCumSub = axarr[1, 0]
            axImCorr.imshow(imCorr, interpolation="nearest", origin="lower")
            axImCorr.plot(pos_centro_i, pos_centro_i, "*w", markersize=9)
            axImCorr.set_xlim([0, imCorr.shape[1]])
            axImCorr.set_ylim([0, imCorr.shape[1]])
            axImCCorr.imshow(
                np.corrcoef(imCorr), interpolation="nearest", origin="lower"
            )
            axImCorr.set_title("Correlation Matrix")
            axImCCorr.set_title("Correlation of Correlation Matrix")
            axImCCorr.set_xlim([0, imCorr.shape[1]])
            axImCCorr.set_ylim([0, imCorr.shape[1]])
            for id_chrom_j in list(level_pyr.dict_contigs.keys()):
                if id_chrom_i != id_chrom_j:
                    raw_sub = data_i[id_chrom_j]["raw"]
                    pos_centro_j = round(
                        level_pyr.all_localization[id_chrom_j]["pxl"][id_chrom_j][
                            1
                        ].mean()
                    )
                    cumul_mat = (
                        cumul_mat
                        + raw_sub[
                            pos_centro_i - delta : pos_centro_i + delta + 1,
                            pos_centro_j - delta : pos_centro_j + delta + 1,
                        ]
                    )

            cumul_mat = gaussian_filter(cumul_mat, size_filter)
            # cumul_contact_sum = cumul_mat.sum((1))

            cumul_contact_max = cumul_mat.max((1))
            axCumSub.imshow(cumul_mat, interpolation="nearest")

            tick_kb = np.array(level_pyr.dict_contigs[id_chrom_i]["tick_kb"])
            sub_tick_kb = np.array(level_pyr.dict_contigs[id_chrom_i]["sub_tick_kb"])
            tick = tick_kb[pos_centro_i - delta : pos_centro_i + delta + 1]
            subTick = sub_tick_kb[
                np.all([sub_tick_kb >= tick[0], sub_tick_kb <= tick[-1]], axis=0)
            ]
            collect_pos = []
            for i in range(0, N_samples):
                data_boot = np.random.poisson(cumul_contact_max)
                coeff_x, data_fit = gaussian_fit_kb(data_boot, tick, subTick)
                pos_kb = coeff_x[2]
                collect_pos.append(pos_kb)
            collect_pos = np.array(collect_pos)
            axBootStrap.hist(
                collect_pos, facecolor="b", edgecolor="b", linewidth=0, alpha=0.5
            )
            precision = collect_pos.std()
            mean_position = collect_pos.mean()

            if is_cerevisiae:
                ground_truth = np.array(self.ground_truth[id_chrom_i])
                error_loc = np.min(np.abs(ground_truth - mean_position))
                axBootStrap.axvspan(
                    ground_truth[0],
                    ground_truth[1],
                    edgecolor="r",
                    facecolor="r",
                    alpha=1,
                )
                axBootStrap.set_xlim([ground_truth[0] - 10000, ground_truth[1] + 10000])
                axBootStrap.text(
                    0.8,
                    0.90,
                    "mean error(bp) = " + str(np.around(error_loc, 3)),
                    fontsize=7,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axBootStrap.transAxes,
                )
                axBootStrap.legend(
                    ["ground truth", "bootstrap estimates"],
                    loc="upper left",
                    prop={"size": 5},
                )
                axBootStrap.set_title("Bootstrap distribution and ground truth")
            else:
                axBootStrap.axvspan(
                    mean_position - 200,
                    mean_position + 200,
                    edgecolor="r",
                    facecolor="r",
                    alpha=1,
                )
                axBootStrap.set_title("Bootstrap distribution")
                axBootStrap.legend(
                    ["mean position", "bootstrap estimates"],
                    loc="upper left",
                    prop={"size": 6},
                )
                axBootStrap.text(
                    0.8,
                    0.90,
                    "mean position(bp) = " + str(np.around(mean_position, 3)),
                    fontsize=7,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axBootStrap.transAxes,
                )
                axBootStrap.set_xlim([mean_position - 10000, mean_position + 10000])
            axBootStrap.text(
                0.8,
                0.95,
                "precision(bp) = " + str(np.around(precision, 3)),
                fontsize=7,
                horizontalalignment="center",
                verticalalignment="center",
                transform=axBootStrap.transAxes,
            )

            plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
            plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
            plt.suptitle(name_contig + ": detection centromere")
            plt.draw()
            f.savefig(
                os.path.join(
                    level_pyr.output_folder, "localization_" + name_contig + ".pdf"
                )
            )

    def pre_detection(self, delta_bp):

        ultimate_pyr = self.level[self.max_level]
        delta_pxl = ultimate_pyr.dist_bp_2_frag(delta_bp)
        self.pre_detection_level(self.max_level, delta_pxl)
        self.detection = dict()
        self.list_id_chrom = list(ultimate_pyr.dict_contigs.keys())
        for chrom_id in self.list_id_chrom:
            self.detection[chrom_id] = dict()
            self.detection[chrom_id]["name"] = ultimate_pyr.dict_contigs[chrom_id][
                "name"
            ]
            for id_level in list(self.level.keys()):
                self.detection[chrom_id][id_level] = dict()
                self.detection[chrom_id][id_level]["bootstrap"] = []
            self.detection[chrom_id]["pre_detect_pos"] = ultimate_pyr.all_localization[
                chrom_id
            ]["kb"][chrom_id][0]

    def refine_localization(self, delta_bp, size_filter_bp, N_samples):
        """

        :param delta:
        """
        keys_level = list(self.level.keys())
        fact_mult_mat = 200
        for id_level in keys_level:

            level_pyr = self.level[id_level]
            delta = level_pyr.dist_bp_2_frag(delta_bp)
            print("level pyramid = ", id_level)
            print("delta = ", delta)
            size_filter = level_pyr.dist_bp_2_frag(size_filter_bp)

            for chrom_id_i in self.list_id_chrom:
                name_chrom_i = self.detection[chrom_id_i]["name"]
                data_i = level_pyr.all_data[chrom_id_i]["sub_matrices"]
                pre_pos_bp_i = self.detection[chrom_id_i]["pre_detect_pos"]
                full_inter_mat = level_pyr.all_data[chrom_id_i]["inter_raw_matrix"]

                pre_pos_pxl_i = self.bp_coord_2_pxl_coord(
                    pre_pos_bp_i, chrom_id_i, id_level
                )
                sub_inter_mat = full_inter_mat[
                    pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1, :
                ]
                sub_coverage = sub_inter_mat.sum(axis=1)
                cumul_mat = np.zeros((delta * 2 + 1, delta * 2 + 1))
                stock_cumul_mat = []
                print("shape cumul mat = ", cumul_mat.shape)
                print("pos centro i = ", pre_pos_pxl_i)
                for chrom_id_j in self.list_id_chrom:
                    if chrom_id_i != chrom_id_j:
                        name_chrom_j = self.detection[chrom_id_j]["name"]
                        print("chrom : ", name_chrom_i, name_chrom_j)
                        pre_pos_bp_j = self.detection[chrom_id_j]["pre_detect_pos"]
                        pre_pos_pxl_j = self.bp_coord_2_pxl_coord(
                            pre_pos_bp_j, chrom_id_j, id_level
                        )
                        raw_sub = data_i[chrom_id_j]["raw"]
                        print("shape raw sup = ", raw_sub.shape)
                        print("pos centro j = ", pre_pos_pxl_j)

                        tmp = raw_sub[
                            pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1,
                            pre_pos_pxl_j - delta : pre_pos_pxl_j + delta + 1,
                        ]
                        print("shape tmp =", tmp.shape)
                        cumul_mat = cumul_mat + tmp
                #                        stock_cumul_mat.append(tmp)
                base_contact = cumul_mat.min() + cumul_mat.std() / 2.
                # base_contact = cumul_mat.min() + cumul_mat.std()
                #                cumul_mat = cumul_mat / base_contact

                #                over_smoothed = gaussian_filter(cumul_mat, delta, mode="constant", cval=0.0)
                #
                #                cumul_mat = cumul_mat  - over_smoothed
                #                cumul_mat = gaussian_filter(cumul_mat, 1, mode="constant", cval=0.0)
                #                cumul_mat = gaussian_filter(cumul_mat, size_filter, mode="constant", cval=0.0)
                cumul_mat[cumul_mat < 0] = 0
                self.detection[chrom_id_i][id_level]["cumul_mat"] = cumul_mat
                self.detection[chrom_id_i][id_level]["bootstrap"] = []
                self.detection[chrom_id_i][id_level]["stack_mat"] = stock_cumul_mat
                self.detection[chrom_id_i][id_level]["coverage"] = sub_coverage
                # cumul_contact_max = cumul_mat.sum((1)) * cumul_mat.mean((1))
                cumul_contact_max = cumul_mat.sum((1))
                tick_kb = np.array(level_pyr.dict_contigs[chrom_id_i]["tick_kb"])
                sub_tick_kb = np.array(
                    level_pyr.dict_contigs[chrom_id_i]["sub_tick_kb"]
                )
                end_tick_kb = np.array(
                    level_pyr.dict_contigs[chrom_id_i]["end_frags_kb"]
                )

                tick = tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]
                sub_tick = sub_tick_kb[
                    pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1
                ]
                end_tick = end_tick_kb[
                    pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1
                ]
                subTick = sub_tick_kb[
                    np.all([sub_tick_kb >= tick[0], sub_tick_kb <= tick[-1]], axis=0)
                ]

                for i in range(0, N_samples):
                    tmp = np.random.poisson(cumul_mat)
                    tmp = tmp - gaussian_filter(tmp, delta, mode="constant", cval=0.0)
                    tmp[tmp < 0] = 0
                    #            cumul_mat = gaussian_filter(cumul_mat,1,mode="constant", cval=0.0)
                    tmp_thresh = np.copy(tmp)
                    thresh = 0.35
                    tmp_thresh[tmp_thresh < tmp_thresh.max() * thresh] = (
                        tmp_thresh.max() * thresh
                    )
                    #                    tmp_ov_sm = gaussian_filter(tmp,delta,mode="constant",cval=0.0)
                    #                    tmp = tmp - tmp_ov_sm
                    #                    tmp[tmp<0] = 0
                    #                    tmp_coverage = np.random.poisson(sub_coverage)
                    #                    tmp_normalized = tmp / tmp_coverage
                    #                    data_boot = tmp_normalized.max((1)) * tmp_normalized.std((1))
                    #                    tmp_new = np.float32(tmp) / tmp.sum(axis=1)
                    data_boot = tmp_thresh.max(axis=1)
                    coeff_x, data_fit = gaussian_fit_kb(data_boot, tick, subTick)
                    #                    coeff_x = fitgaussian(tmp_new,tick)
                    pos_kb = coeff_x[2]
                    #                    tmp_corr = np.corrcoef(
                    #                    mon_delta = 10
                    #                    pos_kb, out_position_kb, score, data_fit_kb, sub_pixel_pos_pxl,out_position_pxl, \
                    #                    data_fit_pxl = self.estimate_location_bloc_corr(tmp_corr, mon_delta, tick, sub_tick)

                    self.detection[chrom_id_i][id_level]["bootstrap"].append(pos_kb)
                self.detection[chrom_id_i][id_level]["cumul_mat"] = gaussian_filter(
                    tmp, 1
                )
                self.detection[chrom_id_i][id_level]["bootstrap"] = np.array(
                    self.detection[chrom_id_i][id_level]["bootstrap"]
                )

    def plot_results(self, is_cerevisiae, delta_bp):
        """
        what else...
        :param is_cerevisiae: is cerevisiae
        """
        list_level = list(self.level.keys())

        list_name = []
        for chrom in list(self.detection.keys()):

            pos_centro_kb = self.detection[chrom]["pre_detect_pos"]
            f, axarr = plt.subplots(len(list_level), 3, figsize=(10, 10))
            id_lev = 0
            name_chrom = self.detection[chrom]["name"]
            list_name.append(name_chrom)
            for id_level in list_level:
                lev_pyr = self.level[id_level]
                data = lev_pyr.all_data[chrom]["sub_matrices"]
                imCorr = data[chrom]["corr"]
                imCumu = self.detection[chrom][id_level]["cumul_mat"]
                pos_centro_pxl = self.bp_coord_2_pxl_coord(
                    pos_centro_kb, chrom, id_level
                )
                axImCorr = axarr[id_lev, 0]
                axCumSub = axarr[id_lev, 1]
                axBootStrap = axarr[id_lev, 2]
                axImCorr.imshow(imCorr, interpolation="nearest", origin="lower")
                axImCorr.plot(pos_centro_pxl, pos_centro_pxl, "*w", markersize=9)
                axImCorr.set_xlim([0, imCorr.shape[0]])
                axImCorr.set_ylim([0, imCorr.shape[1]])
                axImCorr.set_title("Correlation Matrix", fontsize=10)
                axCumSub.imshow(imCumu, interpolation="nearest", origin="lower")
                axCumSub.set_xlim([0, imCumu.shape[0]])
                axCumSub.set_ylim([0, imCumu.shape[1]])
                axCumSub.set_title("Cumulated inter matrix", fontsize=10)
                self.detection[chrom][id_level]["bootstrap"] = np.array(
                    self.detection[chrom][id_level]["bootstrap"], dtype=np.float32
                )
                collect_pos = self.detection[chrom][id_level]["bootstrap"]
                axBootStrap.hist(
                    collect_pos, facecolor="b", edgecolor="b", linewidth=0, alpha=0.5
                )
                precision = collect_pos.std()
                mean_position = collect_pos.mean()
                deviation = 10000
                if is_cerevisiae:
                    ground_truth = np.array(self.ground_truth[chrom])
                    error_loc = np.min(np.abs(ground_truth - mean_position))
                    axBootStrap.axvspan(
                        ground_truth[0],
                        ground_truth[1],
                        edgecolor="r",
                        facecolor="r",
                        alpha=1,
                    )
                    axBootStrap.set_xlim(
                        [ground_truth[0] - deviation, ground_truth[1] + deviation]
                    )
                    #                    axBootStrap.set_xlim([ground_truth[0] - delta_bp, ground_truth[1] + delta_bp])
                    axBootStrap.text(
                        0.8,
                        0.90,
                        "mean error(bp) = " + str(np.around(error_loc, 3)),
                        fontsize=5,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axBootStrap.transAxes,
                    )
                    axBootStrap.legend(
                        ["ground truth", "bootstrap estimates"],
                        loc="upper left",
                        prop={"size": 5},
                    )
                    axBootStrap.set_title(
                        "Bootstrap distribution and ground truth", fontsize=10
                    )
                else:
                    axBootStrap.axvspan(
                        mean_position - 200,
                        mean_position + 200,
                        edgecolor="r",
                        facecolor="r",
                        alpha=1,
                    )
                    axBootStrap.legend(
                        ["mean position", "bootstrap estimates"],
                        loc="upper left",
                        prop={"size": 5},
                    )
                    axBootStrap.text(
                        0.8,
                        0.90,
                        "mean position(bp) = " + str(np.around(mean_position, 3)),
                        fontsize=5,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axBootStrap.transAxes,
                    )
                    axBootStrap.set_xlim(
                        [mean_position - delta_bp, mean_position + delta_bp]
                    )
                    axBootStrap.set_title("Bootstrap distribution", fontsize=10)
                axBootStrap.text(
                    0.8,
                    0.95,
                    "precision(bp) = " + str(np.around(precision, 3)),
                    fontsize=5,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axBootStrap.transAxes,
                )
                id_lev += 1

            for a in range(0, len(list_level) - 1):
                for b in range(1, 3):
                    plt.setp(axarr[a, b].get_xticklabels(), visible=False)
                    plt.setp(axarr[a, b].get_yticklabels(), visible=False)
            for a in range(0, len(list_level) - 1):
                plt.setp(axarr[a, 0].get_xticklabels(), visible=False)
            for b in range(1, 3):
                plt.setp(axarr[len(list_level) - 1, b].get_yticklabels(), visible=False)
            plt.setp(axarr[len(list_level) - 1, 3 - 1].get_xticklabels(), fontsize=9)
            plt.suptitle(name_chrom + ": detection centromere")
            plt.draw()
            f.savefig(
                os.path.join(self.all_res_folder, "localization_" + name_chrom + ".pdf")
            )
            plt.close()
        val_mean = []
        for id_level in list_level:
            data = []
            f = plt.figure(figsize=(10, 10))
            ax1 = f.add_subplot(111)
            matbootstrap = []
            for i in range(1, len(list_name) + 1):
                t = self.detection[i][id_level]["bootstrap"]
                matbootstrap.append(t)
                if is_cerevisiae:
                    data.append(
                        t - (self.ground_truth[i][0] + self.ground_truth[i][0]) / 2
                    )
                else:
                    data.append(t - t.mean())
                val_mean.append(t.mean())
            # plt.boxplot(data,notch=0, sym='', vert=1, whis=1.5,usermedians=val_mean)
            plt.boxplot(data, notch=0, sym="", vert=1, whis=1.5)

            plt.axhspan(0, 0)
            xtickNames = plt.setp(ax1, xticklabels=list_name)
            plt.setp(xtickNames, rotation=45, fontsize=8)
            if is_cerevisiae:
                ax1.set_title(
                    "Distribution of bootstrap estimates VS groundtruth centromeric position"
                )
            else:
                ax1.set_title(
                    "Distribution of bootstrap estimates VS mean estimated centromeric position"
                )
            f.savefig(
                os.path.join(self.all_res_folder, "boxplot_" + str(id_level) + ".pdf")
            )
            plt.close()
            fviolin = plt.figure(figsize=(10, 10))
            ax_violin = fviolin.add_subplot(111)
            violin_plot(ax_violin, data, list(range(1, len(list_name) + 1)), bp=True)
            fviolin.savefig(
                os.path.join(
                    self.all_res_folder, "violin_plot_" + str(id_level) + ".pdf"
                )
            )
            plt.close()
            file_id_level_bootstrap = (
                os.path.join(
                    self.all_res_folder, "bootstrap_positions_" + str(id_level)
                )
                + ".txt"
            )
            matbootstrap = np.array(matbootstrap)
            head = ""
            for i in list_name:
                head = head + i + "\t"
            if np.__version__ == "1.6.1":
                handle_file = open(file_id_level_bootstrap, "w")
                handle_file.write(head)
                np.savetxt(
                    file_id_level_bootstrap,
                    matbootstrap.T,
                    fmt="%.18e",
                    delimiter="\t ",
                )
            else:
                np.savetxt(
                    file_id_level_bootstrap,
                    matbootstrap.T,
                    fmt="%.18e",
                    delimiter="\t ",
                    newline="\n",
                    header=head,
                    footer="",
                    comments="",
                )

    def pre_detection_level(self, id_level, delta):
        # delta = 15
        level_pyr = self.level[id_level]

        for contig_index in level_pyr.dict_contigs:
            if not (contig_index == 17):
                print(
                    "analysing contig :", level_pyr.dict_contigs[contig_index]["name"]
                )
                estimated_pos_kb = dict()
                estimated_pos_pxl = dict()
                score = []
                SUB = level_pyr.all_data[contig_index]["sub_matrices"][contig_index]
                bloc_corr = SUB["corr"]
                tick_kb = level_pyr.dict_contigs[contig_index]["tick_kb"]
                sub_tick_kb = level_pyr.dict_contigs[contig_index]["sub_tick_kb"]
                out_mean_pos_kb, out_position_kb, score, data_fit, out_mean_pos_pxl, out_position_pxl, data_fit_pxl = self.estimate_location_bloc_corr(
                    bloc_corr, delta, tick_kb, sub_tick_kb
                )
                level_pyr.collect_bootstrap[contig_index]["kb"].append(out_mean_pos_kb)
                level_pyr.collect_bootstrap[contig_index]["pxl"].append(
                    out_mean_pos_pxl
                )
                estimated_pos_kb[contig_index] = (out_mean_pos_kb, out_position_kb)
                estimated_pos_pxl[contig_index] = (out_mean_pos_pxl, out_position_pxl)

                level_pyr.all_localization[contig_index] = dict()
                level_pyr.all_localization[contig_index]["kb"] = dict()
                level_pyr.all_localization[contig_index]["kb"] = estimated_pos_kb
                level_pyr.all_localization[contig_index]["pxl"] = dict()
                level_pyr.all_localization[contig_index]["pxl"] = estimated_pos_pxl
                level_pyr.all_data[contig_index]["score"] = dict()
                level_pyr.all_data[contig_index]["data_fit"] = dict()
                level_pyr.all_data[contig_index]["data_fit"]["kb"] = dict()
                level_pyr.all_data[contig_index]["data_fit"]["kb"][
                    contig_index
                ] = data_fit
                level_pyr.all_data[contig_index]["data_fit"]["pxl"] = dict()
                level_pyr.all_data[contig_index]["data_fit"]["pxl"][
                    contig_index
                ] = data_fit_pxl
                level_pyr.all_data[contig_index]["score"][contig_index] = score

    def estimate_location_bloc_corr(self, bloc_corr, delta, tick_kb, sub_tick_kb):
        score = []
        corrected_bloc = np.copy(bloc_corr)
        for i in range(0, corrected_bloc.shape[0] - 1):
            corrected_bloc[i, i] = (
                corrected_bloc[i, i - 1] + corrected_bloc[i, i + 1]
            ) / 2
        zoom_bloc_corr = np.copy(corrected_bloc)
        zoom_bloc_corr = zoom_bloc_corr ** 3
        for i in range(0, 2):
            score.append(0)
        for i in range(2, zoom_bloc_corr.shape[0] - 2):
            is_self = True
            r, a, s = self.la_cruz_azul_roja(zoom_bloc_corr, i, i, delta, is_self)
            score.append(s)
        for i in range(zoom_bloc_corr.shape[0] - 2, zoom_bloc_corr.shape[0]):
            score.append(0)
        score = np.array(score)
        raw_pxl_pos = np.argmax(score)
        sub_pixel_pos_kb, data_fit_kb, sub_pixel_pos_pxl, data_fit_pxl = self.sub_pixel_pos(
            score, tick_kb, sub_tick_kb
        )
        out_position_kb = [raw_pxl_pos, sub_pixel_pos_kb]
        out_position_kb = np.array(out_position_kb)
        out_position_pxl = [raw_pxl_pos, sub_pixel_pos_pxl]
        out_position_pxl = np.array(out_position_pxl)
        return (
            sub_pixel_pos_kb,
            out_position_kb,
            score,
            data_fit_kb,
            sub_pixel_pos_pxl,
            out_position_pxl,
            data_fit_pxl,
        )

    def sub_pixel_pos(self, score, tickX, sub_tickX):
        para_fit_kb, data_fit_kb = gaussian_fit_kb(score, tickX, sub_tickX)
        para_fit_pxl, data_fit_pxl = gaussian_fit_pxl(score, tickX, sub_tickX)

        sub_pix_pos_kb = para_fit_kb[2]
        sub_pix_pos_pxl = para_fit_pxl[2]

        return sub_pix_pos_kb, data_fit_kb, sub_pix_pos_pxl, data_fit_pxl

    def la_cruz_azul_roja(self, im, x, y, delta, is_self):
        """
        :param im: image
        :param x: coord x
        :param y: coord y
        :param delta: size cross
        :return: (x,y) cross score in im
        """
        output_graph = False
        bound_x = im.shape[0]
        bound_y = im.shape[1]
        delta_azul = delta + 10
        # out_interpolate = scp.zoom(im, 2,order=5)
        v_x = np.ones((bound_x))
        v_y = np.ones((bound_y))
        v_x[:3] = 0
        v_x[-3:] = 0
        v_y[:3] = 0
        v_y[-3:] = 0

        nuance_x = gaussian_filter(v_x, 1)
        nuance_y = gaussian_filter(v_y, 1)
        ###############################################
        if x - delta < 0:
            delta_x_down = x
            offset_x_up = abs(x - delta)
        else:
            delta_x_down = delta
            offset_x_up = 0
        if x + delta >= bound_x:
            delta_x_up = bound_x - x - 1
            offset_x_down = x + delta - bound_x
        else:
            delta_x_up = delta
            offset_x_down = 0
        if y - delta < 0:
            delta_y_down = y
            offset_y_up = abs(y - delta)
        else:
            delta_y_down = delta
            offset_y_up = 0
        if y + delta >= bound_y:
            delta_y_up = bound_y - y - 1
            offset_y_down = y + delta - bound_y
        else:
            delta_y_up = delta
            offset_y_down = 0
            ###############################################
        if x - delta_azul < 0:
            azul_delta_x_down = x
            azul_offset_x_up = abs(x - delta_azul)
        else:
            azul_delta_x_down = delta_azul
            azul_offset_x_up = 0
        if x + delta_azul >= bound_x:
            azul_delta_x_up = bound_x - x - 1
            azul_offset_x_down = x + delta_azul - bound_x
        else:
            azul_delta_x_up = delta_azul
            azul_offset_x_down = 0
        if y - delta_azul < 0:
            azul_delta_y_down = y
            azul_offset_y_up = abs(y - delta_azul)
        else:
            azul_delta_y_down = delta_azul
            azul_offset_y_up = 0
        if y + delta_azul >= bound_y:
            azul_delta_y_up = bound_y - y - 1
            azul_offset_y_down = y + delta_azul - bound_y
        else:
            azul_delta_y_up = delta_azul
            azul_offset_y_down = 0

        bloc_corr = im[
            x - delta_x_down : x + delta_x_up, y - delta_y_down : y + delta_y_up
        ]

        vect_x = im[x - delta_x_down : x + delta_x_up, :]
        left_x = vect_x[:5].mean()
        right_x = vect_x[-5:].mean()
        min_corr = im.min()
        max_corr = im.max()
        interv = max_corr - min_corr
        extrem_azul_x = (np.mean([left_x, right_x]) - min_corr) / interv
        vect_y = im[:, y - delta_y_down : y + delta_y_up]
        left_y = vect_y[:5].mean()
        right_y = vect_y[-5:].mean()
        extrem_azul_y = (np.mean([left_y, right_y]) - min_corr) / interv
        # print "extrem azul x = ", extrem_azul_x
        # print "extrem azul y = ", extrem_azul_y

        extrem_azul = np.mean([extrem_azul_x, extrem_azul_y])
        # print "extrem azul = ", extrem_azul
        ###############################################
        id_cruz_roja = [[], []]
        delta_3 = min(delta_x_up, delta_y_down)
        id_cruz_roja[0].extend([i for i in range(x + delta_3, x, -1)])
        id_cruz_roja[1].extend([j for j in range(y - delta_3, y, +1)])
        delta_4 = min(delta_x_down, delta_y_up)
        id_cruz_roja[0].extend([i for i in range(x - 1, x - delta_4 - 1, -1)])
        id_cruz_roja[1].extend([j for j in range(y + 1, y + delta_4 + 1, +1)])

        if not (is_self):
            delta_1 = min(delta_x_down, delta_y_down)
            id_cruz_roja = [
                [i for i in range(x - delta_1, x)],
                [j for j in range(y - delta_1, y)],
            ]
            delta_2 = min(delta_x_up, delta_y_up)
            id_cruz_roja[0].extend([i for i in range(x + 1, x + delta_2 + 1)])
            id_cruz_roja[1].extend([i for i in range(y + 1, y + delta_2 + 1)])

        if len(id_cruz_roja[0]) > 0 and id_cruz_roja[1] > 1:
            cruz_roja = im[id_cruz_roja]
            id_azul_y = list(range(y - azul_delta_y_down, y))
            id_azul_y.extend(list(range(y + 1, y + azul_delta_y_up + 1)))
            id_azul_x = list(range(x - azul_delta_x_down, x))
            id_azul_x.extend(list(range(x + 1, x + azul_delta_x_up + 1)))
            cruz_azul_1 = im[x, id_azul_y]
            cruz_azul_2 = im[id_azul_x, y]
            cruz_azul = [v for v in cruz_azul_1]
            cruz_azul.extend([v for v in cruz_azul_2])

            val_al_lado = im[x - 1 : x + 1, y - 1 : y + 1]
            cruz_azul = np.array(cruz_azul)
            cruz_roja = np.array(cruz_roja)
            if output_graph:
                plt.figure()
                plt.imshow(im, interpolation="nearest")
                for i, j in zip(id_cruz_roja[1], id_cruz_roja[0]):
                    plt.plot(i, j, "*w")
                for i in id_azul_x:
                    plt.plot(y, i, "*b")
                for i in id_azul_y:
                    plt.plot(i, x, "*b")
                plt.show()
                # cruz_roja = (cruz_roja - min_corr)/interv
            score_rojo = cruz_roja.mean()
            cruz_azul = (cruz_azul - min_corr) / interv
            score_azul = cruz_azul.mean()
            std_azul = cruz_azul.std()
            std_rojo = cruz_roja.std()
            if is_self:
                score = (
                    (score_rojo / score_azul)
                    * std_azul
                    * std_rojo
                    * nuance_x[x]
                    * bloc_corr.mean()
                )

            else:
                score = (
                    val_al_lado[1, 1]
                    * (score_rojo / score_azul)
                    * std_azul
                    * std_rojo
                    * bloc_corr.mean()
                )
        else:
            score_rojo = 0
            score_azul = 0
            score = 0
        return score_rojo, score_azul, score


if __name__ == "__main__":
    hostname = socket.gethostname()
    print("Host name:", hostname)
    ordi = hostname.split(".")[0]
    size_pyramid = 4
    factor = 3
    min_bin_per_contig = 1
    size_chunk = 10000
    name = "curr_genome"
    data_set = dict()
    data_set["hansenii"] = "hansenii/"
    data_set["castelli"] = "castelli/"
    data_set["S1"] = "S1/"
    data_set["ohmeri"] = "ohmeri/"
    data_set["kapsulata"] = "kapsulata/"

    selected = "kapsulata"

    if ordi == "matisse":
        data_set_root = "/media/hervemn/data/data_set_assembly/"
    if ordi == "rv-retina":
        data_set_root = "/Volumes/BigData/HiC/data_set_assembly/"
    if ordi == "loopkin":
        data_set_root = "/data/data_set_assembly/"
    if ordi == "casa":
        data_set_root = "/media/BigData/HiC/data_set_assembly/"
    if ordi == "renoir":
        data_set_root = "/Users/hervemn/data_set_assembly/"

    max_level = size_pyramid - 1

    base_folder = os.path.join(data_set_root, data_set[selected], "analysis")
    pyramid_hic = pyr.build_and_filter(
        base_folder, size_pyramid, factor, min_bin_per_contig, size_chunk, max_level
    )
    print("pyramid loaded")
    ####################################################################################################################
    output_folder = os.path.join(data_set_root, "results")
    if not (os.path.exists(output_folder)):
        os.mkdir(output_folder)
    output_folder = os.path.join(data_set_root, "results", data_set[selected])
    if not (os.path.exists(output_folder)):
        os.mkdir(output_folder)
    if not (os.path.exists(output_folder)):
        os.mkdir(output_folder)
    output_folder = os.path.join(
        data_set_root, "results", data_set[selected], "centro_detect"
    )
    if not (os.path.exists(output_folder)):
        os.mkdir(output_folder)
        ####################################################################################################################

    start = 1
    A = analysis(pyramid_hic, output_folder, size_pyramid, start)

    delta_bp_predetection = 75000
    A.pre_detection(delta_bp_predetection)
    #    delta_bp = 53000
    delta_bp = 30000
    size_filter_bp = 300
    N_samples = 200
    is_cerevisiae = selected == "S1"
    #    multi = pb.parallel(8,A)
    #    multi.launch_computing(N_samples, 2, delta_bp, size_filter_bp)
    #    A.refine_localization(delta_bp, size_filter_bp, N_samples)
    #    A.bootstrap(id_level=1,delta_bp=delta_bp,size_filter_bp=size_filter_bp,N=N_samples)
    A.bootstrap(
        id_level=2, delta_bp=delta_bp, size_filter_bp=size_filter_bp, N=N_samples
    )
    A.bootstrap(
        id_level=3, delta_bp=delta_bp, size_filter_bp=size_filter_bp, N=N_samples
    )
    A.plot_results(is_cerevisiae, delta_bp)
