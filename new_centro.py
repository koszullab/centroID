__author__ = "hervemarie-nelly"
# coding: utf-8
import math, sys, time
import pp
import os
import sys, socket
import pyramid as pyr
import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_laplace
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import optimization as optim
import cuda_lib as cuda
import visualize as viz
from scipy.io import savemat


class level:
    def __init__(self, level, output_folder, is_max_level):
        self.level = level
        self.output_folder = os.path.join(output_folder, str(level))
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
        self.dict_contigs = dict()
        ContFrags = pyramid.spec_level[str(self.level)]["contigs_dict"]
        coord_cont = dict()
        if np.__version__ == "1.7.1" or np.__version__ == "1.8.0.dev-1a9aa5a":
            self.cis_free_im_init = np.empty_like(self.im_init)
            np.copyto(self.cis_free_im_init, self.im_init)
        else:
            self.cis_free_im_init = np.copy(self.im_init)
        #        self.cis_free_im_init = np.float32(self.cis_free_im_init)
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
                self.dict_contigs[id_cont]["tick_kb"] = np.array(tick_kb)
                self.dict_contigs[id_cont]["end_frags_kb"] = np.array(end_frags_kb)
                self.dict_contigs[id_cont]["sub_tick_kb"] = np.array(sub_tick_kb)
                self.dict_contigs[id_cont]["tick"] = np.array(tick)
            #######################
        print("cis free matrix ...")
        total_trans = 0
        n_tot = 0
        for id_cont in range(1, len(self.dict_contigs) + 1):
            print("current chrom  = ", id_cont)
            if not (id_cont == 17):
                full = self.cis_free_im_init[coord_cont[id_cont], :]
                intra = self.cis_free_im_init[
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
                    mat_tmp = self.cis_free_im_init[
                        np.ix_(coord_cont[id_cont], coord_cont[id_cont])
                    ]
                    mat_tmp[:, :] = self.mean_value_trans
                    mat_tmp = np.random.poisson(mat_tmp)
                    self.cis_free_im_init[
                        np.ix_(coord_cont[id_cont], coord_cont[id_cont])
                    ] = mat_tmp
                else:
                    self.cis_free_im_init[
                        np.ix_(coord_cont[id_cont], coord_cont[id_cont])
                    ] = self.mean_value_trans

        self.distri_frag = np.array(self.distri_frag)
        self.define_inter_chrom_coord()
        self.init_data()
        self.n_contigs = len(self.dict_contigs)

    def init_data(self,):
        print("init data ", self.level)

        if np.__version__ == "1.7.1" or np.__version__ == "1.8.0.dev-1a9aa5a":
            tmp = np.empty_like(self.im_init)
            np.copyto(tmp, self.im_init)
            tmp_cis_free = np.empty_like(self.cis_free_im_init)
            np.copyto(tmp_cis_free, self.cis_free_im_init)
        else:
            tmp = np.copy(self.im_init)
            tmp_cis_free = np.copy(self.cis_free_im_init)
        tmp = np.float32(tmp)
        tmp_cis_free = np.float32(tmp_cis_free)
        self.im_curr = tmp
        self.im_cis_free_curr = tmp_cis_free

        if self.is_max_level:
            self.corr_im = np.corrcoef(self.im_cis_free_curr)
        else:
            self.corr_im = self.im_cis_free_curr

    def perform_scn(self, mat, n_iter):
        """
        :param mat: initial matrix
        :param n iter: number of iterations
        return: normalized matrix
        """
        if np.__version__ == "1.7.1" or np.__version__ == "1.8.0.dev-1a9aa5a":
            new_mat = np.empty_like(mat)
            np.copyto(new_mat, mat)
        else:
            new_mat = np.copy(mat)
        new_mat = np.float32(new_mat)
        for k in range(0, n_iter):
            #            print "iteration = ", k
            v_norm = new_mat.sum(axis=1)
            new_mat = new_mat.T / v_norm
        return new_mat

    def update_data_scn(self, fact_sub_sampling):
        print("init data for scn.. level = ", self.level)
        tmp = np.random.poisson(self.im_init / fact_sub_sampling)
        tmp_trans_free = np.random.poisson(self.cis_free_im_init / fact_sub_sampling)
        n_iter = 10
        tmp = np.float32(tmp)
        tmp_trans_free = np.float32(tmp_trans_free)

        #        self.im_curr = cuda.scn(tmp, n_iter)
        #        self.im_cis_free_curr = cuda.scn(tmp_trans_free, n_iter)
        self.im_curr = self.perform_scn(tmp, n_iter)
        self.im_cis_free_curr = self.perform_scn(tmp_trans_free, n_iter)

        if self.is_max_level:
            self.corr_im = np.corrcoef(self.im_cis_free_curr)
        else:
            self.corr_im = self.im_cis_free_curr

    def plot_sub_matrices(self, folder, thresh, thresh_im_init):
        im_less_cis = np.copy(self.im_init)
        for id_cont_X in self.dict_contigs:
            intra_coord = self.inter_coord[id_cont_X][id_cont_X]
            im_less_cis[intra_coord] = 0
        corr_im = np.corrcoef(im_less_cis)
        fig = plt.figure()
        plt.imshow(
            corr_im, interpolation="nearest", vmin=0, vmax=thresh * corr_im.max()
        )
        fig.savefig(
            os.path.join(
                folder, str(self.level) + "_full_im_corr_" + str(thresh) + ".pdf"
            )
        )
        plt.close()
        fig = plt.figure()
        plt.imshow(
            self.im_curr,
            interpolation="nearest",
            vmin=0,
            vmax=thresh_im_init * self.im_curr.max(),
        )
        fig.savefig(
            os.path.join(
                folder,
                str(self.level) + "_full_im_init_" + str(thresh_im_init) + ".pdf",
            )
        )
        plt.close()
        print("plot sub matrices in " + folder)

        for id_cont_X in self.dict_contigs:
            name_contig = self.dict_contigs[id_cont_X]["name"]
            print(name_contig)
            intra_coord = self.inter_coord[id_cont_X][id_cont_X]
            tmp_c = corr_im[intra_coord]
            tmp_i = self.im_init[intra_coord]
            fig = plt.figure()
            plt.imshow(
                tmp_c, interpolation="nearest", vmin=0, vmax=thresh * tmp_c.max()
            )
            fig.savefig(
                os.path.join(folder, str(self.level) + "_corr" + name_contig + ".pdf")
            )
            plt.close()

    def gpu_update_data_scn(self, cuda_gen, iteration, fact_sub_sampling):

        print("init data for scn.. level = ", self.level)
        n_iter_scn = 10
        do_random = iteration != 0
        self.im_cis_free_curr = cuda_gen.generate_new_matrix(
            n_iter_scn, do_random, fact_sub_sampling
        )
        self.im_curr = self.im_cis_free_curr
        if self.is_max_level:
            self.corr_im = np.corrcoef(self.im_cis_free_curr)
        else:
            self.corr_im = self.im_cis_free_curr

    def update_data(self,):
        print("init data .. level = ", self.level)
        tmp = np.random.poisson(self.im_init)
        tmp_cis_free = np.random.poisson(self.cis_free_im_init)

        tmp = np.float32(tmp)
        tmp_cis_free = np.float32(tmp_cis_free)
        self.im_curr = tmp
        self.im_cis_free_curr = tmp_cis_free

        if self.is_max_level:
            self.corr_im = np.corrcoef(self.im_cis_free_curr)
        else:
            self.corr_im = self.im_cis_free_curr

    def dist_bp_2_frag(self, dist_bp):
        mean_length_bp = self.distri_frag.mean()
        output_pxl = int(dist_bp / mean_length_bp)
        return output_pxl

    def define_inter_chrom_coord(self):
        """

        """
        self.inter_coord = dict()
        self.all_data = dict()
        print("define inter chrom coord")
        for id_cont_X in self.dict_contigs:
            print("id chrom = ", id_cont_X)
            if not (id_cont_X == 17):
                self.inter_coord[id_cont_X] = dict()
                self.all_data[id_cont_X] = dict()
                coord_intra_X = self.dict_contigs[id_cont_X]["intra_coord"]
                self.inter_coord[id_cont_X]["all"] = np.ix_(
                    coord_intra_X, np.arange(0, self.n_frags)
                )
                for id_cont_Y in self.dict_contigs:
                    print("id chrom = ", id_cont_Y)
                    if not (id_cont_Y == 17):
                        self.all_data[id_cont_X][id_cont_Y] = dict()
                        coord_intra_Y = self.dict_contigs[id_cont_Y]["intra_coord"]
                        self.inter_coord[id_cont_X][id_cont_Y] = np.ix_(
                            coord_intra_X, coord_intra_Y
                        )


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

    def plot_4_sup_material(
        self, size_sub_win_bp, folder, thresh, thresh_im_init, thresh_im_norm
    ):
        iter_scn = 10
        list_level = list(self.level.keys())
        for lev in list_level:
            matlab_file = os.path.join(folder, str(lev) + "_data.mat")
            data_matlab = dict()
            level_pyr = self.level[lev]
            delta = level_pyr.dist_bp_2_frag(size_sub_win_bp)
            im_less_cis = np.copy(level_pyr.im_init)
            curr_im = level_pyr.im_init
            data_matlab["distri_length_frag"] = level_pyr.distri_frag
            for id_cont_X in level_pyr.dict_contigs:
                intra_coord = level_pyr.inter_coord[id_cont_X][id_cont_X]
                im_less_cis[intra_coord] = 0
            corr_im = np.corrcoef(im_less_cis)
            im_norm = level_pyr.perform_scn(im_less_cis, iter_scn)
            for id_cont_X in level_pyr.dict_contigs:
                intra_coord = level_pyr.inter_coord[id_cont_X][id_cont_X]
                im_norm[intra_coord] = level_pyr.perform_scn(
                    level_pyr.im_init[intra_coord], iter_scn
                )
            siz_fig = 9
            fig = plt.figure(figsize=(siz_fig, siz_fig))
            plt.hist(level_pyr.distri_frag, 300)
            plt.title("fragment size distribution (bp)")
            fig.savefig(
                os.path.join(
                    folder, str(level_pyr.level) + "_frag_length_distribution" ".pdf"
                )
            )
            plt.close()

            fig = plt.figure(figsize=(siz_fig, siz_fig))
            plt.imshow(
                corr_im, interpolation="nearest", vmin=0, vmax=thresh * corr_im.max()
            )
            fig.savefig(
                os.path.join(
                    folder,
                    str(level_pyr.level) + "_full_im_corr_" + str(thresh) + ".pdf",
                )
            )
            plt.close()

            fig = plt.figure(figsize=(siz_fig, siz_fig))
            plt.imshow(
                level_pyr.im_curr,
                interpolation="nearest",
                vmin=0,
                vmax=thresh_im_init * level_pyr.im_curr.max() * 3 ** (lev - 3),
            )
            fig.savefig(
                os.path.join(
                    folder,
                    str(level_pyr.level)
                    + "_full_im_init_"
                    + str(thresh_im_init)
                    + ".pdf",
                )
            )
            plt.close()
            fig = plt.figure(figsize=(siz_fig, siz_fig))
            plt.imshow(
                im_norm,
                interpolation="nearest",
                vmin=0,
                vmax=thresh_im_norm * im_norm.max() * 3 ** (lev - 3),
            )
            fig.savefig(
                os.path.join(
                    folder,
                    str(level_pyr.level)
                    + "_full_im_norm_"
                    + str(thresh_im_norm)
                    + ".pdf",
                )
            )
            plt.close()

            data_matlab["full_raw_matrix"] = curr_im
            data_matlab["full_corr_matrix"] = corr_im
            data_matlab["full_norm_matrix"] = curr_im

            print("plot sub matrices in " + folder)

            for id_cont_X in level_pyr.dict_contigs:
                ground_truth = np.array(self.ground_truth[id_cont_X])
                tick_kb = level_pyr.dict_contigs[id_cont_X]["tick_kb"]
                name_contig = level_pyr.dict_contigs[id_cont_X]["name"]
                print(name_contig)
                data_matlab[name_contig + "_ground_truth"] = ground_truth
                intra_coord = level_pyr.inter_coord[id_cont_X][id_cont_X]
                tmp_c = corr_im[intra_coord]
                tmp_i = level_pyr.im_init[intra_coord]
                fig = plt.figure(figsize=(siz_fig, siz_fig))
                plt.imshow(
                    tmp_c, interpolation="nearest", vmin=0, vmax=thresh * tmp_c.max()
                )
                fig.savefig(
                    os.path.join(
                        folder, str(level_pyr.level) + "_corr" + name_contig + ".pdf"
                    )
                )
                plt.close()

                if lev == list_level[-1]:
                    tick_score_kb = self.level[lev].dict_contigs[id_cont_X]["tick_kb"]
                    score = self.level[list_level[-1]].all_data[id_cont_X]["score"][
                        id_cont_X
                    ]
                    score = np.array(score)
                    data_matlab[name_contig + "_tick_score_bp"] = tick_score_kb
                    data_matlab[name_contig + "_score_rough_estimate"] = score
                    fig = plt.figure(figsize=(siz_fig, siz_fig))
                    plt.bar(
                        tick_score_kb, score, linewidth=2, facecolor="r", edgecolor="r"
                    )
                    plt.title("score pre detection " + name_contig)
                    plt.xlabel("genomic position (bp)")
                    fig.savefig(
                        os.path.join(
                            folder,
                            str(level_pyr.level)
                            + "_"
                            + name_contig
                            + "_score_rough_detect"
                            + ".pdf",
                        )
                    )
                    plt.close()
                pre_pos_bp_i = self.detection[id_cont_X]["pre_detect_pos"]
                pre_pos_pxl_i = self.bp_coord_2_pxl_coord(pre_pos_bp_i, id_cont_X, lev)
                tick = tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]
                data_matlab["tick_cum_mat_" + name_contig] = tick
                cumul_mat_norm = np.zeros((delta * 2 + 1, delta * 2 + 1))
                cumul_mat_raw = np.zeros((delta * 2 + 1, delta * 2 + 1))
                for id_cont_Y in self.list_id_chrom:
                    if id_cont_X != id_cont_Y:
                        coord_ij = level_pyr.inter_coord[id_cont_X][id_cont_Y]
                        pre_pos_bp_j = self.detection[id_cont_Y]["pre_detect_pos"]
                        pre_pos_pxl_j = self.bp_coord_2_pxl_coord(
                            pre_pos_bp_j, id_cont_Y, lev
                        )
                        cumul_mat_norm = (
                            cumul_mat_norm
                            + im_norm[coord_ij][
                                pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1,
                                pre_pos_pxl_j - delta : pre_pos_pxl_j + delta + 1,
                            ]
                        )
                        cumul_mat_raw = (
                            cumul_mat_raw
                            + curr_im[coord_ij][
                                pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1,
                                pre_pos_pxl_j - delta : pre_pos_pxl_j + delta + 1,
                            ]
                        )
                data_matlab[name_contig + "intra_corr"] = tmp_c
                data_matlab[name_contig + "intra_raw"] = tmp_i
                fig = plt.figure(figsize=(siz_fig, siz_fig))
                plt.imshow(
                    cumul_mat_raw,
                    interpolation="nearest",
                    vmin=0,
                    vmax=thresh * cumul_mat_raw.max(),
                )
                fig.savefig(
                    os.path.join(
                        folder,
                        str(level_pyr.level) + "_cumul_raw" + name_contig + ".pdf",
                    )
                )
                plt.close()
                fig = plt.figure(figsize=(siz_fig, siz_fig))
                plt.plot(tick, cumul_mat_raw.sum(axis=1))
                plt.xlabel("genomic_position (bp)")
                plt.title("raw data inter sum profile " + name_contig)
                plt.axvspan(
                    ground_truth[0],
                    ground_truth[1],
                    edgecolor="r",
                    facecolor="r",
                    alpha=1,
                )
                plt.legend(["contact enrichment", "ground truth"])
                fig.savefig(
                    os.path.join(
                        folder,
                        str(level_pyr.level)
                        + "_sum_profile_cumul_raw"
                        + name_contig
                        + ".pdf",
                    )
                )
                plt.close()
                fig = plt.figure(figsize=(siz_fig, siz_fig))
                plt.plot(tick, cumul_mat_raw.max(axis=1))
                plt.axvspan(
                    ground_truth[0],
                    ground_truth[1],
                    edgecolor="r",
                    facecolor="r",
                    alpha=1,
                )
                plt.xlabel("genomic_position (bp)")
                plt.title("raw data inter sum profile " + name_contig)
                plt.legend(["contact enrichment", "ground truth"])
                fig.savefig(
                    os.path.join(
                        folder,
                        str(level_pyr.level)
                        + "_max_profile_cumul_raw"
                        + name_contig
                        + ".pdf",
                    )
                )
                plt.close()
                data_matlab[name_contig + "cum_raw"] = cumul_mat_raw
                data_matlab[name_contig + "cum_norm"] = cumul_mat_norm
                fig = plt.figure(figsize=(siz_fig, siz_fig))
                plt.imshow(
                    cumul_mat_norm,
                    interpolation="nearest",
                    vmin=0,
                    vmax=thresh * cumul_mat_norm.max(),
                )
                fig.savefig(
                    os.path.join(
                        folder,
                        str(level_pyr.level) + "_cumul_norm" + name_contig + ".pdf",
                    )
                )
                plt.close()

                fig = plt.figure(figsize=(siz_fig, siz_fig))
                plt.plot(tick, cumul_mat_norm.sum(axis=1))
                plt.axvspan(
                    ground_truth[0],
                    ground_truth[1],
                    edgecolor="r",
                    facecolor="r",
                    alpha=1,
                )
                plt.xlabel("genomic_position (bp)")
                plt.title("normalized data inter sum profile " + name_contig)
                plt.legend(["contact enrichment", "ground truth"])
                fig.savefig(
                    os.path.join(
                        folder,
                        str(level_pyr.level)
                        + "_sum_profile_cumul_norm"
                        + name_contig
                        + ".pdf",
                    )
                )
                plt.close()
                fig = plt.figure(figsize=(siz_fig, siz_fig))
                plt.plot(tick, cumul_mat_norm.max(axis=1))
                plt.axvspan(
                    ground_truth[0],
                    ground_truth[1],
                    edgecolor="r",
                    facecolor="r",
                    alpha=1,
                )
                plt.xlabel("genomic_position (bp)")
                plt.title("normalized data inter max profile " + name_contig)
                plt.legend(["contact enrichment", "ground truth"])
                fig.savefig(
                    os.path.join(
                        folder,
                        str(level_pyr.level)
                        + "_max_profile_cumul_norm"
                        + name_contig
                        + ".pdf",
                    )
                )
                plt.close()

            savemat(matlab_file, data_matlab)

    def manual_pre_detection(self):
        """

        """
        ultimate_pyr = self.level[self.max_level]

        self.detection = dict()
        self.list_id_chrom = list(ultimate_pyr.dict_contigs.keys())

        for chrom_id in self.list_id_chrom:
            intra_coord = ultimate_pyr.inter_coord[chrom_id][chrom_id]
            auto_corr = ultimate_pyr.corr_im[intra_coord]
            tick_kb = ultimate_pyr.dict_contigs[chrom_id]["tick_kb"]
            tmp = viz.manual_detect(auto_corr, tick_kb)

            estimated_pos_kb = tmp.pos_kb
            estimated_pos_pxl = tmp.pos_pxl
            score = np.zeros((auto_corr.shape[0],), dtype=np.float32)
            score[estimated_pos_pxl] = 1
            data_fit = score
            data_fit_pxl = score

            ultimate_pyr.all_localization[chrom_id] = dict()
            ultimate_pyr.all_localization[chrom_id]["kb"] = dict()
            ultimate_pyr.all_localization[chrom_id]["kb"] = estimated_pos_kb
            ultimate_pyr.all_localization[chrom_id]["pxl"] = dict()
            ultimate_pyr.all_localization[chrom_id]["pxl"] = estimated_pos_pxl
            ultimate_pyr.all_data[chrom_id]["score"] = dict()
            ultimate_pyr.all_data[chrom_id]["data_fit"] = dict()
            ultimate_pyr.all_data[chrom_id]["data_fit"]["kb"] = dict()
            ultimate_pyr.all_data[chrom_id]["data_fit"]["kb"][chrom_id] = data_fit
            ultimate_pyr.all_data[chrom_id]["data_fit"]["pxl"] = dict()
            ultimate_pyr.all_data[chrom_id]["data_fit"]["pxl"][chrom_id] = data_fit_pxl
            ultimate_pyr.all_data[chrom_id]["score"][chrom_id] = score

            self.detection[chrom_id] = dict()
            self.detection[chrom_id]["name"] = ultimate_pyr.dict_contigs[chrom_id][
                "name"
            ]
            for id_level in list(self.level.keys()):
                self.detection[chrom_id][id_level] = dict()
                self.detection[chrom_id][id_level]["bootstrap_norm"] = []
                self.detection[chrom_id][id_level]["bootstrap_raw"] = []
            self.detection[chrom_id]["pre_detect_pos"] = estimated_pos_kb

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
                self.detection[chrom_id][id_level]["bootstrap_norm"] = []
                self.detection[chrom_id][id_level]["bootstrap_raw"] = []
            self.detection[chrom_id]["pre_detect_pos"] = ultimate_pyr.all_localization[
                chrom_id
            ]["kb"][chrom_id][0]

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
                coord_intra_contig = level_pyr.inter_coord[contig_index][contig_index]
                bloc_corr = level_pyr.corr_im[coord_intra_contig]
                tick_kb = level_pyr.dict_contigs[contig_index]["tick_kb"]
                sub_tick_kb = level_pyr.dict_contigs[contig_index]["sub_tick_kb"]
                end_tick_kb = level_pyr.dict_contigs[contig_index]["end_frags_kb"]
                out_mean_pos_kb, out_position_kb, score, data_fit, out_mean_pos_pxl, out_position_pxl, data_fit_pxl = self.estimate_location_bloc_corr(
                    bloc_corr, delta, tick_kb, sub_tick_kb, end_tick_kb
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

    def estimate_location_bloc_corr(
        self, bloc_corr, delta, tick_kb, sub_tick_kb, end_tick
    ):
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
            score, tick_kb, sub_tick_kb, end_tick
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

    def sub_pixel_pos(self, score, tickX, sub_tickX, end_tick):
        para_fit_kb, data_fit_kb = optim.gaussian_fit_kb(
            score, tickX, sub_tickX, end_tick
        )
        para_fit_pxl, data_fit_pxl = optim.gaussian_fit_pxl(score, tickX, sub_tickX)

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

        nuance_x = optim.gaussian_filter(v_x, 1)
        nuance_y = optim.gaussian_filter(v_y, 1)
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

    def bootstrap_norm_data(
        self, delta_bp, size_filter_bp, N, thresh, fact_sub_sampling
    ):
        """

        :param delta_bp:
        :param size_filter_bp:
        :param N:
        """
        keys_level = list(self.level.keys())
        print(" start bootstrap on norm data , sub sampling = ", fact_sub_sampling)
        for id_level in keys_level:
            level_pyr = self.level[id_level]
            n_rng = 100
            if not (level_pyr.is_max_level):
                print("cis free matrix type = ", level_pyr.cis_free_im_init.dtype)
                cuda_gen = cuda.randomize(level_pyr.cis_free_im_init, n_rng)
            for i in range(0, N):
                print("##########################")
                print("bootstrap iteration = ", i)
                if not (level_pyr.is_max_level):
                    level_pyr.gpu_update_data_scn(cuda_gen, i, fact_sub_sampling)
                else:
                    level_pyr.update_data_scn(fact_sub_sampling)
                self.refine_localization_4_boot(
                    id_level, delta_bp, size_filter_bp, thresh
                )
                print("##########################")
            if not (level_pyr.is_max_level):
                cuda_gen.free_gpu()
                del cuda_gen
            for chrom_id_i in self.list_id_chrom:
                self.detection[chrom_id_i][id_level]["bootstrap_norm"] = np.array(
                    self.detection[chrom_id_i][id_level]["bootstrap_norm"]
                )

    def bootstrap_raw_data(
        self, delta_bp, size_filter_bp, N_samples, thresh, fact_sub_sampling
    ):
        """

        :param delta_bp:
        :param size_filter_bp:
        :param N_samples:
        """
        keys_level = list(self.level.keys())
        for id_level in keys_level:

            level_pyr = self.level[id_level]
            level_pyr.init_data()
            delta = level_pyr.dist_bp_2_frag(delta_bp)
            print("level pyramid = ", id_level)
            print("delta = ", delta)
            size_filter = level_pyr.dist_bp_2_frag(size_filter_bp)
            ########################################################################################################################
            image = level_pyr.im_cis_free_curr
            ########################################################################################################################
            for chrom_id_i in self.list_id_chrom:
                name_chrom_i = self.detection[chrom_id_i]["name"]
                pre_pos_bp_i = self.detection[chrom_id_i]["pre_detect_pos"]
                pre_pos_pxl_i = self.bp_coord_2_pxl_coord(
                    pre_pos_bp_i, chrom_id_i, id_level
                )
                cumul_mat = np.zeros((delta * 2 + 1, delta * 2 + 1))
                for chrom_id_j in self.list_id_chrom:
                    if chrom_id_i != chrom_id_j:
                        name_chrom_j = self.detection[chrom_id_j]["name"]
                        coord_ij = level_pyr.inter_coord[chrom_id_i][chrom_id_j]
                        pre_pos_bp_j = self.detection[chrom_id_j]["pre_detect_pos"]
                        pre_pos_pxl_j = self.bp_coord_2_pxl_coord(
                            pre_pos_bp_j, chrom_id_j, id_level
                        )
                        cumul_mat = (
                            cumul_mat
                            + image[coord_ij][
                                pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1,
                                pre_pos_pxl_j - delta : pre_pos_pxl_j + delta + 1,
                            ]
                        )
                cumul_mat[cumul_mat < 0] = 0
                self.detection[chrom_id_i][id_level]["cumul_mat"] = cumul_mat
                self.detection[chrom_id_i][id_level]["bootstrap_raw"] = []

                tick_kb = level_pyr.dict_contigs[chrom_id_i]["tick_kb"]
                sub_tick_kb = level_pyr.dict_contigs[chrom_id_i]["sub_tick_kb"]
                end_tick_kb = level_pyr.dict_contigs[chrom_id_i]["end_frags_kb"]

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
                    if i > 0:
                        tmp = np.random.poisson(cumul_mat / fact_sub_sampling)
                    else:
                        tmp = np.copy(cumul_mat / fact_sub_sampling)
                    tmp = tmp - optim.gaussian_filter(
                        tmp, delta, mode="constant", cval=0.0
                    )
                    tmp[tmp < 0] = 0
                    tmp_thresh = np.copy(tmp)
                    #                    thresh = 0.2
                    tmp_thresh[tmp_thresh < tmp_thresh.max() * thresh] = (
                        tmp_thresh.max() * thresh
                    )
                    ########################################################################
                    tmp_thresh = optim.gaussian_filter(
                        tmp_thresh, 1, mode="constant", cval=0.0
                    )
                    ########################################################################
                    data_boot = tmp_thresh.max(axis=1)
                    ########################################################################
                    #                    data_boot = tmp_thresh.sum(axis=1)
                    ########################################################################
                    coeff_x, data_fit = optim.gaussian_fit_kb(
                        data_boot, tick, subTick, end_tick
                    )
                    pos_kb = coeff_x[2]

                    self.detection[chrom_id_i][id_level]["bootstrap_raw"].append(pos_kb)
                self.detection[chrom_id_i][id_level]["cumul_mat_raw"] = cumul_mat
                self.detection[chrom_id_i][id_level]["bootstrap_raw"] = np.array(
                    self.detection[chrom_id_i][id_level]["bootstrap_raw"]
                )

    def refine_localization_4_boot(self, id_level, delta_bp, size_filter_bp, thresh):
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
        ########################################################################################################################
        image = level_pyr.im_cis_free_curr
        ########################################################################################################################
        for chrom_id_i in self.list_id_chrom:
            pre_pos_bp_i = self.detection[chrom_id_i]["pre_detect_pos"]
            pre_pos_pxl_i = self.bp_coord_2_pxl_coord(
                pre_pos_bp_i, chrom_id_i, id_level
            )
            cumul_mat = np.zeros((delta * 2 + 1, delta * 2 + 1))
            coord_all_i = level_pyr.inter_coord[chrom_id_i]["all"]
            data_i = np.copy(image[coord_all_i])
            for chrom_id_j in self.list_id_chrom:
                if chrom_id_i != chrom_id_j:
                    #                    coord_ij = level_pyr.inter_coord[chrom_id_i][chrom_id_j]
                    coord_intra_j = level_pyr.dict_contigs[chrom_id_j]["intra_coord"]
                    pre_pos_bp_j = self.detection[chrom_id_j]["pre_detect_pos"]
                    pre_pos_pxl_j = self.bp_coord_2_pxl_coord(
                        pre_pos_bp_j, chrom_id_j, id_level
                    )
                    cumul_mat = (
                        cumul_mat
                        + data_i[:, coord_intra_j][
                            pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1,
                            pre_pos_pxl_j - delta : pre_pos_pxl_j + delta + 1,
                        ]
                    )
            #                    cumul_mat = cumul_mat + image[coord_ij][pre_pos_pxl_i - delta: pre_pos_pxl_i + delta + 1,
            #                                  pre_pos_pxl_j - delta: pre_pos_pxl_j + delta + 1]
            #            base_contact = cumul_mat.min() + cumul_mat.std()
            #
            #            cumul_mat = cumul_mat / base_contact
            cumul_mat = optim.gaussian_filter(
                cumul_mat, size_filter, mode="constant", cval=0.0
            )

            self.detection[chrom_id_i][id_level]["cumul_mat_norm"] = cumul_mat

            tick_kb = level_pyr.dict_contigs[chrom_id_i]["tick_kb"]
            end_tick_kb = level_pyr.dict_contigs[chrom_id_i]["end_frags_kb"]

            sub_tick_kb = level_pyr.dict_contigs[chrom_id_i]["sub_tick_kb"]
            tick = tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]
            sub_tick = sub_tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]
            end_tick = end_tick_kb[pre_pos_pxl_i - delta : pre_pos_pxl_i + delta + 1]

            subTick = sub_tick_kb[
                np.all([sub_tick_kb >= tick[0], sub_tick_kb <= tick[-1]], axis=0)
            ]
            cumul_mat = cumul_mat - optim.gaussian_filter(
                cumul_mat, delta, mode="constant", cval=0.0
            )
            cumul_mat[cumul_mat < 0] = 0
            cumul_mat = optim.gaussian_filter(
                cumul_mat, size_filter, mode="constant", cval=0.0
            )
            cumul_mat_thresh = np.copy(cumul_mat)
            #            thresh = 0.2
            cumul_mat_thresh[cumul_mat_thresh < cumul_mat_thresh.max() * thresh] = (
                cumul_mat_thresh.max() * thresh
            )
            #####################################################################################
            cumul_mat_thresh = optim.gaussian_filter(
                cumul_mat_thresh, 1, mode="constant", cval=0.0
            )
            #####################################################################################
            data_to_fit = cumul_mat_thresh.max(axis=1)
            #####################################################################################
            #            data_to_fit = cumul_mat_thresh.sum(axis=1)
            #####################################################################################
            coeff_x, data_fit = optim.gaussian_fit_kb(
                data_to_fit, tick, subTick, end_tick
            )
            pos_kb = coeff_x[2]
            self.detection[chrom_id_i][id_level]["bootstrap_norm"].append(pos_kb)

    def plot_results(self, is_cerevisiae, delta_bp, fact_sub_sampling):
        """
        what else...
        :param is_cerevisiae: is cerevisiae
        """
        list_level = list(self.level.keys())
        list_name = []
        folder_sub_sampling = os.path.join(self.all_res_folder, str(fact_sub_sampling))
        if not (os.path.exists((folder_sub_sampling))):
            os.mkdir(folder_sub_sampling)
        for chrom in list(self.detection.keys()):
            pos_centro_kb = self.detection[chrom]["pre_detect_pos"]
            n_rows = len(list_level)
            #            f, axarr = plt.subplots(len(list_level),5, figsize=(10, 10))
            siz_fig = 9
            f = plt.figure(figsize=((n_rows - 1) * siz_fig, siz_fig))
            #            f= plt.figure()
            gs = gridspec.GridSpec(n_rows, 5)
            gs.update(wspace=0.1, hspace=0.5)

            id_lev = n_rows - 1
            name_chrom = self.detection[chrom]["name"]
            list_name.append(name_chrom)
            for id_level in list_level:
                lev_pyr = self.level[id_level]
                intra_coord = lev_pyr.inter_coord[chrom][chrom]
                imCumu_raw = self.detection[chrom][id_level]["cumul_mat_raw"]
                extentCumu_raw = (0, imCumu_raw.shape[0], 0, imCumu_raw.shape[1])

                imCumu_norm = self.detection[chrom][id_level]["cumul_mat_norm"]
                extentCumu_norm = (0, imCumu_norm.shape[0], 0, imCumu_norm.shape[1])

                pos_centro_pxl = self.bp_coord_2_pxl_coord(
                    pos_centro_kb, chrom, id_level
                )
                imCorr = lev_pyr.corr_im[intra_coord]
                if lev_pyr.is_max_level:
                    axImCorr = plt.subplot(gs[id_lev, 0])
                    extentCorr = (0, imCorr.shape[0], 0, imCorr.shape[1])
                    axImCorr.imshow(
                        imCorr,
                        interpolation="nearest",
                        origin="lower",
                        extent=extentCorr,
                    )
                    axImCorr.plot(pos_centro_pxl, pos_centro_pxl, "*w", markersize=9)
                    axImCorr.set_xlim([0, imCorr.shape[0]])
                    axImCorr.set_ylim([0, imCorr.shape[1]])
                    axImCorr.set_title("Correlation Matrix", fontsize=10)
                elif id_level == 2:
                    axImCorr = plt.subplot(gs[id_lev, 0])
                    list_level[-1]
                    tick_score_kb = self.level[list_level[-1]].dict_contigs[chrom][
                        "tick_kb"
                    ]
                    score = self.level[list_level[-1]].all_data[chrom]["score"][chrom]
                    score = np.array(score)
                    axImCorr.bar(
                        tick_score_kb, score, linewidth=2, facecolor="r", edgecolor="r"
                    )
                    #                    axImCorr.bar(tick_score_kb, score, facecolor='r', edgecolor='r', linewidth=0, alpha=1)
                    axImCorr.set_xlim([tick_score_kb[0], tick_score_kb[-1]])
                    axImCorr.set_ylim([score.min(), score.max()])
                    axImCorr.set_title("Score detection centromere", fontsize=10)

                axCumSub_raw = plt.subplot(gs[id_lev, 1])
                axBootStrap_raw = plt.subplot(gs[id_lev, 2])
                axCumSub_norm = plt.subplot(gs[id_lev, 3])
                axBootStrap_norm = plt.subplot(gs[id_lev, 4])

                axCumSub_raw.imshow(
                    imCumu_raw,
                    interpolation="nearest",
                    origin="lower",
                    extent=extentCumu_raw,
                    aspect="equal",
                )
                axCumSub_raw.set_xlim([0, imCumu_raw.shape[0]])
                axCumSub_raw.set_ylim([0, imCumu_raw.shape[1]])
                axCumSub_raw.set_title("Cumulated raw inter matrix", fontsize=10)

                axCumSub_norm.imshow(
                    imCumu_norm,
                    interpolation="nearest",
                    origin="lower",
                    extent=extentCumu_norm,
                    aspect="equal",
                )
                axCumSub_norm.set_xlim([0, imCumu_norm.shape[0]])
                axCumSub_norm.set_ylim([0, imCumu_norm.shape[1]])
                axCumSub_norm.set_title(
                    "Cumulated normalized inter matrix", fontsize=10
                )

                collect_pos_raw = self.detection[chrom][id_level]["bootstrap_raw"]
                axBootStrap_raw.hist(
                    collect_pos_raw,
                    facecolor="b",
                    edgecolor="b",
                    linewidth=0,
                    alpha=0.5,
                )
                precision_raw = collect_pos_raw.std()
                mean_position_raw = collect_pos_raw.mean()

                collect_pos_norm = self.detection[chrom][id_level]["bootstrap_norm"]
                axBootStrap_norm.hist(
                    collect_pos_norm,
                    facecolor="b",
                    edgecolor="b",
                    linewidth=0,
                    alpha=0.5,
                )
                precision_norm = collect_pos_norm.std()
                mean_position_norm = collect_pos_norm.mean()
                deviation = 10000
                mean_size_pxl = np.median(lev_pyr.distri_frag)
                if is_cerevisiae:
                    ground_truth = np.array(self.ground_truth[chrom])
                    error_loc_raw = np.min(np.abs(ground_truth - mean_position_raw))
                    axBootStrap_raw.axvspan(
                        ground_truth[0],
                        ground_truth[1],
                        edgecolor="r",
                        facecolor="r",
                        alpha=1,
                    )
                    axBootStrap_raw.set_xlim(
                        [ground_truth[0] - deviation, ground_truth[1] + deviation]
                    )
                    axBootStrap_raw.text(
                        0.8,
                        0.90,
                        "mean error(bp) = " + str(np.around(error_loc_raw, 3)),
                        fontsize=5,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axBootStrap_raw.transAxes,
                    )
                    axBootStrap_raw.legend(
                        ["ground truth", "bootstrap estimates"],
                        loc="upper left",
                        prop={"size": 5},
                    )
                    axBootStrap_raw.set_title(
                        "Bootstrap distribution (raw data) "
                        + "res = "
                        + str(mean_size_pxl)
                        + " bp",
                        fontsize=9,
                    )

                    error_loc_norm = np.min(np.abs(ground_truth - mean_position_norm))
                    axBootStrap_norm.axvspan(
                        ground_truth[0],
                        ground_truth[1],
                        edgecolor="r",
                        facecolor="r",
                        alpha=1,
                    )
                    axBootStrap_norm.set_xlim(
                        [ground_truth[0] - deviation, ground_truth[1] + deviation]
                    )
                    axBootStrap_norm.text(
                        0.8,
                        0.90,
                        "mean error(bp) = " + str(np.around(error_loc_norm, 3)),
                        fontsize=5,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axBootStrap_norm.transAxes,
                    )
                    axBootStrap_norm.legend(
                        ["ground truth", "bootstrap estimates"],
                        loc="upper left",
                        prop={"size": 5},
                    )
                    axBootStrap_norm.set_title(
                        "Bootstrap distribution (norm data) "
                        + "res = "
                        + str(mean_size_pxl)
                        + " bp",
                        fontsize=9,
                    )

                else:
                    axBootStrap_raw.axvspan(
                        mean_position_raw - 200,
                        mean_position_raw + 200,
                        edgecolor="r",
                        facecolor="r",
                        alpha=1,
                    )
                    axBootStrap_raw.legend(
                        ["mean position", "bootstrap estimates"],
                        loc="upper left",
                        prop={"size": 5},
                    )
                    axBootStrap_raw.text(
                        0.8,
                        0.90,
                        "mean position(bp) = " + str(np.around(mean_position_raw, 3)),
                        fontsize=5,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axBootStrap_raw.transAxes,
                    )
                    axBootStrap_raw.set_xlim(
                        [mean_position_raw - delta_bp, mean_position_raw + delta_bp]
                    )
                    axBootStrap_raw.set_title(
                        "Bootstrap distribution (raw data)"
                        + "res = "
                        + str(mean_size_pxl)
                        + " bp",
                        fontsize=9,
                    )

                    axBootStrap_norm.axvspan(
                        mean_position_norm - 200,
                        mean_position_norm + 200,
                        edgecolor="r",
                        facecolor="r",
                        alpha=1,
                    )
                    axBootStrap_norm.legend(
                        ["mean position", "bootstrap estimates"],
                        loc="upper left",
                        prop={"size": 5},
                    )
                    axBootStrap_norm.text(
                        0.8,
                        0.90,
                        "mean position(bp) = " + str(np.around(mean_position_norm, 3)),
                        fontsize=5,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axBootStrap_norm.transAxes,
                    )
                    axBootStrap_norm.set_xlim(
                        [mean_position_norm - delta_bp, mean_position_norm + delta_bp]
                    )
                    axBootStrap_norm.set_title(
                        "Bootstrap distribution"
                        + "res = "
                        + str(mean_size_pxl)
                        + " bp",
                        fontsize=9,
                    )

                axBootStrap_raw.text(
                    0.8,
                    0.95,
                    "precision(bp) = " + str(np.around(precision_raw, 3)),
                    fontsize=5,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axBootStrap_raw.transAxes,
                )
                axBootStrap_norm.text(
                    0.8,
                    0.95,
                    "precision(bp) = " + str(np.around(precision_norm, 3)),
                    fontsize=5,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axBootStrap_norm.transAxes,
                )
                id_lev -= 1
                aspectratio = 1.0
                aspectratio_hist = 0.5

                ratio_boot_norm = float(
                    axBootStrap_norm.get_xlim()[1] - axBootStrap_norm.get_xlim()[0]
                ) / float(
                    axBootStrap_norm.get_ylim()[1] - axBootStrap_norm.get_ylim()[0]
                )
                axBootStrap_norm.set_aspect(ratio_boot_norm * aspectratio_hist)
                plt.setp(axBootStrap_norm.get_yticklabels(), visible=False)
                plt.setp(axBootStrap_norm.get_xticklabels(), fontsize=7)

                ratio_boot_raw = float(
                    axBootStrap_raw.get_xlim()[1] - axBootStrap_norm.get_xlim()[0]
                ) / float(axBootStrap_raw.get_ylim()[1] - axBootStrap_raw.get_ylim()[0])
                axBootStrap_raw.set_aspect(ratio_boot_raw * aspectratio_hist)
                plt.setp(axBootStrap_raw.get_yticklabels(), visible=False)
                plt.setp(axBootStrap_raw.get_xticklabels(), fontsize=7)

                ratio_cum_raw = float(
                    axCumSub_raw.get_xlim()[1] - axCumSub_raw.get_xlim()[0]
                ) / float(axCumSub_raw.get_ylim()[1] - axCumSub_raw.get_ylim()[0])
                axCumSub_raw.set_aspect(ratio_cum_raw * aspectratio)
                plt.setp(axCumSub_raw.get_xticklabels(), visible=False)

                ratio_cum_norm = float(
                    axCumSub_norm.get_xlim()[1] - axCumSub_norm.get_xlim()[0]
                ) / float(axCumSub_norm.get_ylim()[1] - axCumSub_norm.get_ylim()[0])
                axCumSub_norm.set_aspect(ratio_cum_norm * aspectratio)
                plt.setp(axCumSub_norm.get_xticklabels(), visible=False)

                axCumSub_norm.get_yaxis().set_visible(False)
                axCumSub_norm.get_xaxis().set_visible(False)
                axCumSub_raw.get_yaxis().set_visible(False)
                axCumSub_raw.get_xaxis().set_visible(False)
                if lev_pyr.is_max_level:
                    ratio_im_corr = float(
                        axImCorr.get_xlim()[1] - axImCorr.get_xlim()[0]
                    ) / float(axImCorr.get_ylim()[1] - axImCorr.get_ylim()[0])
                    axImCorr.set_aspect(ratio_im_corr * aspectratio)
                    plt.setp(axImCorr.get_xticklabels(), visible=False)
                    plt.setp(axImCorr.get_yticklabels(), visible=False)
                elif id_level == 2:
                    ratio_im_corr = float(
                        axImCorr.get_xlim()[1] - axImCorr.get_xlim()[0]
                    ) / float(axImCorr.get_ylim()[1] - axImCorr.get_ylim()[0])
                    axImCorr.set_aspect(ratio_im_corr * aspectratio)
                    plt.setp(axImCorr.get_yticklabels(), visible=False)
                    plt.setp(axImCorr.get_xticklabels(), fontsize=3)

            plt.suptitle(name_chrom + ": detection centromere")
            plt.draw()
            f.savefig(
                os.path.join(
                    folder_sub_sampling,
                    "localization_"
                    + name_chrom
                    + "_"
                    + str(fact_sub_sampling)
                    + ".pdf",
                )
            )
            # plt.close(f)
        val_mean_raw = []
        val_mean_norm = []
        for id_level in list_level:
            data_raw = []
            f_raw = plt.figure(figsize=(10, 10))
            ax1 = f_raw.add_subplot(111)
            matbootstrap_raw = []
            for i in range(1, len(list_name) + 1):
                t = self.detection[i][id_level]["bootstrap_raw"]
                matbootstrap_raw.append(t)
                if is_cerevisiae:
                    data_raw.append(
                        t - (self.ground_truth[i][0] + self.ground_truth[i][0]) / 2
                    )
                else:
                    data_raw.append(t - t.mean())
                val_mean_raw.append(t.mean())
            plt.boxplot(data_raw, notch=0, sym="", vert=1, whis=1.5)

            plt.axhspan(0, 0)
            xtickNames = plt.setp(ax1, xticklabels=list_name)
            plt.setp(xtickNames, rotation=45, fontsize=8)
            if is_cerevisiae:
                ax1.set_title(
                    "Distribution of bootstrap estimates (raw data) VS groundtruth centromeric position"
                )
            else:
                ax1.set_title(
                    "Distribution of bootstrap estimates (raw data) VS mean estimated centromeric position"
                )
            f_raw.savefig(
                os.path.join(
                    folder_sub_sampling,
                    "boxplot_raw_"
                    + str(id_level)
                    + "_"
                    + str(fact_sub_sampling)
                    + ".pdf",
                )
            )
            # plt.close(f_raw)

            file_id_level_bootstrap_raw = (
                os.path.join(
                    folder_sub_sampling, "bootstrap_positions_raw_" + str(id_level)
                )
                + "_"
                + str(fact_sub_sampling)
                + ".txt"
            )
            matbootstrap_raw = np.array(matbootstrap_raw)
            head = ""
            for i in list_name:
                head = head + i + "\t"
            if np.__version__ == "1.6.1":
                handle_file_raw = open(file_id_level_bootstrap_raw, "w")
                handle_file_raw.write(head)
                np.savetxt(
                    file_id_level_bootstrap_raw,
                    matbootstrap_raw.T,
                    fmt="%.18e",
                    delimiter="\t ",
                )
            else:
                np.savetxt(
                    file_id_level_bootstrap_raw,
                    matbootstrap_raw.T,
                    fmt="%.18e",
                    delimiter="\t ",
                    newline="\n",
                    header=head,
                    footer="",
                    comments="",
                )

            data_norm = []
            f_norm = plt.figure(figsize=(10, 10))
            ax1 = f_norm.add_subplot(111)
            matbootstrap_norm = []
            for i in range(1, len(list_name) + 1):
                t = self.detection[i][id_level]["bootstrap_norm"]
                matbootstrap_norm.append(t)
                if is_cerevisiae:
                    data_norm.append(
                        t - (self.ground_truth[i][0] + self.ground_truth[i][0]) / 2
                    )
                else:
                    data_norm.append(t - t.mean())
                val_mean_norm.append(t.mean())
            plt.boxplot(data_norm, notch=0, sym="", vert=1, whis=1.5)

            plt.axhspan(0, 0)
            xtickNames = plt.setp(ax1, xticklabels=list_name)
            plt.setp(xtickNames, rotation=45, fontsize=8)
            if is_cerevisiae:
                ax1.set_title(
                    "Distribution of bootstrap estimates (norm. data) VS groundtruth centromeric position"
                )
            else:
                ax1.set_title(
                    "Distribution of bootstrap estimates (norm. data) VS mean estimated centromeric position"
                )
            f_norm.savefig(
                os.path.join(
                    folder_sub_sampling,
                    "boxplot_norm_"
                    + str(id_level)
                    + "_"
                    + str(fact_sub_sampling)
                    + ".pdf",
                )
            )
            # plt.close(f_norm)

            file_id_level_bootstrap_norm = (
                os.path.join(
                    folder_sub_sampling, "bootstrap_positions_norm_" + str(id_level)
                )
                + "_"
                + str(fact_sub_sampling)
                + ".txt"
            )
            matbootstrap_norm = np.array(matbootstrap_norm)
            head = ""
            for i in list_name:
                head = head + i + "\t"
            if np.__version__ == "1.6.1":
                handle_file_norm = open(file_id_level_bootstrap_norm, "w")
                handle_file_norm.write(head)
                np.savetxt(
                    file_id_level_bootstrap_norm,
                    matbootstrap_norm.T,
                    fmt="%.18e",
                    delimiter="\t ",
                )
            else:
                np.savetxt(
                    file_id_level_bootstrap_norm,
                    matbootstrap_norm.T,
                    fmt="%.18e",
                    delimiter="\t ",
                    newline="\n",
                    header=head,
                    footer="",
                    comments="",
                )

        for chrom in self.list_id_chrom:
            for id_level in list_level:
                self.detection[chrom][id_level]["bootstrap_raw"] = []
                self.detection[chrom][id_level]["bootstrap_norm"] = []
                self.detection[chrom][id_level]["cumul_mat_norm"] = 0
                self.detection[chrom][id_level]["cumul_mat_raw"] = 0


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

    selected = "S1"

    if ordi == "matisse":
        data_set_root = "/media/hervemn/data/data_set_assembly/"
    if ordi == "rv-retina" or "duvel":
        data_set_root = "/Volumes/BigData/HiC/data_set_assembly"
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
    start = 2
    A = analysis(pyramid_hic, output_folder, size_pyramid, start)

    delta_bp_predetection = 70000
    auto_detect = True
    #    levmax = A.level[max_level]
    my_fold = "/Users/hervemarie-nelly/Desktop/to_chr/"

    if auto_detect:
        A.pre_detection(delta_bp_predetection)
    else:
        A.manual_pre_detection()
    delta_bp = 40000

    #    for l in range(start, 4):
    #        A.level[l].plot_sub_matrices(my_fold, 0.8,0.001)

    A.plot_4_sup_material(delta_bp, my_fold, 0.8, 0.003, 0.02)


#    size_filter_bp = 120
#    N_samples = 1
#    threshold = 0.25
#    is_cerevisiae = selected == "S1"
##    list_fact_sub_sampling = np.array([1], dtype=np.float32)
#    list_fact_sub_sampling = np.float32([ 0.1])
#    for fact_sub_sampling in list_fact_sub_sampling:
#        A.bootstrap_raw_data(delta_bp, size_filter_bp, N_samples, threshold, fact_sub_sampling)
#        A.bootstrap_norm_data(delta_bp, size_filter_bp, N_samples, threshold, fact_sub_sampling)
#        A.plot_results(is_cerevisiae, delta_bp, fact_sub_sampling)
