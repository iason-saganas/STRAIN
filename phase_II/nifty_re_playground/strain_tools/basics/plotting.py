from typing import Literal
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage.restoration import denoise_tv_chambolle
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import re

from .common_utils import raise_warning

__all__ = ["usual_plot", "visualize_stress", "thesis_plot", "thesis_multiplot", "red", "light_red", "blue", "light_blue",
           "lighter_blue", "lightest_blue", "green", "light_green", "save_figure", "smooth_matrix", "title_fontsize_pts",
           "label_fontsize_pts", "tick_label_fontsize", "legend_label_fontsize"]

thesis_fontsize_pts = 12
title_fontsize_pts = 1.6 * thesis_fontsize_pts
label_fontsize_pts = 1.6 * thesis_fontsize_pts
tick_label_fontsize = 1.4 * thesis_fontsize_pts
legend_label_fontsize = 1.4 * thesis_fontsize_pts

red = (0.74, 0.1, 0.1, 1)
light_red = (0.74, 0.1, 0.1, 0.4)
blue = (0, 0.37, 0.99, 1)
light_blue = (0.42, 0.8, 0.93, 0.4)
lighter_blue = (0.42, 0.8, 0.93, 0.3)
lightest_blue = (0.42, 0.8, 0.93, 0.1)
green = (0.23, 0.85, 0.25, 1)
light_green = (0.23, 0.85, 0.25, 0.4)

plt.style.use("/Users/iason/PycharmProjects/STRAIN/data/style_components/standardStyle.mplstyle")

# Standard style extensions
mpl.rcParams["axes.titlesize"] = title_fontsize_pts
mpl.rcParams["axes.labelsize"] = label_fontsize_pts
mpl.rcParams["xtick.labelsize"] = tick_label_fontsize
mpl.rcParams["ytick.labelsize"] = tick_label_fontsize
mpl.rcParams["legend.fontsize"] = legend_label_fontsize
mpl.rcParams["text.usetex"] = False

mpl.rcParams["font.family"] = "Hubballi"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Hubballi"
mpl.rcParams["mathtext.it"] = "Hubballi"
mpl.rcParams["mathtext.bf"] = "Hubballi"


def wrap_latex_label(label, width=20):
    parts = re.split(r'(\$.*?\$)', label)  # split into math / text
    lines = []
    current = ""
    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            if len(current) + len(part) > width:
                lines.append(current.strip())
                current = part
            else:
                current += part
        else:
            for word in part.split():
                if len(current) + len(word) + 1 > width:
                    lines.append(current.strip())
                    current = word + " "
                else:
                    current += word + " "
    if current:
        lines.append(current.strip())
    return "\n".join(lines)


def thesis_plot(
                mode:Literal["basic", "longer", "square"]="basic",
                xl=r"Time $t$ $\mathrm{[s]}$",
                yl=r"$d(t)$ $\:\mathrm{[10^{-19}]}$",
                title=None,
                xlim=None,
                ylim=None,
                show=True,
                close=False,
                save_fig=False,
                save_path="",
                custom_ax=None,
                tight_ly=True):

    mode_dict = {"basic": (8., 2.),
                 "longer": (8., 4.),
                 "square": (4.,4.)}

    if mode not in mode_dict:
        raise ValueError(f"Please provide one of following modes for `thesis_plot`:\n{mode_dict.keys()}")
    else:
        md = mode_dict[mode]
        fig_size = tuple(plt.gca().figure.get_size_inches())

        if mode == "square":
            if not abs(fig_size[0] - fig_size[1]) < 1e-6:  # allow tiny floating error
                raise ValueError(f"You are in `thesis_plot` 'square' mode, expected a square figure, "
                                 f"but got `{fig_size}` instead.")
        else:
            if fig_size != md:
                raise ValueError(f"You are in `thesis_plot` '{mode}' mode, corresponding to a fig size of `{md}`, "
                                 f"but you passed `{fig_size}` instead.")

    if custom_ax is None:
        ax = plt.gca()
    else:
        ax = custom_ax

    yl_wrapped = wrap_latex_label(yl, width=60)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl_wrapped)
    ax.set_title(title)

    labels = ax.get_legend_handles_labels()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if labels != ([], []):
        ax.legend()

    save_figure(save_fig, show, close, save_path, tight_ly)


def save_figure(save_fig, show=True, close=False, save_path="", tight_ly=True):
    if save_fig:
        if tight_ly:
            plt.tight_layout()
        current_date = datetime.datetime.now()
        if save_path == "":
            plt.savefig(f"{current_date}.pdf")
        else:
            plt.savefig(f"{save_path}_{current_date}.pdf")

    if show:
        if tight_ly:
            plt.tight_layout()
        plt.show()
    if close:
        plt.close()


def thesis_multiplot():
    pass


def usual_plot(xl=r"Time $t$ $\mathrm{[sec]}$", yl=r"Strain $h$ $\mathrm{[10^{-19}]}$", title=None, xlim=None, ylim=None,
               show=True, close=False, save_fig=False, save_path="", custom_ax=None):
    raise_warning("`usual_plot` will be deprecated, please use `thesis_plot` instead")
    if custom_ax is None:
        ax = plt.gca()
    else:
        ax = custom_ax

    ax.set_xlabel(xl, fontsize=20)
    ax.set_ylabel(yl, fontsize=20)
    ax.set_title(title, fontsize=25)

    labels = ax.get_legend_handles_labels()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if labels != ([], []):
        ax.legend()
    if save_fig:
        plt.tight_layout()
        current_date = datetime.datetime.now()
        if save_path=="":
            plt.savefig(f"{current_date}.png")
        else:
            plt.savefig(f"{save_path}_{current_date}.png")

    if show:
        plt.tight_layout()
        plt.show()
    if close:
        plt.close()


_smoothing_modes = Literal[
    "gaussian",
    "median",
    "uniform",
    "bilateral",
    "anisotropic"
]

def smooth_matrix(
    mat: np.ndarray,
    smoothing_lvl,
    mode: _smoothing_modes = "gaussian"
) -> np.ndarray:

    if mode == "gaussian":
        return gaussian_filter(mat, sigma=smoothing_lvl)

    elif mode == "median":
        size = int(max(1, smoothing_lvl))
        return median_filter(mat, size=size)

    elif mode == "uniform":
        size = int(max(1, smoothing_lvl))
        return uniform_filter(mat, size=size)

    elif mode == "bilateral":
        # smoothing_lvl used as intensity + spatial scale
        return cv2.bilateralFilter(
            mat.astype(np.float32),
            d=5,
            sigmaColor=smoothing_lvl * 20,
            sigmaSpace=smoothing_lvl * 5,
        )

    elif mode == "anisotropic":
        # total variation denoising
        return denoise_tv_chambolle(mat, weight=smoothing_lvl)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def detect_outliers_in_stress(stress_matrix, fac, cols, rows):
    mean_stress, std_stress = np.mean(stress_matrix), np.std(stress_matrix)
    thresh = mean_stress + fac * std_stress

    larger_than_threshhold_idcs = np.where(stress_matrix > thresh)
    t, f = cols[larger_than_threshhold_idcs[1]], rows[larger_than_threshhold_idcs[0]]

    return t, f


def DEPR_visualize_stress(stress_matrix, rows, cols, smooth=False, detect_outliers=False, tl="", hlines=None, vlines=None,
                     smoothing_level=5, cmap="plasma", plot_colorbar=True, colorbar_label="Stress",
                     xl=r"Time $\mathrm{[s]}$", yl=r"Frequency $\mathrm{[Hz]}$", custom_ax=None, **kwargs):
    if custom_ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    else:
        ax = custom_ax
    stress_matrix = stress_matrix.real

    cols_are_increasing = np.all(np.diff(cols) > 0)  # strictly increasing
    rows_are_increasing = np.all(np.diff(rows) > 0)  # strictly increasing
    if not cols_are_increasing:
        raise ValueError("Columns must be increasing")
    if not rows_are_increasing:
        stress_matrix = np.fft.fftshift(stress_matrix, axes=0)  # shift DC frequency to middle
        rows = np.fft.fftshift(rows, axes=0)
        print("\t\tRows must be increasing, assuming a priori standard DFT order and moving DC to the middle")
        # Must be increasing because we want to plot from - frequency to 0 to + frequency on the y-axis

    if smooth:
        stress_matrix = smooth_matrix(stress_matrix, smoothing_level)

    if detect_outliers:

        t_outlier, f_outlier = detect_outliers_in_stress(stress_matrix, fac=10, cols=cols, rows=rows)

        plt.plot(t_outlier, f_outlier, "b.", markersize=5)
        plt.show()

        plt.hist2d(t_outlier, f_outlier, bins=[50, 50], cmap='magma')
        plt.colorbar(label='Counts')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Outlier density in (t, f)')
        plt.show()

        bins = np.linspace(t_outlier.min(), t_outlier.max(), 50)
        bin_indices = np.digitize(t_outlier, bins)
        mean_f = [f_outlier[bin_indices == i].mean() for i in range(1, len(bins))]

        plt.plot(bins[:-1], mean_f, 'o-')
        plt.xlabel('Time')
        plt.ylabel('Mean frequency of outliers')
        plt.show()

    ax.imshow(stress_matrix, origin='lower', aspect='auto',
               extent=[np.min(cols), np.max(cols), np.min(rows), np.max(rows)],
               cmap=cmap, interpolation='nearest')

    if hlines is not None:
        ax.hlines(hlines, 0, np.max(cols), color="r", ls="-")
    if vlines is not None:
        ax.vlines(vlines, 0, np.max(rows), color="r", ls="-")

    if plot_colorbar:
        ax.colorbar(label=colorbar_label)

    usual_plot(xl=xl, yl=yl, title=tl, custom_ax=ax, **kwargs)


def visualize_stress(stress_matrix, rows, cols, smooth=False, smoothing_level=5, custom_ax=None, delay_plot=False,
                     plot_colorbar=True, colorbar_label="Stress", cmap="plasma", tl="", hlines=None, vlines=None,
                     xl=r"Time $\mathrm{[s]}$", yl=r"Frequency $\mathrm{[Hz]}$", return_aux=False, **kwargs):
    if custom_ax is None:
        plt.figure(figsize=(4., 4.))
        ax = plt.gca()
    else:
        ax = custom_ax
    stress_matrix = stress_matrix.real

    cols_are_increasing = np.all(np.diff(cols) > 0)  # strictly increasing
    rows_are_increasing = np.all(np.diff(rows) > 0)  # strictly increasing
    if not cols_are_increasing:
        raise ValueError("Columns must be increasing")
    if not rows_are_increasing:
        stress_matrix = np.fft.fftshift(stress_matrix, axes=0)  # shift DC frequency to middle
        rows = np.fft.fftshift(rows, axes=0)
        print("\t\tRows must be increasing, assuming a priori standard DFT order and moving DC to the middle")
        # Must be increasing because we want to plot from - frequency to 0 to + frequency on the y-axis

    if smooth:
        stress_matrix = smooth_matrix(stress_matrix, smoothing_level)

    im = ax.imshow(stress_matrix, origin='lower', aspect='auto',
               extent=[np.min(cols), np.max(cols), np.min(rows), np.max(rows)],
               cmap=cmap, interpolation='nearest')  # nearest: No smoothing

    if hlines is not None:
        ax.hlines(hlines, 0, np.max(cols), color="r", ls="-")
    if vlines is not None:
        ax.vlines(vlines, 0, np.max(rows), color="r", ls="-")

    if plot_colorbar:
        cb = ax.colorbar(label=colorbar_label)
    else:
        cb = None

    if not delay_plot:
        thesis_plot(mode="square", xl=xl, yl=yl, title=tl, custom_ax=ax, tight_ly=False, **kwargs)

    if return_aux:
        aux = (cb, im)
        return aux