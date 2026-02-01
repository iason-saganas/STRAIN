"""
This file will be depreaceted, I will be keeping strain_tools.plotting up-to-date instead.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import re

__all__ = ["thesis_plot", "thesis_multiplot", "red", "light_red", "blue", "light_blue",
           "lighter_blue", "lightest_blue", "green", "light_green"]

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
                mode="basic",
                xl=r"Time $t$ $\mathrm{[s]}$",
                yl=r"Strain $h$ $\mathrm{[10^{-19}]}$",
                title=None,
                xlim=None,
                ylim=None,
                show=True,
                close=False,
                save_fig=False,
                save_path="",
                custom_ax=None):

    mode_dict = {"basic": (8., 4.)}

    if mode not in mode_dict.keys():
        raise ValueError("Please provide one of following modes for `thesis_plot`:\n, ", mode_dict.keys())
    else:
        md = mode_dict[mode]
        fig_size = tuple(plt.gca().figure.get_size_inches())
        if fig_size != md:
            raise ValueError(f"You are in `thesis_plot` '{mode}' mode, corresponding to a fig size of `{md}', but"
                             f"you passed `{fig_size}` instead.")


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
    if save_fig:
        plt.tight_layout()
        current_date = datetime.datetime.now()
        if save_path == "":
            plt.savefig(f"{current_date}.pdf")
        else:
            plt.savefig(f"{save_path}_{current_date}.pdf")

    if show:
        plt.tight_layout()
        plt.show()
    if close:
        plt.close()


def thesis_multiplot():
    pass