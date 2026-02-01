from matplotlib_style import *
import matplotlib.pyplot as plt
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# fonts = sorted({f.name for f in fm.fontManager.ttflist})

first_slection = ["American Typewriter", "Annai MN", "Arima Koshi", "Athelas", "BIZ UDMincho", "BM Hanna Air",
                  "Bradley Hand", "Cochin", "Galvji", "HanziPen SC", "Hannotate SC", "Hubballi"]

second_selection = ["BIZ UDMincho", "Cochin", "Hubballi"]

# for font in fonts:
#     print("font: ", font)
#     plt.figure(figsize=(6, 1))
#     plt.text(0.01, 0.5, font, fontname=font, fontsize=16, va="center")
#     plt.axis("off")
#     plt.show()


## Example plot using custom fonts

x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)

for tfont in [second_selection[-1]]:
    # Set font globally for this figure
    # mpl.rcParams["font.family"] = tfont
    # mpl.rcParams["mathtext.fontset"] = "custom"
    # mpl.rcParams["mathtext.rm"] = tfont
    # mpl.rcParams["mathtext.it"] = tfont
    # mpl.rcParams["mathtext.bf"] = tfont

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="plot")
    thesis_plot(title=f"Font: {tfont}", show=True, close=True, save_fig=False)
