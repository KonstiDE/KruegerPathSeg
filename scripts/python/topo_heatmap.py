import matplotlib.pyplot as plt
import numpy as np

import statistics as s

from matplotlib import colors

def matrix():
    data = [
        # Baseline
        [
            [0.440620385726923,0.440620385726923,0.440620385726923,0.440620385726923,0.440620385726923,0.404196517918995,0.404196517918995,0.404196517918995,0.404196517918995,0.404196517918995],
            [0.319212164198162,0.319212164198162,0.319212164198162,0.319212164198162,0.319212164198162,0.260160289881113,0.260160289881113,0.260160289881113,0.260160289881113,0.260160289881113],
            [0.328229268949258,0.328229268949258,0.328229268949258,0.328229268949258,0.328229268949258,0.277882360036654,0.277882360036654,0.277882360036654,0.277882360036654,0.277882360036654],
            [0.950063177517482,0.950063177517482,0.950063177517482,0.950063177517482,0.950063177517482,0.961083586000679,0.961083586000679,0.961083586000679,0.961083586000679,0.961083586000679],
            [0.232706796047077,0.232706796047077,0.232706796047077,0.232706796047077,0.232706796047077,0.191519232072917,0.191519232072917,0.191519232072917,0.191519232072917,0.191519232072917]
        ],

        [
            [0.440620385726923,0.435363856621776,0.41028521984543,0.43731606217645,0.427012543040734,0.404196517918995,0.387411462611744,0.355077894336821,0.382149088931332,0.40512090383295],
            [0.319212164198162,0.319492796029788,0.326103628289947,0.335273626397464,0.371197417160048,0.260160289881113,0.298132680429341,0.309993934184675,0.316867114291215,0.33205451875663],
            [0.328229268949258,0.331048071623709,0.326067575220356,0.342723479412847,0.361404595215504,0.277882360036654,0.301132216265229,0.294750808812587,0.304315855681351,0.325263979386986],
            [0.950063177517482,0.952845810295699,0.950949504777983,0.952150999725639,0.951133548439323,0.961083586000679,0.960550844071302,0.959381172090868,0.960031451987582,0.960050603617793],
            [0.232706796047077,0.246571407260319,0.24196798788476,0.255268025215329,0.270125587732399,0.191519232072917,0.215022097753555,0.211412178478588,0.217758845590436,0.233090993660601]
        ]
    ]

    change_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    change_matrix_short = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

    for i in range(len(data[0][0])):
        change_matrix[0][i] = percentage_change(data[0][0][i], data[1][0][i], "u") # Precision
        change_matrix[1][i] = percentage_change(data[0][1][i], data[1][1][i], "u") # Recall
        change_matrix[2][i] = percentage_change(data[0][2][i], data[1][2][i], "u") # F1-Score
        change_matrix[3][i] = percentage_change(data[0][3][i], data[1][3][i], "u") # Accuracy
        change_matrix[4][i] = percentage_change(data[0][4][i], data[1][4][i], "u") # IoU

    for i in range(len(change_matrix)):
        change_matrix_short[i] = [change_matrix[i][k] * -100 for k in [1, 2, 3, 4, 6, 7, 8, 9]]

    classes = [
        "Test - B-V2",
        "Test - B-V3",
        "Test - B-V4",
        "Test - B-V5",
        "Test - C-V2",
        "Test - C-V3",
        "Test - C-V4",
        "Test - C-V5"
    ]
    metrics = ["Precision", "Recall", "F1-Score", "Accuracy", "IoU"]

    fig, ax = plt.subplots()

    minimum = np.asarray(change_matrix_short).min()
    maximum = np.asarray(change_matrix_short).max()

    if abs(maximum) > abs(minimum):
        divnorm = colors.TwoSlopeNorm(vmin=-abs(maximum), vcenter=0, vmax=maximum)
    else:
        divnorm = colors.TwoSlopeNorm(vmin=minimum, vcenter=0, vmax=abs(minimum))

    im = ax.imshow(change_matrix_short, cmap='coolwarm_r', norm=divnorm)
    im_ratio = 5 / 13

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.075 * im_ratio)
    cbar.ax.set_ylabel(["Percentage Improvement over V1"], rotation=-90, va="bottom")

    for y in range(len(change_matrix_short)):
        for x in range(len(change_matrix_short[0])):
            ax.text(x, y, "{:.2f}".format(change_matrix_short[y][x]), horizontalalignment='center',
                    verticalalignment='center', size=9)

    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(metrics)), labels=metrics)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45)

    fig.tight_layout()

    plt.xlabel("* B-V#, C-V# ≡ Statically-Buffered-V#, Optically-Clustered-V#", labelpad=20)
    plt.savefig("heatmap_topological.pdf", dpi=600)
    plt.show()


def percentage_change(v2_plain, v2_attention, direction):
    if direction == "u":
        return float(v2_plain - v2_attention)
    elif direction == "d":
        return float(v2_attention - v2_plain)
    else:
        raise ValueError("Invalid direction; use 'u' for upward or 'd' for downward.")


if __name__ == '__main__':
    matrix()
