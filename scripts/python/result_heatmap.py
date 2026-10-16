import matplotlib.pyplot as plt
import numpy as np

import statistics as s

from matplotlib import colors

def matrix():
    data = [
        # Baseline
        [
            [0.2655469,  0.2655469,  0.2655469, 0.2655469, 0.2655469,        0.2858349,   0.2858349,   0.2858349, 0.2858349, 0.2858349], # DICEsb
            [0.08028221, 0.08028221, 0.08028221, 0.08028221, 0.08028221,     0.06605243,  0.06605243,  0.06605243, 0.06605243, 0.06605243], # MAE
            [0.01623168, 0.01623168, 0.01623168, 0.01623168, 0.01623168,     0.014076613, 0.014076613, 0.014076613, 0.014076613, 0.014076613], # MedAE
            [0.1942439,  0.1942439,  0.1942439, 0.1942439, 0.1942439,        0.1723842,   0.1723842,   0.1723842, 0.1723842, 0.1723842], # RMSE
            [0.4296520,  0.4296520,  0.4296520, 0.4296520, 0.4296520,        0.4610217,   0.4610217,   0.4610217, 0.4610217, 0.4610217], # SSIM
            [0.4367897,  0.4367897,  0.4367897, 0.4367897, 0.4367897,        0.4055141,   0.4055141,   0.4055141, 0.4055141, 0.4055141]  # ZNCC
        ],

        [
            [0.2655469,  0.2489308, 0.2548098, 0.2496713, 0.2480763,       0.2858349,   0.2765857, 0.2737751, 0.2673838, 0.2695454], # DICEsb
            [0.08028221, 0.07535406, 0.07947304, 0.07799057, 0.07704716,   0.06605243,  0.06289689, 0.06171785, 0.06022337, 0.06299317], # MAE
            [0.01623168, 0.01657103, 0.01743363, 0.01748543, 0.01358295,   0.014076613, 0.010431920, 0.011612688, 0.008517256, 0.010119966], # MedAE
            [0.1942439,  0.1865776, 0.1899007, 0.1878820, 0.1890921,       0.1723842,   0.1710973, 0.1711978, 0.1693185, 0.1703470], # RMSE
            [0.4296520,  0.4051654, 0.3865578, 0.3957437, 0.4651332,       0.4610217,   0.4979164, 0.4675985, 0.5517043, 0.5123487], # SSIM
            [0.4367897,  0.4682947, 0.4556158, 0.4677088, 0.4720259,       0.4055141,   0.4227881, 0.4250275, 0.4391240, 0.4402666]  # ZNCC
        ]
    ]

    change_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

    for i in range(len(data[0][0])):
        change_matrix[0][i] = percentage_change(data[0][0][i], data[1][0][i], "d") # DICEsb
        change_matrix[1][i] = percentage_change(data[0][1][i], data[1][1][i], "d") # MAE
        change_matrix[2][i] = percentage_change(data[0][2][i], data[1][2][i], "d") # MedAE
        change_matrix[3][i] = percentage_change(data[0][3][i], data[1][3][i], "d") # RMSE
        change_matrix[4][i] = percentage_change(data[0][4][i], data[1][4][i], "u") # SSIM
        change_matrix[5][i] = percentage_change(data[0][5][i], data[1][5][i], "u") # ZNCC

    print(change_matrix)

    for i in range(len(change_matrix)):
        mean_line = s.mean(change_matrix[i])
        std_line = s.stdev(change_matrix[i])
        if std_line == 0:
            std_line = 1

        for t in range(len(change_matrix[i])):
            if change_matrix[i][t] <= 0:
                change_matrix[i][t] = -(abs(change_matrix[i][t]) - mean_line) / std_line
            else:
                change_matrix[i][t] = (change_matrix[i][t] - mean_line) / std_line

    for i in range(len(change_matrix)):
        change_matrix_short[i] = [change_matrix[i][k] * -1 for k in [1, 2, 3, 4, 6, 7, 8, 9]]

    classes = [
        "Buffer - V2",
        "Buffer - V3",
        "Buffer - V4",
        "Buffer - V5",
        "Cluster - V2",
        "Cluster - V3",
        "Cluster - V4",
        "Cluster - V5"
    ]
    metrics = ["DICEsb", "MAE", "MedAE", "RMSE", "SSIM", "ZNCC"]

    fig, ax = plt.subplots()

    minimum = np.asarray(change_matrix_short).min()
    maximum = np.asarray(change_matrix_short).max()

    if abs(maximum) > abs(minimum):
        divnorm = colors.TwoSlopeNorm(vmin=-abs(maximum), vcenter=0, vmax=maximum)
    else:
        divnorm = colors.TwoSlopeNorm(vmin=minimum, vcenter=0, vmax=abs(minimum))

    im = ax.imshow(change_matrix_short, cmap='coolwarm_r', norm=divnorm)
    im_ratio = 5 / 6.85

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.047 * im_ratio)
    cbar.ax.set_ylabel(["Z-Score Improvement over V1"], rotation=-90, va="bottom")

    for y in range(len(change_matrix_short)):
        for x in range(len(change_matrix_short[0])):
            ax.text(x, y, "{:.2f}".format(change_matrix_short[y][x]), horizontalalignment='center',
                    verticalalignment='center', size=9)

    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(metrics)), labels=metrics)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", rotation=45)

    fig.tight_layout()
    plt.savefig("heatmap.pdf", dpi=600)
    plt.show()


def percentage_change(v2_plain, v2_attention, direction):
    if direction == "d":
        return float(v2_attention - v2_plain)
    else:
        return -float(v2_attention - v2_plain)


if __name__ == '__main__':
    matrix()
