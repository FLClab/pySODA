import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import utils
import sklearn.metrics
from scipy.optimize import curve_fit
import scipy.optimize
plt.style.use('dark_background')

UPPER_LEFT = [0.0, 1.0]


def compute_auotc(X, Y):
    return sklearn.metrics.auc(X, Y)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='white')


def exponential_decay(x, A, B, D):
    return A * (1 - exp((-B * x))) + D


def normalize_0_1(distances):
    if np.max(distances) - np.min(distances) == 0:
        return distances, np.max(distances), np.min(distances)
    else:
        normalized_distances = (distances - np.min(distances)) / \
            (np.max(distances) - np.min(distances))
        return normalized_distances, np.max(distances), np.min(distances)


def denormalize(distances, max_val, min_val):
    denorm_data = (distances * (max_val - min_val)) + min_val
    return denorm_data


def compute_characteristic_distance(x, y):
    """
    Computes a single number metric that is giving a distance. 
    This distance corresponds to the distance with which the OTC value is closest to 1.0

    Params:
    ----------
    x (array): distances used to compute the OTC values
    y (array): OTC values

    Returns: 
    ----------
    x_P1 (float): single number metric
    """
    x, max_x, min_x = normalize_0_1(x)
    params, _ = curve_fit(exponential_decay, x, y)
    xfit = np.linspace(np.min(x), np.max(x), 1000)
    yfit = exponential_decay(xfit, *params)

    coords = list(zip(xfit, yfit))

    point = np.array((0, 1))
    distances = np.linalg.norm(coords - point, axis=1)
    min_index = np.argmin(distances)
    closest_p = coords[min_index]
    x, y = closest_p[0], closest_p[1]

    return (x, y), (xfit, yfit), (max_x, min_x)


def main_auotc():
    gfp, nt, rescue, shrna, shrescue, random, coloc, distances = utils.load_actin_data()
    plko, shfus, random_als, coloc_als, distances_als = utils.load_als_data()

    data = [gfp, nt, rescue, shrna]
    labels = ['GFP', 'NT', 'Rescue', 'shRNA-BCaMKII', 'Random', 'Colocalized']
    colors = ['limegreen', 'gainsboro', 'lightblue',
              'lightcoral', 'magenta', 'gold']
    x_coords = [0, 1, 2, 3, 4, 5]
    auotc_vals = []
    fig = plt.figure()
    for i, d in enumerate(data):
        auotc = compute_auotc(distances, d[0])
        auotc_vals.append(auotc)
        x, metric, (xfit, yfit) = compute_characteristic_distance(
            distances, d[0])
        plt.plot(xfit * 20 * 20, yfit, ls='--')
        plt.plot(distances, d[0])
    fig.savefig('./temp.png')

    # Plot
    # auotc_vals = [round(item, 3) for item in auotc_vals]
    # fig, ax = plt.subplots()
    # rects = ax.bar(x_coords, auotc_vals, color=colors)
    # autolabel(rects, ax)
    # ax.set_xticks(x_coords, labels, rotation=45)
    # ax.set_ylim([0, 20])
    # ax.set_title('Actin - CaMKII AUOTC')
    # ax.set_ylabel('AUOTC')
    # plt.tight_layout()
    # fig.savefig('./results/data_bichette/AUOTC.png')


def main_distance():
    plko, pd, random, coloc, distances = utils.load_pd_data()
    data = [plko, pd, random, coloc]
    labels = ['GFP', 'NT', 'Rescue', 'shRNA', 'Colocalized']
    colors = ['limegreen', 'gainsboro', 'lightblue',
              'lightcoral', 'gold']

    fig = plt.figure()
    for i, d in enumerate(data):
        closest_p, (x_fit, y_fit), (max_x, min_x) = compute_characteristic_distance(
            distances, d[0])
        temp_dist, min_dist, max_dist = normalize_0_1(distances)
        plt.plot(x_fit, y_fit, ls='--',
                 color=colors[i])
        plt.plot(temp_dist,
                 d[0], color=colors[i], label=labels[i])
        plt.plot(closest_p[0], closest_p[1],
                 color=colors[i], marker='o', ms=10, mew=3)
        metric_val = denormalize(closest_p[0], max_x, min_x)
        print("X Value for {}: {}".format(labels[i], metric_val * 20))
    plt.xlabel('Normalized distance', fontsize=14)
    plt.ylabel('OTC', fontsize=14)
    plt.legend(fontsize=12)
    fig.savefig('./temp.png')


def main():
    is_auotc = False
    if is_auotc:
        main_auotc()
    else:
        main_distance()


if __name__ == "__main__":
    main()
