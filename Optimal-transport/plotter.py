import os
import matplotlib.pyplot as plt
import numpy
import utils
plt.style.use('dark_background')


def plot_OTC(*data):
    xtick_locs = [1, 5, 10, 15, 20]
    xtick_labels = [str(item * 20) for item in xtick_locs]
    labels = ['Block', 'KCl', 'Random', 'Colocalized']
    colors = ['lightblue', 'lightcoral', 'magenta', 'gold']
    fig = plt.figure()
    distances = data[0]['distances']
    for i, d in enumerate(data):
        plt.plot(distances, d['otc'], color=colors[i], label=labels[i])
        plt.fill_between(distances, d['otc'] - d['confidence'],
                         d['otc'] + d['confidence'], facecolor=colors[i], alpha=0.3)
    plt.xlabel('Distance (nm)', fontsize=16)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=16)
    plt.legend(fontsize=14)
    plt.title('Actin - Cofilin')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0, 10])
    fig.savefig('./figures/ActinCofilin/block_kcl_otc.png',
                bbox_inches='tight')
    fig.savefig('./figures/ActinCofilin/block_kcl.pdf',
                transparent=True, bbox_inches='tight')


def main():
    block, kcl, random, coloc = utils.load_actin_cofilin()
    plot_OTC(block, kcl, random, coloc)


if __name__ == "__main__":
    main()
