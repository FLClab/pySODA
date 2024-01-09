import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
import random
from main import compute_OTC
plt.style.use('dark_background')


def generate_randomly_distributed(size=128, num_regions_a=50, multiplier=3.0):
    img_a = np.zeros((size, size))
    img_b = np.zeros((size, size))
    # num_regions_b = int(num_regions_a * b_mult) # for checking density
    num_regions_b = num_regions_a  # for checking size
    for _ in range(num_regions_a):
        x_c = np.random.randint(0, size)
        y_c = np.random.randint(0, size)
        radius = round(np.random.normal(loc=3, scale=1))
        if radius == 0:
            radius = 1
        rr, cc = skimage.draw.disk(
            (x_c, y_c), radius=radius, shape=img_a.shape)
        for r in rr:
            for c in cc:
                img_a[r, c] = random.random()
    for _ in range(num_regions_b):
        x_c = np.random.randint(0, size)
        y_c = np.random.randint(0, size)
        radius = round(np.random.normal(loc=multiplier, scale=1))
        if radius == 0:
            radius = 1
        rr, cc = skimage.draw.disk(
            (x_c, y_c), radius=radius, shape=img_b.shape)
        for r in rr:
            for c in cc:
                img_b[r, c] = random.random()
    return img_a, img_b


def generate_colocalized(size=128, num_regions_a=50, multiplier=3.0):
    img_a = np.zeros((size, size))
    img_b = np.zeros((size, size))
    num_regions_b = num_regions_a
    coords_a = []
    # Populating image a with a random distribution of proteins
    for _ in range(num_regions_a):
        x_a = np.random.randint(0, size)
        y_a = np.random.randint(0, size)
        coords_a.append((x_a, y_a))
        # rmin_a, rmaj_a = np.random.randint(1, 4), np.random.randint(1, 4)
        radius = round(np.random.normal(loc=multiplier, scale=1))
        if radius == 0:
            radius = 1
        rr_a, cc_a = skimage.draw.disk(
            (x_a, y_a), radius=radius, shape=img_a.shape)
        for r in rr_a:
            for c in cc_a:
                img_a[r, c] = random.random()

    for _ in range(num_regions_b):
        prob = random.uniform(0, 1)
        if prob < 0.10:
            x_b = np.random.randint(0, size)
            y_b = np.random.randint(0, size)
        else:
            idx = np.random.randint(0, len(coords_a))
            (temp_x, temp_y) = coords_a[idx]
            x_b = round(np.random.normal(loc=temp_x, scale=0.5))
            y_b = round(np.random.normal(loc=temp_y, scale=0.5))
        # rmin_b, rmaj_b = np.random.randint(1, 4), np.random.randint(1, 4)
        radius = round(np.random.normal(loc=multiplier, scale=1))
        if radius == 0:
            radius = 1
        rr_b, cc_b = skimage.draw.disk(
            (x_b, y_b), radius=radius, shape=img_b.shape)
        for r in rr_b:
            for c in cc_b:
                img_b[r, c] = random.random()
    return img_a, img_b


def create_crops(num_crops, multiplier):
    crops_a, crops_b = [], []
    for i in range(num_crops):
        img_a, img_b = generate_randomly_distributed(multiplier=multiplier)
        crops_a.append(img_a)
        crops_b.append(img_b)
    return crops_a, crops_b


def plot_results(otcs, confidences, distances):
    xtick_locs = [1, 5, 10, 15, 20]
    xtick_labels = [str(item * 20) for item in xtick_locs]
    colors = ['lightblue', 'lightcoral', 'magenta', 'gold', 'limegreen']
    labels = ['0.5', '0.75', '1.0', '1.5', '2.0']
    fig = plt.figure()
    for i in range(len(otcs)):
        o, c = otcs[i], confidences[i]
        plt.plot(distances, o, color=colors[i], label=labels[i])
        plt.fill_between(distances, o - c, o + c,
                         facecolor=colors[i], alpha=0.3)
    plt.xlabel('Distance (nm)', fontsize=16)
    plt.xticks(ticks=xtick_locs, labels=xtick_labels)
    plt.ylabel('OTC', fontsize=16)
    plt.legend(fontsize=12)
    plt.title('OTC - colocalized distributions')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.savefig('./colocalized_density.png', bbox_inches='tight')
    fig.savefig('./colocalized_density.pdf',
                transparent=True, bbox_inches='tight')


def sanity_check(ch0, ch1):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(ch0, cmap='hot')
    axs[1].imshow(ch1, cmap='hot')
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig('./sanity_check_size_random5.png', bbox_inches='tight')


def main():
    img0, img1 = generate_randomly_distributed(multiplier=5.0)
    sanity_check(img0, img1)
    # otcs, confidences, distances = [], [], None
    # # multipliers = [0.50, 0.75, 1.0, 1.50, 2.0] # for checking density
    # multipliers = [1.0, 2.0, 3.0, 4.0, 5.0]  # for checking size
    # for m in multipliers:
    #     crops_a, crops_b = create_crops(num_crops=1000, multiplier=m)
    #     otc, std, confidence, distances = compute_OTC(crops=(crops_a, crops_b))
    #     otcs.append(otc)
    #     confidences.append(confidence)
    # np.savez('./var_size_random', otc_list=otcs,
    #          confidence_list=confidences, distances=distances)
    # plot_results(otcs, confidences, distances)


if __name__ == "__main__":
    main()
