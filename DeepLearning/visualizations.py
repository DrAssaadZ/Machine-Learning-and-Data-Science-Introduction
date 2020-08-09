"""
This file containes helper function, mainly for visualization.
Author: ksopyla (https://github.com/ksopyla)
"""

import matplotlib.pyplot as plt
import seaborn as sns


def losses_accuracies_plots(confMat, train_losses, train_acc, test_losses, test_acc, plot_title="Loss, train acc, test acc",
                            step=100):

    training_iters = len(train_losses)
    # iters_steps
    iter_steps = [step * k for k in range(training_iters)]

    imh = plt.figure(1, figsize=(15, 14), dpi=160)

    final_acc = test_acc[-1]
    img_title = "{}, test acc={:.4f}".format(plot_title, final_acc)
    imh.suptitle(img_title)
    plt.subplot(231)
    plt.semilogy(iter_steps, train_losses, '-g', label='Trn Loss')
    plt.title('Train Loss ')
    plt.subplot(232)
    plt.plot(iter_steps, train_acc, '-r', label='Trn Acc')
    plt.title('Train Accuracy')

    plt.subplot(234)
    plt.semilogy(iter_steps, test_losses, '-g', label='Tst Loss')
    plt.title('Test Loss')
    plt.subplot(235)
    plt.plot(iter_steps, test_acc, '-r', label='Tst Acc')
    plt.title('Test Accuracy')

    plt.subplot(233)
    plt.imshow(confMat, cmap='GnBu')
    plt.colorbar()
    plt.title('Confusion matrix HeatMap')

    # plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # for saving the plots
    # plot_file = "./plots/{}.png".format(plot_title.replace(" ", "_"))
    # plt.savefig(plot_file)
    plt.show()




