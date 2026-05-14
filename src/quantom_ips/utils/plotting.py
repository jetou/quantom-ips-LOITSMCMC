import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_2d_event_histogram(
    real_events,
    fake_events,
    plot_style="bmh",
    save_file=None,
    fig_title=None,
    labels=("x", "y"),
):
    with plt.style.context(plot_style):
        # Compute the residuals:
        residuals = real_events - fake_events

        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        if fig_title is not None:
            fig.suptitle(fig_title)
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        axs[0, 0].hist(
            real_events[:, 0], 100, histtype="step", linewidth=1.5, label="Real"
        )
        axs[0, 0].hist(
            fake_events[:, 0], 100, histtype="step", linewidth=1.5, label="Gen."
        )
        axs[0, 0].set_xlabel("Observable " + labels[0])
        axs[0, 0].set_ylabel("Entries")
        axs[0, 0].legend()

        axs[0, 1].hist(residuals[:, 0], 100, histtype="step", linewidth=1.5)
        axs[0, 1].set_xlabel("Residual " + labels[0])
        axs[0, 1].set_ylabel("Entries")

        axs[1, 0].hist(
            real_events[:, 1], 100, histtype="step", linewidth=1.5, label="Real"
        )
        axs[1, 0].hist(
            fake_events[:, 1], 100, histtype="step", linewidth=1.5, label="Gen."
        )
        axs[1, 0].set_xlabel("Observable " + labels[1])
        axs[1, 0].set_ylabel("Entries")
        axs[1, 0].legend()

        axs[1, 1].hist(residuals[:, 1], 100, histtype="step", linewidth=1.5)
        axs[1, 1].set_xlabel("Residual " + labels[1])
        axs[1, 1].set_ylabel("Entries")
        fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()
    plt.close(fig)
    plt.clf()


def plot_2d_event_correlations(
    real_events,
    fake_events,
    plot_style="bmh",
    save_file=None,
    fig_title=None,
    labels=("x", "y"),
):
    with plt.style.context(plot_style):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
        if fig_title is not None:
            fig.suptitle(fig_title)
        axs[0].set_title("Real")
        axs[0].hist2d(real_events[:, 0], real_events[:, 1], 100, norm=LogNorm())
        axs[0].set_xlabel("Observable " + labels[0])
        axs[0].set_ylabel("Observable " + labels[1])

        axs[1].set_title("Gen.")
        axs[1].hist2d(fake_events[:, 0], fake_events[:, 1], 100, norm=LogNorm())
        axs[1].set_xlabel("Observable " + labels[0])
        fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()
    plt.close(fig)
    plt.clf()


def plot_2d_pdfs(
    pdfs,
    plot_style="bmh",
    save_file=None,
    fig_title=None,
    x_ticks=None,
    x_ticklabels=None,
    y_ticks=None,
    y_ticklabels=None,
):
    assert len(pdfs) == 2
    pdfs.append(np.log10(np.abs(pdfs[1] - pdfs[0]) / pdfs[0]))
    with plt.style.context(plot_style):
        fig, axs = plt.subplots(1, 3)
        if fig_title is not None:
            fig.suptitle(fig_title)
        ax_titles = ["Real", "Gen.", "Gen. - Real"]
        for i, ax in enumerate(axs):
            pcm = ax.pcolormesh(pdfs[i])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(pcm, cax=cax)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            if x_ticklabels is not None:
                ax.set_xticklabels(x_ticklabels)

            if y_ticks is not None:
                ax.set_yticks(y_ticks)
            if y_ticklabels is not None:
                ax.set_yticklabels(y_ticklabels)

            ax.set_title(ax_titles[i])

        fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()
    plt.close(fig)
    plt.clf()


def plot_1d_event_histogram(
    real_events,
    fake_events,
    plot_style="bmh",
    save_file=None,
    fig_title=None,
    labels="x",
):
    with plt.style.context(plot_style):
        # Compute the residuals:
        residuals = real_events - fake_events

        fig, axs = plt.subplots(1, 2, figsize=(6, 6))
        fig.subplots_adjust(wspace=0.35)
        if fig_title is not None:
            fig.suptitle(fig_title)

        axs[0].hist(real_events, 100, histtype="step", linewidth=1.5, label="Real")
        axs[0].hist(fake_events, 100, histtype="step", linewidth=1.5, label="Gen.")
        axs[0].set_xlabel("Observable " + labels)
        axs[0].set_ylabel("Entries")
        axs[0].legend()

        axs[1].hist(residuals, 100, histtype="step", linewidth=1.5)
        axs[1].set_xlabel("Residual " + labels)
        axs[1].set_ylabel("Entries")

        fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()
    plt.close(fig)
    plt.clf()


def plot_1d_pdfs(
    pdfs,
    plot_style="bmh",
    save_file=None,
    fig_title=None,
    x_ticks=None,
    x_ticklabels=None,
):
    assert len(pdfs) == 2
    with plt.style.context(plot_style):
        fig, axs = plt.subplots(1, 2)
        if fig_title is not None:
            fig.suptitle(fig_title)
        ax_titles = ["Real", "Gen."]
        for i, ax in enumerate(axs):
            pcm = ax.pcolormesh(pdfs[i])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(pcm, cax=cax)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            if x_ticklabels is not None:
                ax.set_xticklabels(x_ticklabels)

            ax.set_title(ax_titles[i])
        fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()
    plt.close(fig)
    plt.clf()


def plot_single_image(
    image,
    plot_style="bmh",
    save_file=None,
    fig_title=None,
    x_ticks=None,
    x_ticklabels=None,
):
    with plt.style.context(plot_style):
        fig, axs = plt.subplots()
        if fig_title is not None:
            fig.suptitle(fig_title)

            pcm = axs.pcolormesh(image)
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(pcm, cax=cax)
            axs.set_aspect("equal")
            axs.set_xticks([])
            axs.set_yticks([])

            if x_ticks is not None:
                axs.set_xticks(x_ticks)
            if x_ticklabels is not None:
                axs.set_xticklabels(x_ticklabels)

        fig.tight_layout()
    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()
    plt.close(fig)
    plt.clf()


def plot_gan_history(
    data,
    reference_point=0,
    shift_type=None,
    plot_style="bmh",
    save_file=None,
    epochs=None,
):
    with plt.style.context(plot_style):
        for d in data:
            val = data[d]
            if shift_type == "sub":
                val = val - reference_point
            elif shift_type == "div":
                val = val / reference_point

            if epochs is not None:
                plt.plot(epochs, val, label=d)
            else:
                plt.plot(val, label=d)
        if shift_type == "sub":
            plt.axhline(y=0, linestyle=":", color="k")
            plt.yscale("asinh")
            low, high = plt.ylim()
            bound = max(abs(low), abs(high))
            plt.ylim(-bound, bound)
        elif shift_type == "div":
            plt.axhline(y=1, linestyle=":", color="k")
        else:
            plt.axhline(y=reference_point, linestyle=":", color="k")

        plt.legend()
        plt.xlabel("Epochs")
        if shift_type is not None:
            plt.ylabel("Normalized Loss")
        else:
            plt.ylabel("Loss")
        plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
    plt.close()
    plt.clf()
