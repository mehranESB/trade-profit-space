import matplotlib.pyplot as plt
import numpy as np


def plot_bar_chart(ax, data: tuple, linewidth=1):
    """
    Plots a bar chart of market data (OHLC) on the given axis with open and close represented as small horizontal lines.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis on which to plot the chart.
    - data (tuple): Contains the market data ("Open", "High", "Low", "Close") in that order.
    - linewidth (float, optional): The line width for the plot. Default is 1.
    """
    op, hi, lo, cl = data

    # Plot the high-low range as vertical bars
    ax.vlines(np.arange(len(op)), lo, hi, color="black", linewidth=linewidth)

    # Plot small horizontal lines for Open (on the left) and Close (on the right)
    ax.hlines(
        op,
        np.arange(len(op)) - 0.2,
        np.arange(len(op)),
        color="black",
        linewidth=linewidth,
    )  # Open line on the left
    ax.hlines(
        cl,
        np.arange(len(op)),
        np.arange(len(op)) + 0.2,
        color="black",
        linewidth=linewidth,
    )  # Close line on the right
