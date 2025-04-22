from .utils import plot_bar_chart
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# constant as infinity
INF = 1e5


class SpaceMaker:
    def __init__(self, market_data: pd.DataFrame, max_holds: int = 15):
        """
        Initializes the SpaceMaker instance.

        Parameters:
        - market_data (pd.DataFrame): Market OHLC data used to compute trade results.
        - max_holds (int): Maximum number of bars to hold a position. Must be at least 1.
        """

        # Ensure required OHLC columns are present
        required_columns = {"Open", "High", "Low", "Close"}
        if not required_columns.issubset(market_data.columns):
            raise ValueError(f"market_data must contain columns: {required_columns}")

        # Market data is OHLC data that needs to be used to find trade results
        self.op = market_data["Open"].to_numpy(dtype=np.float32)
        self.hi = market_data["High"].to_numpy(dtype=np.float32)
        self.lo = market_data["Low"].to_numpy(dtype=np.float32)
        self.cl = market_data["Close"].to_numpy(dtype=np.float32)

        # max_holds means number of bars to hold the position
        self.max_holds = max(max_holds, 1)

        # Number of bars in the market data
        self.nBars = len(market_data)

    def __len__(self):
        return self.nBars

    def __getitem__(self, index):
        # Ensure the index is within the valid range
        if not (0 <= index < self.nBars):
            raise IndexError(
                f"Index {index} is out of bounds for market data with {self.nBars} bars."
            )

        # Determine end of slice, limited by data length
        to_index = min(index + self.max_holds, self.nBars)

        # Slice the OHLC data
        op = self.op[index:to_index]
        hi = self.hi[index:to_index]
        lo = self.lo[index:to_index]
        cl = self.cl[index:to_index]

        return ProfitSpace(op, hi, lo, cl)


class ProfitSpace:
    def __init__(self, op: np.ndarray, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray):
        """
        Initializes the ProfitSpace instance.

        Parameters:
        - op (np.ndarray): The opening prices for each bar from the current bar to future bars.
        - hi (np.ndarray): The highest prices for each bar from the current bar to future bars.
        - lo (np.ndarray): The lowest prices for each bar from the current bar to future bars.
        - cl (np.ndarray): The closing prices for each bar from the current bar to future bars.
        """

        # Copy input arrays to avoid modifying the original data
        self.op = op.copy()
        self.hi = hi.copy()
        self.lo = lo.copy()
        self.cl = cl.copy()

        # Get the highest high and lowest low across the bars from current bar to future bars
        self.hh = np.maximum.accumulate(self.hi)
        self.ll = np.minimum.accumulate(self.lo)

        # Get number of stored bars
        self.nBars = len(op)

        # The price that order is executed
        self.exeprice = self.op[0]

        # Gefine the regions
        self.define_regions()

    def define_buy_region(self):
        """
        Defines a polygon-like region between highest high (hh) and lowest low (ll),
        representing the space in which a buy order would be profitable based on market movement.

        The region is stored as a shapely.geometry.Polygon object in self.buyreg.
        """
        hh = self.hh - self.exeprice
        ll = self.ll - self.exeprice

        points = [(hh[0], ll[0])]

        for i in range(1, self.nBars):
            # Check if both hh and ll changed at the same time
            same_time_change = (hh[i - 1] != hh[i]) and (ll[i - 1] != ll[i])

            if not same_time_change:
                points.append((hh[i], ll[i]))
            else:
                # Create a right-angle (90-degree) path to avoid ambiguity
                points.append((hh[i - 1], ll[i]))
                points.append((hh[i], ll[i]))

        # Close the polygon path
        points.append((hh[-1], -INF))
        points.append((0.0, -INF))
        points.append((0.0, ll[0]))

        # Store the region as a Polygon
        self.buyreg = Polygon(points)

    def define_sell_region(self):
        """
        Defines a polygon-like region between highest high (hh) and lowest low (ll),
        representing the space in which a sell order would be profitable based on market movement.

        The region is stored as a shapely.geometry.Polygon object in self.sellreg.
        """
        hh = self.hh - self.exeprice
        ll = self.ll - self.exeprice

        points = [(hh[0], ll[0])]

        for i in range(1, self.nBars):
            # Check if both hh and ll changed at the same time
            same_time_change = (hh[i - 1] != hh[i]) and (ll[i - 1] != ll[i])

            if not same_time_change:
                points.append((hh[i], ll[i]))
            else:
                # Create a right-angle (90-degree) path to avoid ambiguity
                points.append((hh[i], ll[i - 1]))
                points.append((hh[i], ll[i]))

        # Close the polygon path
        points.append((INF, ll[-1]))
        points.append((INF, 0.0))
        points.append((hh[0], 0.0))

        # Store the region as a Polygon
        self.sellreg = Polygon(points)

    def define_unknown_region(self):
        """
        Defines the ambiguous (unknown) region in profit space where both buy/sell outcomes
        change simultaneously — the zone that doesn't clearly belong to either side.

        Stores the result in self.unkreg as a shapely Polygon.
        """
        hh = self.hh - self.exeprice
        ll = self.ll - self.exeprice

        points_R = [(hh[0], 0.0), (hh[0], ll[0])]
        points_L = [(0.0, ll[0]), (hh[0], ll[0])]

        for i in range(1, self.nBars):
            # Check if both highest high and lowest low change at the same time
            same_time_change = (hh[i - 1] != hh[i]) and (ll[i - 1] != ll[i])

            if not same_time_change:
                # Regular step (narrow path)
                points_R.append((hh[i], ll[i]))
                points_L.append((hh[i], ll[i]))
            else:
                # Create 90-degree zig-zag path to capture ambiguity
                # From right side
                points_R.append((hh[i], ll[i - 1]))
                points_R.append((hh[i], ll[i]))
                # From left side
                points_L.append((hh[i - 1], ll[i]))
                points_L.append((hh[i], ll[i]))

        # Reverse left-side points to walk back properly and close the shape
        points_L.reverse()

        # Close polygon: right path  → left path → base
        full_points = points_R + points_L + [(0.0, 0.0)]

        # Store the polygon
        self.unkreg = Polygon(full_points)

    def define_expire_region(self):
        """
        Defines the 'expiration region' in profit space, representing trades that reach
        the maximum hold time without triggering profit or stop conditions.

        The region starts at the last (hh, ll) point and extends infinitely to the right
        and downward, forming an open-ended rectangle.

        Stores the result in self.expreg as a shapely Polygon.
        """
        hh = self.hh - self.exeprice
        ll = self.ll - self.exeprice

        # Use the last bar's hh and ll values
        end_hh = hh[-1]
        end_ll = ll[-1]

        # Create polygon points going clockwise
        points = [
            (end_hh, end_ll),  # bottom-left corner
            (INF, end_ll),  # bottom-right
            (INF, -INF),  # far bottom-right
            (end_hh, -INF),  # far bottom-left
        ]

        # Create and store the expiration region polygon
        self.expreg = Polygon(points)

    def define_regions(self):
        """
        Defines all strategic regions in the profit space:
        - Buy region (self.buyreg)
        - Sell region (self.sellreg)
        - Unknown/ambiguous region (self.unkreg)
        - Expiration region (self.expreg)

        Each region is stored as a Shapely Polygon object.
        """
        self.define_buy_region()
        self.define_sell_region()
        self.define_unknown_region()
        self.define_expire_region()

    def plot_price_range(self, ax=None, chandle: bool = False):
        """
        Plots the highest high and lowest low over time, optionally with candlestick bars.

        Parameters:
        - ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates one if None.
        - chandle (bool): If True, also plots candlestick bars using plot_bar_chart.
        """
        # Create a new axis if one isn't provided
        if ax is None:
            fig, ax = plt.subplots()

        # Plot candlestick bars if requested
        if chandle:
            ohlc = (self.op, self.hi, self.lo, self.cl)
            plot_bar_chart(ax, ohlc)

        # Plot the highest high and lowest low
        ax.plot(
            range(len(self.hh)),
            self.hh,
            label="Highest High",
            color="blue",
            linewidth=1,
        )
        ax.plot(
            range(len(self.ll)), self.ll, label="Lowest Low", color="red", linewidth=1
        )

        # Axis labels and legend
        ax.set_xlabel("Bar Index")
        ax.set_ylabel("Price")
        ax.set_title("Price Range")
        ax.legend()
        ax.grid(True)

    def plot_profit_space(self, ax=None):
        """
        Plots the profit space defined by self.buyreg and self.sellreg as shaded polygons.

        Parameters:
        - ax (matplotlib.axes.Axes, optional): Axis to draw on. Creates a new one if None.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Plot Buy Region
        buyreg_patch = MplPolygon(
            list(self.buyreg.exterior.coords),
            closed=True,
            facecolor="blue",
            edgecolor="black",
            alpha=0.5,
            label="Buy Region",
        )
        ax.add_patch(buyreg_patch)

        # Plot Sell Region
        sellreg_patch = MplPolygon(
            list(self.sellreg.exterior.coords),
            closed=True,
            facecolor="red",
            edgecolor="black",
            alpha=0.5,
            label="Sell Region",
        )
        ax.add_patch(sellreg_patch)

        # Plot Unknown Region
        unkreg_patch = MplPolygon(
            list(self.unkreg.exterior.coords),
            closed=True,
            facecolor="yellow",
            edgecolor="black",
            alpha=0.5,
            label="Unknown Region",
        )
        ax.add_patch(unkreg_patch)

        # Plot Expiration Region
        expreg_patch = MplPolygon(
            list(self.expreg.exterior.coords),
            closed=True,
            facecolor="silver",
            edgecolor="black",
            alpha=0.5,
            label="Expire Region",
        )
        ax.add_patch(expreg_patch)

        # Axis limits
        max_xlim = max(self.hh - self.exeprice)
        min_ylim = min(self.ll - self.exeprice)
        max_len = max(max_xlim, -min_ylim)
        ax.set_xlim(0.0, 1.1 * max_len)
        ax.set_ylim(-1.1 * max_len, 0.0)

        # Visual polish
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("upper target from order price")
        ax.set_ylabel("lower target from order price")
        ax.set_title("Profit Space")
        ax.legend()
        ax.grid(True)

    def plot_map_targets(self):
        fig, (ax_left, ax_right) = plt.subplots(1, 2)

        self.plot_price_range(ax_left, chandle=True)
        self.plot_profit_space(ax_right)

        # Horizontal lines on ax_left
        utline = ax_left.axhline(y=self.exeprice, color="red", linestyle="--")
        ltline = ax_left.axhline(y=self.exeprice, color="blue", linestyle="--")

        # Middle x-coordinate of ax_left for annotation placement
        xlim = ax_left.get_xlim()
        mid_x = (xlim[0] + xlim[1]) / 2

        # Annotations that follow the lines
        ut_annot = ax_left.text(
            mid_x,
            self.exeprice,
            "Upper Target",
            ha="center",
            va="bottom",
            color="red",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6),
        )
        lt_annot = ax_left.text(
            mid_x,
            self.exeprice,
            "Lower Target",
            ha="center",
            va="top",
            color="blue",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6),
        )

        # Crosshair lines in ax_right
        vline_cross = ax_right.axvline(color="gray", linestyle="--")
        hline_cross = ax_right.axhline(color="gray", linestyle="--")

        def on_move(event):
            if (
                event.inaxes == ax_right
                and event.xdata is not None
                and event.ydata is not None
            ):
                ut = max(event.xdata, 0.0)
                lt = min(event.ydata, 0.0)

                # Update target lines in ax_left
                new_ut_y = self.exeprice + ut
                new_lt_y = self.exeprice + lt
                utline.set_ydata([new_ut_y] * 2)
                ltline.set_ydata([new_lt_y] * 2)

                # Update annotation positions
                xlim = ax_left.get_xlim()
                mid_x = (xlim[0] + xlim[1]) / 2
                ut_annot.set_position((mid_x, new_ut_y))
                lt_annot.set_position((mid_x, new_lt_y))

                # Update crosshair in ax_right
                vline_cross.set_xdata([event.xdata] * 2)
                hline_cross.set_ydata([event.ydata] * 2)

                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)
