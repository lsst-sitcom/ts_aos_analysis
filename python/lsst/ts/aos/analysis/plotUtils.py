import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from lsst.ts.xml.tables.m1m3 import FATable

__all__ = ["plot_zernikes", "plot_m2_forces", "plot_m1m3_forces"]


def plot_zernikes(sensor_zernikes: dict) -> None:
    """Plots Zernike coefficients for each detector in the
    given sensor_zernikes dictionary.

    Parameters
    ----------
    sensor_zernikes: dict
        Zernike dictionary returned by ofcUtils.return_zernikes.
        The keys are sensor names, and values are array of zernikes.
    """
    num_detectors = len(sensor_zernikes)
    num_cols = 3
    num_rows = (num_detectors + num_cols - 1) // num_cols

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(12, 3 * num_rows), constrained_layout=True
    )
    axs_arr = axs.flatten() if num_detectors > 1 else [axs]

    for idx, (detector, zk_values) in enumerate(sensor_zernikes.items()):
        if idx >= len(axs_arr):
            break

        axs_arr[idx].plot(
            np.arange(4, 4 + len(zk_values)), zk_values, ".", markersize=8
        )

        axs_arr[idx].set_title(f"Detector {detector}", fontsize=10)
        axs_arr[idx].grid()
        if idx % num_cols == 0:
            axs_arr[idx].set_ylabel("Wavefront Error (μm)", fontsize=10)
        if idx >= num_detectors - num_cols:
            axs_arr[idx].set_xlabel("Zernike Noll Index", fontsize=10)

    for ax in axs_arr[num_detectors:]:
        ax.set_visible(False)
    plt.show()


def plot_m2_forces(m2_forces: np.ndarray, ax: plt.Axes, fig: plt.Figure) -> plt.Axes:
    """Plot the M2 forces on the given axis.

    Parameters
    ----------
    m2_forces: np.ndarray
        Array of forces per actuator (72,1)
    ax: plt.Axes
        Axes from the figure where the forces will be plotted
    fig: plt.Figure
        Figure where forces will be plotted

    Returns
    -------
    plt.Axes:
        Axes of the subplot where the forces are plotted.
    """
    m2_location_path = Path(
        f'{os.environ["TS_CONFIG_MTTCS_DIR"]}/MTM2/v2/harrisLUT/cell_geom.yaml'
    )
    with open(m2_location_path, "r") as yaml_file:
        m2_info = yaml.safe_load(yaml_file)
    xy_actuators = np.array(m2_info["locAct_axial"])
    xact = xy_actuators[:72, 0]
    yact = xy_actuators[:72, 1]

    forces = m2_forces
    max_val = np.max(np.abs(m2_forces))

    img = ax.scatter(
        xact, yact, s=200, c=forces, cmap="seismic", vmin=-max_val, vmax=max_val
    )

    font_size = 0.5 * np.sqrt(200)
    for x, y, _id in zip(xact, yact, np.arange(0, 72)):
        ax.text(
            x,
            y,
            f"{_id}",
            color="w",
            ha="center",
            va="center",
            fontsize=font_size,
        )

    m2_outer_diameter = 1.7 * 2
    off = m2_outer_diameter * 0.48
    ax.annotate(
        "x",
        xy=(0.3 - off, 0 - off),
        xytext=(-0.3 - off, 0 - off),
        arrowprops=dict(arrowstyle="->"),
        ha="center",
        va="center",
    )
    ax.annotate(
        "y",
        xy=(0 - off, 0.3 - off),
        xytext=(0 - off, -0.3 - off),
        arrowprops=dict(arrowstyle="->"),
        ha="center",
        va="center",
    )
    fig.colorbar(img, ax=ax, cmap="seismic", label="Force (N)")
    ax.set_ylabel("Y position (m)")
    ax.set_xlabel("X position (m)")

    ax.axis("equal")
    ax.set_title("M2 forces")
    ax.set_axis_off()

    return ax


def plot_m1m3_forces(
    m1m3_forces: np.ndarray, ax: plt.Axes, fig: plt.Figure
) -> plt.Axes:
    """Plot the M1M3 forces on the given

    Parameters
    ----------
    m1m3_forces: np.ndarray
        Array of forces per actuator (156,1)
    ax: plt.Axes
        Axes from the figure where the forces will be plotted
    fig: plt.Figure
        Figure where forces will be plotted

    Returns
    -------
    plt.Axes:
        Axes of the subplot where the forces are plotted.
    """
    size = 150
    font_size = 7
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fa_table = FATable

    # Show mirror area
    m1_outer_diameter = 8.405  # meters
    m1_inner_diameter = 5.116  # meters
    m3_outer_diameter = 5.016  # meters
    m3_inner_diameter = 1.100  # meters

    m1_mirror = plt.Circle((0, 0), m1_outer_diameter / 2.0, fc="k", alpha=0.05)
    ax.add_patch(m1_mirror)

    m1_inner = plt.Circle((0, 0), m1_inner_diameter / 2.0, fc="w")
    ax.add_patch(m1_inner)

    m3_mirror = plt.Circle((0, 0), m3_outer_diameter / 2.0, fc="k", alpha=0.05)
    ax.add_patch(m3_mirror)

    m3_inner = plt.Circle((0, 0), m3_inner_diameter / 2.0, fc="w")
    ax.add_patch(m3_inner)

    # Get the position of the actuators
    ids = [fa.actuator_id for fa in fa_table]
    xact = -np.float64([fa.x_position for fa in fa_table])
    yact = -np.float64([fa.y_position for fa in fa_table])

    data = m1m3_forces
    max_val = np.max(np.abs(m1m3_forces))

    # Fill plot with empty actuators
    data[data == 0] = np.nan

    ax.scatter(xact[np.isnan(data)], yact[np.isnan(data)], c="black", s=size)

    # Plot valid data
    im = ax.scatter(
        xact, yact, c=data, s=size, cmap="seismic", vmin=-max_val, vmax=max_val
    )

    font_size = 0.5 * np.sqrt(size)
    for x, y, _id in zip(xact, yact, ids):
        ax.text(
            x,
            y,
            f"{_id}",
            color="w",
            ha="center",
            va="center",
            fontsize=font_size,
        )

    off = m1_outer_diameter * 0.45
    ax.annotate(
        "x",
        xy=(1.0 - off, 0 - off),
        xytext=(-0.5 - off, 0 - off),
        arrowprops=dict(arrowstyle="->"),
        ha="center",
        va="center",
    )
    ax.annotate(
        "y",
        xy=(0 - off, 1.0 - off),
        xytext=(0 - off, -0.5 - off),
        arrowprops=dict(arrowstyle="->"),
        ha="center",
        va="center",
    )

    ax.axis("equal")
    ax.set_title("M1M3 forces")
    ax.set_axis_off()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, cmap="seismic", cax=cax, orientation="vertical")

    cbar.set_label("Force [N]")

    cbar.ax.tick_params(axis="y", labelsize=6)

    return ax
