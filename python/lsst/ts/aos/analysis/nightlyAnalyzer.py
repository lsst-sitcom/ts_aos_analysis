# This file is part of ts_aos_analysis.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import galsim
import matplotlib.pyplot as plt
import itertools

import numpy as np
import pandas as pd
from lsst.daf.butler import Butler, EmptyQueryResultError
from lsst.summit.extras.ringssSeeing import RingssSeeingMonitor
from lsst.summit.utils import (
    ConsDbClient,
    getAirmassSeeingCorrection,
    getBandpassSeeingCorrection,
)

from lsst.summit.utils.efdUtils import (
    getEfdData,
    getMostRecentRowWithDataBefore,
    makeEfdClient,
)

from lsst.ts.wep.utils import convertZernikesToPsfWidth, makeDense


class NightlyAnalyzer:
    """Class to aid nightly analysis of AOS performance."""

    band_colors = {
        "u": "#0c71ff",
        "g": "#49be61",
        "r": "#c61c00",
        "i": "#ffc200",
        "z": "#f341a2",
        "y": "#5d0000",
    }

    def __init__(
        self,
        day_obs: int,
        seq_min: int = 0,
        seq_max: int = 9999,
        butler: Butler | None = None,
        consdb_url: str = "http://consdb-pq.consdb:8080/consdb",
        derotate_zernikes: bool = True,
        fetch_psf: bool = True,
        correct_aos_resid: bool = True,
    ) -> None:
        """Create the analyzer

        Parameters
        ----------
        day_obs : int
            Integer that specifies the observation day.
            E.g., April 22nd, 2025 would be 20250422
        seq_min : int, optional
            The minimum sequence number to fetch.
            The default is 0.
        seq_max : int, optional
            The maximum sequence number to fetch.
            The default is 9999.
        butler : Butler, optional
            Butler instance to query from. If None, uses
            Butler("LSSTCam", collections="LSSTCam/runs/quickLook").
            The default is None.
        consdb_url : str, optional
            URL to create ConsDB client.
            The default is "http://consdb-pq.consdb:8080/consdb".
        derotate_zernikes : bool, optional
            Whether to de-rotate the Zernikes using the physical
            rotator angle. The default is True.
        fetch_psf : bool, optional
            Whether to fetch the PSF FWHM from ConsDB. Can turn off
            if this is causing errors. The default is True.
        correct_aos_resid : bool, optional
            Whether to correct the AOS residual using the empirical
            relation corrected = 1.06 * np.log(1 + original).
            The default is True.
        """
        # Save params
        self.day_obs = day_obs
        self.seq_min = seq_min
        self.seq_max = seq_max
        self.derotate_zernikes = derotate_zernikes
        self._fetch_psf = fetch_psf
        self._correct_aos_resid = correct_aos_resid

        if butler is None:
            butler = Butler("LSSTCam", collections="LSSTCam/runs/quickLook")
        self.butler = butler

        # Create the EFD, ConsDB, Seeing clients
        self.efd_client = makeEfdClient()
        self.cdb_client = ConsDbClient(consdb_url)
        self.seeing_monitor = RingssSeeingMonitor(self.efd_client)

        # Create initial database
        self.table = self._fetch(seq_min, seq_max)

    def _query_consdb(self, seq_min: int, seq_max: int) -> pd.DataFrame:
        """Query ConsDB for the median PSF and airmass

        Parameters
        ----------
        seq_min : int
            The minimum sequence number to fetch.
        seq_max : int
            The maximum sequence number to fetch.

        Returns
        -------
        pd.DataFrame
            Table with data from ConsDB
        """
        if self._fetch_psf:
            query = f"""
                SELECT
                    e.seq_num as seq,
                    e.airmass as airmass,
                    e.physical_filter as band,
                    q.psf_sigma_median as psf_fwhm,
                    e.focus_z,
                    e.altitude,
                    e.science_program
                from
                    cdb_lsstcam.exposure as e ,
                    cdb_lsstcam.visit1_quicklook as q
                where
                    q.visit_id = e.exposure_id and
                    (e.img_type = 'OBJECT' or e.img_type = 'ACQ') and
                    e.day_obs = {self.day_obs} and
                    e.seq_num >= {seq_min} and
                    e.seq_num <= {seq_max}
                --order-by e.seq_num
            """
            cdb_table = self.cdb_client.query(query).to_pandas()
        else:
            query = f"""
                SELECT
                    e.seq_num as seq,
                    e.airmass as airmass,
                    e.physical_filter as band,
                    e.focus_z,
                    e.altitude,
                    e.science_program
                from
                    cdb_lsstcam.exposure as e
                where
                    (e.img_type = 'OBJECT' or e.img_type = 'ACQ') and
                    e.day_obs = {self.day_obs} and
                    e.seq_num >= {seq_min} and
                    e.seq_num <= {seq_max}
                --order-by e.seq_num
            """
            cdb_table = self.cdb_client.query(query).to_pandas()
            cdb_table["psf_fwhm"] = np.full(len(cdb_table), np.nan)

        # Convert PSF sigma to FWHM
        sig2fwhm = 2 * np.sqrt(2 * np.log(2))
        pixel_tilt = 0.2  # arcsec / pixel
        cdb_table["psf_fwhm"] = cdb_table["psf_fwhm"] * sig2fwhm * pixel_tilt

        # Calculate FWHM at zenith at 500nm
        cdb_table.loc[np.isclose(cdb_table["airmass"], 0), "airmass"] = np.nan
        cdb_table["fwhm_zenith_500nm"] = [
            fwhm
            * getAirmassSeeingCorrection(airmass)
            * getBandpassSeeingCorrection(band)
            for fwhm, band, airmass in zip(
                cdb_table["psf_fwhm"], cdb_table["band"], cdb_table["airmass"]
            )
        ]

        # Drop band before returning
        cdb_table = cdb_table.drop("band", axis=1)

        return cdb_table

    def _fetch(self, seq_min: int, seq_max: int) -> pd.DataFrame:
        """Fetch data from the respective databases.

        Parameters
        ----------
        seq_min : int
            The minimum sequence number to fetch.
        seq_max : int
            The maximum sequence number to fetch.

        Returns
        -------
        pd.DataFrame
            Dataframe of new data.
        """
        # Get Zernike references from the Butler
        try:
            refs = self.butler.query_datasets(
                "zernikes",
                where=(
                    f"day_obs={self.day_obs} and "
                    f"exposure.seq_num>={seq_min} and "
                    f"exposure.seq_num<={seq_max}"
                ),
            )
        except EmptyQueryResultError:
            return pd.DataFrame()

        # Loop over refs and pull data from Butler
        seqs = []
        detectors = []

        program = []
        bands = []
        ringss = []
        dimm = []
        rotations = []
        glass_temperatures = []
        above_glass_temperatures = []
        cam_dz = []

        zernikes = []
        noll_indices = np.arange(4, 29)
        zk_cols = [f"Z{j}" for j in noll_indices]
        for ref in refs:
            # Load the Zernike table
            zk_table = self.butler.get(ref)

            # Table is empty if detector had no selected donuts
            if len(zk_table) == 0:
                continue

            # Save ID metadata
            seqs.append(int(ref.dataId["visit"] - self.day_obs * 1e5))
            detectors.append(ref.dataId["detector"])
            bands.append(ref.dataId["physical_filter"])

            # Get the record for querying EFD
            rec = list(
                self.butler.registry.queryDimensionRecords(
                    "exposure",
                    dataId=ref.dataId,
                )
            )[0]

            # Query RINGSS seeing
            try:
                ringss_data = self.seeing_monitor.getSeeingForExpRecord(rec)
                ringss.append(ringss_data.fwhmSector)
            except:
                ringss.append(np.nan)

            # Query DIMM seeing
            try:
                dimm_data = getMostRecentRowWithDataBefore(
                    self.efd_client,
                    "lsst.sal.DIMM.logevent_dimmMeasurement",
                    rec.timespan.end,
                    maxSearchNMinutes=5,
                )
            except ValueError:
                dimm.append(np.nan)
            else:
                dimm.append(dimm_data["fwhm"])

            # Grab rotator value from EFD
            rot_data = getEfdData(
                self.efd_client,
                "lsst.sal.MTRotator.rotation",
                columns=["actualPosition"],
                expRecord=rec,
            )
            rotations.append(rot_data["actualPosition"].mean())

             # Grab temperatures value from EFD
            temp = getEfdData(
                self.efd_client,
                "lsst.sal.MTM1M3TS.glycolLoopTemperature",
                columns=["aboveMirrorTemperature"],
                expRecord=rec,
            )
            above_glass_temperatures.append(temp["aboveMirrorTemperature"].mean())

            temp = getEfdData(
                self.efd_client,
                "lsst.sal.MTM1M3TS.thermalData",
                columns=["absoluteTemperature45"],
                expRecord=rec,
            )
            glass_temperatures.append(temp["absoluteTemperature45"].mean())

            # Determine which Zernike coefficients are in table
            zk_table = zk_table[zk_table["label"] == "average"]
            zk_cols_here = [col for col in zk_table.colnames if col.startswith("Z")]
            noll_indices_here = [int(col.removeprefix("Z")) for col in zk_cols_here]

            # Grab Zernike values, convert to dense array, save
            zk_sparse = zk_table[zk_cols_here].to_pandas().values[0]
            zk_dense = makeDense(
                zk_sparse,
                noll_indices_here,
                noll_indices.max(),
            )
            zernikes.append(zk_dense)

        # If no detectors had selected donuts, return empty
        if len(seqs) == 0:
            return pd.DataFrame()

        # Put metadata in a dataframe
        table = pd.DataFrame(
            {
                "seq": seqs,
                "detector": detectors,
                "band": bands,
                "ringss_seeing": ringss,
                "dimm_seeing": dimm,
                "rotation": rotations,
                "above_glass_temperature": above_glass_temperatures,
                "glass_temperature": glass_temperatures,
            }
        )

        # Add ConsDB data to our table
        cdb_table = self._query_consdb(seq_min=seq_min, seq_max=seq_max)
        table = pd.merge(table, cdb_table, how="left", on="seq")

        # Convert Zernikes from nm to microns
        zernikes = np.array(zernikes) / 1e3

        # Derotate Zernikes to OCS
        if self.derotate_zernikes:
            for i, rot in enumerate(table["rotation"]):
                rot_mat = galsim.zernike.zernikeRotMatrix(
                    noll_indices.max(), -np.deg2rad(rot)
                )
                zernikes[i] = zernikes[i] @ rot_mat[4:, 4:]

        # Calculate FWHM Zernike contributions
        zernikes_fwhm = convertZernikesToPsfWidth(zernikes)
        table[zk_cols] = zernikes_fwhm
        aos_resid = np.sqrt(np.sum(np.square(zernikes_fwhm), axis=1))
        if self._correct_aos_resid:
            aos_resid = 1.06 * np.log(1 + aos_resid)
        table["aos_resid"] = aos_resid

        # Sort by sequence, then detector
        table = table.sort_values(["seq", "detector"])

        return table

    def refresh(self) -> None:
        """Requery for seeing and PSF FWHMs that are NaNs."""
        # Alias table for brevity below
        table = self.table

        # First update missing seeing
        mask = ~np.isfinite(table.ringss_seeing.values.astype(float))
        for idx in self.table[mask].index:
            # Unpack ID info
            seq, detector, band = table.loc[idx, ["seq", "detector", "band"]]

            # Get the exposure record
            if isinstance(seq, str):
                print(seq, type(seq))
            rec = list(
                self.butler.registry.queryDimensionRecords(
                    "exposure",
                    dataId={
                        "instrument": "LSSTCam",
                        "detector": detector,
                        "exposure": self.day_obs * 1e5 + int(seq),
                    },
                )
            )[0]

            # Fill-in new RINGSS data where possible
            try:
                ringss_data = self.seeing_monitor.getSeeingForExpRecord(rec)
                table.loc[idx, "ringss_seeing"] = ringss_data.fwhmSector
            except:
                pass

        # Now update missing PSF FWHMs
        mask = ~np.isfinite(table.psf_fwhm.values.astype(float))

        # Query CDB table
        cdb_table = self._query_consdb(
            seq_min=self.table[mask].seq.min(),
            seq_max=self.table[mask].seq.max(),
        )

        # Merge finite values to replace NaNs
        # this is a messy block of code!
        self.table = (
            table.set_index("seq")
            .combine_first(cdb_table.set_index("seq"))
            .reset_index()
            .set_index(table.index)[table.columns]
        )

    def update(self, verbose: bool = True) -> None:
        """Update the database by grabbing more recent exposures.

        This also calls self.refresh() to replace old NaNs where
        possible.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print how many new exposures were grabbed.
            Default is True.
        """
        # First grab new sequences
        len0 = len(self.table)
        seq_min = self.table["seq"].max() + 1
        self.table = pd.concat(
            [self.table, self._fetch(seq_min, self.seq_max)],
            ignore_index=True,
        )
        if verbose:
            print(f"Grabbed {len(self.table) - len0} more exposures")

        # Now refresh old NaNs
        self.refresh()

    @property
    def zk_cols(self) -> list:
        """List of columns with Zernikes"""
        return [col for col in self.table.columns if col[0] == "Z"]

    @property
    def noll_indices(self) -> np.ndarray:
        """Array of Noll indices in database"""
        return np.array([int(col.split("_")[0][1:]) for col in self.zk_cols])

    def downselect(
        self,
        seq_min: int = 0,
        seq_max: int | None = None,
        lookback: int | None = None,
    ) -> pd.DataFrame:
        """Down-select data in database.

        Parameters
        ----------
        seq_min : int, optional
            The minimum sequence to include.
            The default is 0.
        seq_max : int or None, optional
            The maximum sequence number to include.
            The default is None.
        lookback : int or None, optional
            Number of seqs to look back from seq_max.
            If provided, seq_min is ignored.
            The default is None.
        """
        # Get max seq
        seq_max = self.table["seq"].max() if seq_max is None else seq_max

        # Determine seq_min
        if lookback is not None:
            seq_min = self.table["seq"].max() - lookback

        # Return seqs in range
        mask = (self.table["seq"] >= seq_min) & (self.table["seq"] <= seq_max)
        return self.table[mask].copy()

    def plot_band_color_code(self) -> None:
        """Plot the band color code that appears in other plots"""
        # Create figure
        fig, ax = plt.subplots(figsize=(7, 1))

        # Block brick for each color/band
        for i, band in enumerate("ugrizy"):
            ax.scatter(
                [i],
                [1],
                marker="s",
                s=2000,
                c=self.band_colors[band],
                alpha=0.9,
            )
            ax.text(
                i,
                1,
                band,
                ha="center",
                va="center",
                fontsize=20,
                color="white",
                weight="bold",
            )

        # Set axis properties
        ax.set(xlim=(-1, 6), title="Band color code")
        ax.set_axis_off()

    def _annotate_bands(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Anotate bottom of plot with band colors

        Parameters
        ----------
        data : pd.DataFrame
            Table of data that is being plotted
        ax : plt.Axes
            Axis on which to add annotations
        """
        # Get the bands
        bands = []
        for group, group_data in data.groupby("seq"):
            bands.append(group_data["band"].iloc[0])

        # Get the axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create x grid for band bars
        x = np.arange(len(bands), dtype=float)
        dx = (x[1] - x[0]) / 2
        x -= dx
        x = np.append(x, x[-1] + 2 * dx)

        # Plot band bars at bottom of plot
        for i, band in enumerate(bands):
            ax.plot(
                x[i : i + 2],
                2 * [ylim[0]],
                c=self.band_colors[band[0]],
                lw=7,
            )

        # Restore the axis limits
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    @staticmethod
    def _detector_colors(vals: pd.DataFrame) -> list[str]:
        """List of colors for detector points

        Parameters
        ----------
        vals : pd.DataFrame
            Data that is being plotted and includes detector numbers.

        Returns
        -------
        list[str]
            List of colors for scatter plot
        """
        return [f"C{i}" for i in vals["detector"]]

    @staticmethod
    def _set_xticks(vals: pd.DataFrame, ax: plt.Axes) -> None:
        """Set the xticks so that sequence numbers are not too crowded.

        Parameters
        ----------
        vals : pd.DataFrame
            Data that is being plotted and includes sequence numbers.
        ax : plt.Axes
            Axis for which to set x ticks.
        """
        # Determine the number of sequences
        ticks = np.arange(len(vals))

        # Get their values
        labels = vals.index

        # If there are more than 10, downselect
        if len(ticks) > 10:
            n = len(ticks) // 10
            ticks = ticks[::n]
            labels = labels[::n]

        # Set ticks
        if len(ticks) > 10:
            rot = dict(rotation=30, ha="right")
        else:
            rot = dict()
        ax.set_xticks(ticks=ticks, labels=labels, **rot)

    def _plot_rotator_altitude(self, ax: plt.Axes, data: pd.DataFrame) -> plt.Axes:
        """
        Plot rotator position and elevation.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        data : pandas.DataFrame
            The data to plot

        Returns:
        --------
        ax_alt : matplotlib.axes.Axes
            The secondary y-axis for altitude
        """

        # Plot rotator position and elevation
        rot = data.groupby("seq")["rotation"].median()
        alt = data.groupby("seq")["altitude"].median()

        ax.set_ylabel('Rotation')
        ax.plot(
            rot.values,
            c="blue",
            alpha=0.5,
            lw=1,
            label="Rotator",
        )
        ax.axhline(0., c="C2", ls=":")
        ax.tick_params(axis='y')

        # Create second y-axis that shares the same x-axis
        ax_alt = ax.twinx()

        ax_alt.set_ylabel('Altitude')
        ax_alt.plot(
            alt.values,
            c="red",
            alpha=0.5,
            lw=1,
            label="Altitude",
        )
        ax_alt.tick_params(axis='y')

        # Create legend
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax_alt.get_legend_handles_labels()
        handles += handles2
        labels += labels2
        ax.legend(
            handles, labels,
            bbox_to_anchor=(0, 1.1, 1, 0.13),
            loc="upper center",
            borderaxespad=0,
            ncol=2,
        )

        return ax_alt

    def _plot_temperature(self, ax: plt.Axes, data: pd.DataFrame) -> plt.Axes:
        """
        Plot temperature data.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        data : pandas.DataFrame
            The data to plot

        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the temperature plot
        """
        # Plot temperature difference
        tempAbove = data.groupby("seq")["above_glass_temperature"].median()
        tempBelow = data.groupby("seq")["glass_temperature"].median()
        deltaT = tempAbove - tempBelow

        ax.set_ylabel('T')
        ax.plot(
            tempAbove.values,
            c="C0",
            alpha=0.5,
            lw=1,
            label="Above Glass Temp",
        )
        ax.plot(
            tempBelow.values,
            c="C0",
            alpha=0.5,
            lw=1,
            ls='-.',
            label="Glass Temp",
        )
        ax.set_ylim(
            min(tempBelow.min(), tempAbove.min()) - 1,
            max(tempBelow.max(), tempAbove.max()) + 1
        )
        ax.tick_params(axis='y')

        # Create second y-axis that shares the same x-axis
        ax_delta_t = ax.twinx()
        ax_delta_t.set_ylabel('$\\Delta$ T')
        ax_delta_t.plot(
            deltaT.values,
            c="C1",
            alpha=0.5,
            lw=1,
            ls='--',
            label="Temp Diff",
        )
        ax_delta_t.axhline(0., c="k", ls=":", label="$\\Delta$ T = 0")
        ax_delta_t.set_ylim(
            min(min(deltaT.values) - 0.5, -0.5),
            max(max(deltaT.values) + 0.5, 0.5)
        )
        ax_delta_t.tick_params(axis='y')

        # Create legend
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax_delta_t.get_legend_handles_labels()
        handles += handles2
        labels += labels2
        ax.legend(
            handles,
            labels,
            bbox_to_anchor=(0, 1.1, 1, 0.13),
            loc="upper center",
            borderaxespad=0,
            ncol=4,
        )

        return ax


    def plot_image_quality(
        self,
        seq_min: int = 0,
        seq_max: int | None = None,
        lookback: int | None = None,
        plot_scatter: bool = False,
        plot_rotator:bool = False,
        plot_temp: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot image quality vs sequence number.

        Parameters
        ----------
        seq_min : int, optional
            The minimum sequence to include.
            The default is 0.
        seq_max : int or None, optional
            The maximum sequence number to include.
            The default is None.
        lookback : int or None, optional
            Number of seqs to look back from seq_max.
            If provided, seq_min is ignored.
            The default is None.
        plot_scatter : bool, optional
            Whether to plot scatter points for each detector.
            The default is True.
        plot_rotator : bool, optional
            Whether to include the rotator/altitude plot
            The default is False.
        plot_temp : bool, optional
            Whether to include the temperature plot
            The default is False.
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        plt.Axes
            Matplotlib axis
        """
        # Downselect data
        data = self.downselect(
            seq_min=seq_min,
            seq_max=seq_max,
            lookback=lookback,
        )

        # Determine how many subplots we need
        n_plots = 1  # Main plot always included
        if plot_rotator:
            n_plots += 1
        if plot_temp:
            n_plots += 1

        # Create subplots with appropriate height ratios
        height_ratios = []
        if plot_rotator:
            height_ratios.append(1)
        if plot_temp:
            height_ratios.append(1)
        height_ratios.append(3)  # Main plot is 3x height

        fig, axes = plt.subplots(n_plots, 1, dpi=120, figsize=(8, 2+2*n_plots),
                              gridspec_kw={'height_ratios': height_ratios},
                              sharex=True)

        # If we only have one subplot, make axes iterable
        if n_plots == 1:
            axes = [axes]

        # Keep track of current axis index
        ax_idx = 0

        # Add rotator plot if requested
        if plot_rotator:
            self._plot_rotator_altitude(axes[ax_idx], data)
            ax_idx += 1

        # Add temperature plot if requested
        if plot_temp:
            self._plot_temperature(axes[ax_idx], data)
            ax_idx += 1

        # Main plot is always the last one
        ax = axes[ax_idx]

        # Plot AOS residuals
        # First scatter bar for each group
        if plot_scatter:
            for i, (group, group_data) in enumerate(data.groupby("seq")):
                ax.plot(
                    np.full_like(group_data["aos_resid"], i),
                    group_data["aos_resid"],
                    c="k",
                    lw=1,
                )
                ax.scatter(
                    np.full_like(group_data["aos_resid"], i),
                    group_data["aos_resid"],
                    c=self._detector_colors(group_data),
                    marker=".",
                    zorder=10,
                    s=60,
                )

        lines = []
        # Now line through the medians
        aos_resid = data.groupby("seq")["aos_resid"].median()
        line_aos = ax.plot(aos_resid.values, c="k", lw=1, label="AOS Resid")[0]
        lines.append(line_aos)

        # Plot DIMM seeing
        dimm_seeing = data.groupby("seq")["dimm_seeing"].median()
        line_dimm = ax.plot(
            dimm_seeing.values,
            c="rebeccapurple",
            alpha=0.5,
            lw=1,
            label="DIMM",
        )[0]
        lines.append(line_dimm)

        # Plot Gemini RINGSS seeing
        ringss_seeing = data.groupby("seq")["ringss_seeing"].median()
        line_ringss = ax.plot(ringss_seeing.values, c="silver", lw=1, label="RINGSS")[0]
        lines.append(line_ringss)

        # Plot expected FWHM
        sum_in_quad = np.sqrt(ringss_seeing**2 + aos_resid**2)
        line_sum = ax.plot(sum_in_quad.values, c="C1", label="Sum in quad.", ls="--", lw=1)[0]
        lines.append(line_sum)

        # Plot measured FWHM
        psf_fwhm = data.groupby("seq")["fwhm_zenith_500nm"].median()
        line_psf = ax.plot(psf_fwhm.values, c="C0", label="Measured", ls="--")[0]
        lines.append(line_psf)

        # Plot AOS requirement
        line_req = ax.axhline(0.25, c="C2", label="AOS Requirement", ls=":")
        lines.append(line_req)

        # Axis labels
        self._set_xticks(aos_resid, ax)
        ax.set(xlabel="Sequence number", ylabel="arcsec (zenith, 500nm)")
        self._annotate_bands(data, ax)

        # Create legend
        handles = ax.get_legend_handles_labels()
        legend = ax.legend(
            *handles,
            bbox_to_anchor=(0, 1.05, 1, 0.1),
            loc="upper center",
            borderaxespad=0,
            ncol=3,
        )

        # Enable picking on the legend
        for legline in legend.get_lines():
            legline.set_picker(10)  # Increased tolerance for easier clicking

        # Dictionary to map legend lines to original lines
        lined = {}
        for i, legline in enumerate(legend.get_lines()):
            if i < len(lines):  # Ensure we don't go out of bounds
                lined[legline] = lines[i]


        plt.tight_layout()
        plt.show()
        return fig, axes

    def plot_history_resid(
        self,
        seq_min: int = 0,
        seq_max: int | None = None,
        lookback: int | None = None,
        plot_scatter: bool = True,
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plot history of the AOS residual.

        Also includes Zernike residuals for final sequence.

        Parameters
        ----------
        seq_min : int, optional
            The minimum sequence to include.
            The default is 0.
        seq_max : int or None, optional
            The maximum sequence number to include.
            The default is None.
        lookback : int or None, optional
            Number of seqs to look back from seq_max.
            If provided, seq_min is ignored.
            The default is None.
        plot_scatter : bool, optional
            Whether to plot scatter points for each detector.
            The default is True.

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        np.ndarray[plt.Axes]
            Array of Matplotlib axis objects
        """
        # Downselect data
        data = self.downselect(
            seq_min=seq_min,
            seq_max=seq_max,
            lookback=lookback,
        )

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot AOS Resid vs Time
        # ----------------------
        # First scatter bar for each group
        if plot_scatter:
            for i, (group, group_data) in enumerate(data.groupby("seq")):
                axes[0].plot(
                    np.full_like(group_data["aos_resid"], i),
                    group_data["aos_resid"],
                    c="k",
                    lw=1,
                )
                axes[0].scatter(
                    np.full_like(group_data["aos_resid"], i),
                    group_data["aos_resid"],
                    c=self._detector_colors(group_data),
                    marker=".",
                    zorder=10,
                    s=60,
                )

        # Now line through the medians
        aos_resid = data.groupby("seq")["aos_resid"].median()
        axes[0].plot(aos_resid.values, c="k", lw=1)

        # Axis labels
        self._set_xticks(aos_resid, axes[0])
        axes[0].set(
            xlabel="Sequence number",
            ylabel='AOS FWHM Resid (")',
            title="History of residual AOS FWHM",
        )
        self._annotate_bands(data, axes[0])

        # Residuals for last group
        # ------------------------
        data = data[data["seq"] == data["seq"].max()]

        # Scatter bar for each group
        if plot_scatter:
            for j, col in zip(self.noll_indices, self.zk_cols):
                axes[1].plot(
                    np.full_like(data[col], j),
                    np.abs(data[col]),
                    c="k",
                    lw=1,
                )
                axes[1].scatter(
                    np.full_like(data[col], j),
                    np.abs(data[col]),
                    c=self._detector_colors(data),
                    marker=".",
                    zorder=10,
                    s=60,
                )

        # Now line through the medians
        axes[1].plot(
            self.noll_indices,
            np.abs(data[self.zk_cols]).median(),
            c="k",
            lw=1,
        )

        # Axis labels
        axes[1].set(
            ylabel='AOS FWHM Resid (")',
            xlabel="Noll indices",
            xticks=self.noll_indices,
            title="Residual AOS FWHM, last sequence",
        )

        # Adjust space between axes
        fig.subplots_adjust(hspace=0.4)

        return fig, axes

    def plot_history_zk(
        self,
        seq_min: int = 0,
        seq_max: int | None = None,
        lookback: int | None = None,
        plot_scatter: bool = True,
        skip_odd: bool = False,
        skip_even: bool = False,
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
        """Plot history of the AOS residual.

        Also includes Zernike residuals for final sequence.

        Parameters
        ----------
        seq_min : int, optional
            The minimum sequence to include.
            The default is 0.
        seq_max : int or None, optional
            The maximum sequence number to include.
            The default is None.
        lookback : int or None, optional
            Number of seqs to look back from seq_max.
            If provided, seq_min is ignored.
            The default is None.
        plot_scatter : bool, optional
            Whether to plot scatter points for each detector.
            The default is True.
        skip_odd : bool, optional
            Whether to skip odd-numbered Zernikes when plotting.
            This can clean up the plots if even-odd parity pairs
            are too cluttered. This does not affect z11, which is
            always plotted. The default is False.
        skip_even : bool, optional
            Whether to skip even-numbered Zernikes when plotting.
            This can clean up the plots if even-odd parity pairs
            are too cluttered. This does not affect z4 or z22,
            which are always plotted. The default is False.

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        np.ndarray[plt.Axes]
            Array of Matplotlib axis objects
        """
        # Downselect data
        data = self.downselect(
            seq_min=seq_min,
            seq_max=seq_max,
            lookback=lookback,
        )

        # Create the grid
        fig, axes = plt.subplots(
            7,
            3,
            figsize=(14, 18),
            constrained_layout=True,
        )
        axes[0, 0].set_title("Defocus (4)")
        axes[0, 1].set_title("Spherical (11)")
        axes[0, 2].set_title("2nd Spherical (22)")
        axes[1, 1].set_title("Coma (7, 8)")
        axes[1, 2].set_title("2nd Coma (17, 18)")
        axes[2, 0].set_title("Astigmatism (5, 6)")
        axes[2, 1].set_title("2nd Astigmatism (12, 13)")
        axes[3, 0].set_title("Trefoil (9, 10)")
        axes[3, 1].set_title("2nd Trefoil (18, 19)")
        axes[4, 0].set_title("Quadrafoil (14, 15)")
        axes[5, 0].set_title("Pentafoil (20, 21)")
        axes[6, 0].set_title("Hexafoil (27, 28)")

        ax_dict = {
            4: axes[0, 0],
            5: axes[2, 0],
            6: axes[2, 0],
            7: axes[1, 1],
            8: axes[1, 1],
            9: axes[3, 0],
            10: axes[3, 0],
            11: axes[0, 1],
            12: axes[2, 1],
            13: axes[2, 1],
            14: axes[4, 0],
            15: axes[4, 0],
            16: axes[1, 2],
            17: axes[1, 2],
            18: axes[3, 1],
            19: axes[3, 1],
            20: axes[5, 0],
            21: axes[5, 0],
            22: axes[0, 2],
            27: axes[6, 0],
            28: axes[6, 0],
        }

        # Loop Noll indices
        for j in self.noll_indices:
            # Not every Noll index will be plotted
            if j not in ax_dict:
                continue

            # Do we want to skip this index?
            # (sometimes we do to make plots cleaner)
            if skip_odd and j % 2 == 1 and j != 11:
                continue
            if skip_even and j % 2 == 0 and j not in [4, 22]:
                continue

            # Get axis for this index
            ax = ax_dict[j]

            # Odd Zernikes will get open-face plots
            if j % 2 == 0 or j == 11:
                ls = "-"
                facecolor = lambda data: self._detector_colors(group_data)
            else:
                ls = "--"
                facecolor = lambda data: "none"

            # First scatter bar for each group
            if plot_scatter:
                for i, (group, group_data) in enumerate(data.groupby("seq")):
                    ax.plot(
                        np.full_like(group_data[f"Z{j}"], i),
                        group_data[f"Z{j}"].values,
                        c="k",
                        lw=1,
                        ls=ls,
                    )
                    ax.scatter(
                        np.full_like(group_data[f"Z{j}"], i),
                        group_data[f"Z{j}"].values,
                        facecolor=facecolor(group_data),
                        edgecolor=self._detector_colors(group_data),
                        marker=".",
                        zorder=10,
                        s=60,
                    )

            # Now line through the medians
            medians = data.groupby("seq")[f"Z{j}"].median()
            ax.plot(medians.values, c="k", lw=1, ls=ls)

            # Mark zero
            ax.axhline(0, c="silver", lw=1, zorder=-1)

            # Axis labels
            self._set_xticks(medians, ax)
            ax.set(
                xlabel="Sequence number",
                ylabel='AOS FWHM Resid (")',
            )

        for ax in axes.flatten():
            if ax in ax_dict.values():
                self._annotate_bands(data, ax)
            else:
                ax.set_axis_off()

        return fig, axes

    def plot_zks_quality(
        self,
        seq_min: int = 0,
        seq_max: int | None = None,
        lookback: int | None = None,
        plot_scatter: bool = False,
        plot_rotator:bool = False,
        plot_temp: bool = False,
        jmin: int = 4,
        jmax: int = 28,
        ) -> tuple[plt.Figure, plt.Axes]:

        """
        Plot Zernike coefficients quality as a function of sequence.

        Parameters
        ----------
        seq_min : int, optional
            The minimum sequence to include.
            The default is 0.
        seq_max : int or None, optional
            The maximum sequence number to include.
            The default is None.
        lookback : int or None, optional
            Number of seqs to look back from seq_max.
            If provided, seq_min is ignored.
            The default is None.
        plot_scatter : bool, optional
            Whether to plot scatter points for each detector.
            The default is True.
        plot_rotator : bool, optional
            Whether to include the rotator/altitude plot
            The default is False.
        plot_temp : bool, optional
            Whether to include the temperature plot
            The default is False.
        jmin : int, optional
            The minimum Noll index to include in the plot.
            The default is 4.
        jmax : int, optional
            The maximum Noll index to include in the plot.
            The default is 28.

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        plt.Axes
            Matplotlib axis
        """

        # Downselect data
        data = self.downselect(
            seq_min=seq_min,
            seq_max=seq_max,
            lookback=lookback,
        )

        # Determine how many subplots we need
        n_plots = 1  # Main plot always included
        if plot_rotator:
            n_plots += 1
        if plot_temp:
            n_plots += 1

        # Create subplots with appropriate height ratios
        height_ratios = []
        if plot_rotator:
            height_ratios.append(1)
        if plot_temp:
            height_ratios.append(1)
        height_ratios.append(3)  # Main plot is 3x height

        fig, axes = plt.subplots(n_plots, 1, dpi=120, figsize=(10, 2+2*n_plots),
                              gridspec_kw={'height_ratios': height_ratios},
                              sharex=True)

        # If we only have one subplot, make axes iterable
        if n_plots == 1:
            axes = [axes]

        # Keep track of current axis index
        ax_idx = 0

        # Add rotator plot if requested
        if plot_rotator:
            self._plot_rotator_altitude(axes[ax_idx], data)
            ax_idx += 1

        # Add temperature plot if requested
        if plot_temp:
            self._plot_temperature(axes[ax_idx], data)
            ax_idx += 1

        # Main plot is always the last one
        ax = axes[ax_idx]

        # Zernike coefficient data
        table = {
            "Defocus (4)": [4, None],
            "Spherical (11)": [11, None],
            "2nd Spherical (22)": [22, None],
            "Coma (7, 8)": [7, 8],
            "2nd Coma (17, 18)": [17, 18],
            "Astigmatism (5, 6)": [5, 6],
            "2nd Astigmatism (12, 13)": [12, 13],
            "Trefoil (9, 10)": [9, 10],
            "2nd Trefoil (18, 19)": [18, 19],
            "Quadrafoil (14, 15)": [14, 15],
            "Pentafoil (20, 21)": [20, 21],
            "Hexafoil (27, 28)": [27, 28],
        }

        colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#17becf",  # cyan
            "#aec7e8",  # light blue
            "#ffbb78",  # light orange
        ]
        color_cycle = itertools.cycle(colors)

        # Plot AOS residuals
        # First scatter bar for each group

        legend_lines = []
        legend_colors = []

        for key, zks in table.items():
            # Skip Zernikes outside the specified range
            if zks[0] > jmax:
                continue
            elif zks[1] is not None and zks[1] < jmin:
                continue
            color = next(color_cycle)
            legend_colors.append(color)
            lines = []

            for zk in zks:
                # Again, skip Zernikes outside the specified range
                if zk is None or (zk < jmin or zk > jmax):
                    continue
                else:
                    # Determine line style based on Zernike index
                    ls = "-" if zk % 2 == 0 or zk == 11 else "--"

                    # Plot the Zernike coefficient medians
                    medians = data.groupby("seq")[f"Z{zk}"].median()
                    line = ax.plot(medians.values, c=color, lw=1, ls=ls, label=key)[0]
                    lines.append(line)

                    if plot_scatter:
                        #TODO (05/06/2025) need to sort out setting line color when I have scatter points
                        for i, (group, group_data) in enumerate(data.groupby("seq")):
                            line = ax.plot(
                                np.full_like(group_data[f"Z{zk}"], i),
                                group_data[f"Z{zk}"].values,
                                c="k",
                                lw=1,
                                ls=ls,
                            )
                            scatter = ax.scatter(
                                np.full_like(group_data[f"Z{zk}"], i),
                                group_data[f"Z{zk}"].values,
                                facecolor=color,
                                edgecolor=self._detector_colors(group_data),
                                marker=".",
                                zorder=10,
                                s=60,
                            )
                            lines.append(line[0])
                            lines.append(scatter)
            legend_lines.append(lines)

        # Mark zero
        ax.axhline(0, c="silver", lw=1, zorder=-1)

        # Axis labels
        self._set_xticks(medians, ax)
        ax.set(
            xlabel="Sequence number",
            ylabel='AOS FWHM Resid (")',
        )

        # Create legend
        handles = ax.get_legend_handles_labels()
        legend = ax.legend(
            list(dict.fromkeys(handles[1])),
            bbox_to_anchor=(1.03, 0.5),
            loc="center left",
            borderaxespad=0,
            ncol=2,
        )

        # Set legend line colors
        for i, leg_lines in enumerate(legend.get_lines()):
            leg_lines.set_color(legend_colors[i])

        # Dictionary to map legend lines to original lines
        lined = {}
        for i, legline in enumerate(legend.get_lines()):
            lined[legline] = legend_lines[i]

        plt.tight_layout()
        plt.show()

        return fig, axes
