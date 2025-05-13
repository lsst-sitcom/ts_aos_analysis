from functools import lru_cache
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import QTable
from astropy.time import Time
from lsst.daf.butler import Butler
from lsst.daf.butler.dimensions import DimensionRecord
from lsst.summit.utils.efdUtils import getEfdData, getMostRecentRowWithDataBefore
from lsst.ts.ofc import BendModeToForce, OFCData
from lsst_efd_client import EfdClient

__all__ = ["StateFetcher"]


def m2_force_to_bending_mode(force: np.ndarray) -> np.ndarray:
    """Convert M2 forces to bending modes

    Parameters
    ----------
    force : `np.ndarray`
        The M2 forces in Newtons.  The shape must be (72,)

    Returns
    -------
    bending_mode : `np.ndarray`
        The M2 bending modes in microns.  The shape is (20,)
    """
    m2_bmf = BendModeToForce("M2", OFCData("lsst"))
    return m2_bmf.bending_mode(force)


def m1m3_force_to_bending_mode(force: np.ndarray) -> np.ndarray:
    """Convert M1M3 forces to bending modes

    Parameters
    ----------
    force : `np.ndarray`
        The M1M3 forces in Newtons.  The shape must be (156,)

    Returns
    -------
    bending_mode : `np.ndarray`
        The M1M3 bending modes in microns.  The shape is (20,)
    """
    m1m3_bmf = BendModeToForce("M1M3", OFCData("lsst"))
    return m1m3_bmf.bending_mode(force)


class StateFetcher:
    def __init__(
        self, butler: Butler, efdClient: EfdClient, instrument: str = "LSSTCam"
    ) -> None:
        """StateFetcher class to retrieve relevant information on the hardware
        from EFD given a specific exposure.

        Parameters
        ----------
        butler: Butler
            The Butler object to access the data. It requires a default collection
            provided, where at least the zernike estimates are stored.
        efdClient: EfdClient
            The EfdClient that will be used to retrieve relevant data from EFD.
        instrument: str
            The instrument being used.
        """
        self.butler = butler
        self.client = efdClient
        self.instrument = instrument

    @lru_cache
    def get_exp_record(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
    ) -> DimensionRecord:
        """Get the exposure record.

        Parameters
        ----------
        day_obs : `int`, optional
            Observation day to query.
        seq_num : `int`, optional
            Sequence number to query.
        exp_id : `int`, optional
            Exposure id to query.
        record : `lsst.daf.butler.dimensions.DimensionRecord`, optional
            The exposure record to query.  (Trivial)

        Returns
        -------
        record : `lsst.daf.butler.dimensions.DimensionRecord`
            The exposure record.

        Raises
        ------
        ValueError:
            If cannot determine record from record, day_obs, seq_num, exp_id
        """
        if record is not None:
            if any(k is not None for k in [day_obs, seq_num, exp_id]):
                raise ValueError("Don't specify record if using day_obs/seq_num/exp_id")
            return record
        if day_obs is None and seq_num is None:
            if exp_id is None:
                raise ValueError("Require (day_obs, seq_num) or exp_id")
            where = f"exposure.id = {exp_id} and instrument='{self.instrument}'"
        else:
            if day_obs is None:
                raise ValueError("Require day_obs when seq_num is given")
            elif seq_num is None:
                raise ValueError("Require seq_num when day_obs is given")
            where = f"exposure.day_obs={day_obs} AND exposure.seq_num={seq_num}"
            where += f" and instrument='{self.instrument}'"
        (record,) = self.butler.registry.queryDimensionRecords("exposure", where=where)
        return record

    @lru_cache
    def _get_compensated_hexapod_telemetry(
        self, record: DimensionRecord
    ) -> pd.DataFrame:
        return getEfdData(
            self.client, "lsst.sal.MTHexapod.application", expRecord=record
        )

    @lru_cache
    def _get_uncompensated_hexapod_event(
        self, record: DimensionRecord, salIndex: int
    ) -> pd.Series:
        return getMostRecentRowWithDataBefore(
            self.client,
            "lsst.sal.MTHexapod.logevent_uncompensatedPosition",
            timeToLookBefore=record.timespan.end,
            where=lambda df: df["salIndex"] == salIndex,
        )

    def get_hexapod(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
        component: str | None = None,
        compensated: bool | None = False,
        do_mean: bool = True,
        out_type: str = "position",
    ) -> QTable | np.ndarray | dict:
        """Get hexapod state.

        Parameters
        ----------
        day_obs : `int`, optional
            Observation day to query.
        seq_num : `int`, optional
            Sequence number to query.
        exp_id : `int`, optional
            Exposure id to query.
        record : `lsst.daf.butler.dimensions.DimensionRecord`, optional
            The exposure record containing the timespan to query.
        component : {'M2' | 'Camera' | 'Cam'}
            Which hexapod to query.
        compensated : `bool`, optional
            Whether or not to return compensated values
        do_mean : `bool`, optional
            Return mean value over exposure timespan?  If true,
            then output is a dictionary with keys indicating
            hexapod degree-of-freedom and values that
            are astropy.units.Quantity
        out_type : {'position', 'error', 'demand', 'raw', 'raw_table'}
            Adjusts the output.  Allowed values are:
                position : Returns delivered hexapod positions
                error : Returns error in hexapod positions
                demand : Returns demanded hexapod positions
                raw : Returns raw pandas dataframe with all columns
                raw_table : Returns astropy.table.Table with all columns

        Returns
        -------
        val : `astropy.table.QTable` or `ndarray` or `dict`
            Either a table with columns for times and positions, or a numpy array with
            positions, or a dictionary with mean values.

        Raises
        ------
        ValueError:
            If unknown component
        ValueError:
            If unknown out_type
        """
        if out_type not in ["raw", "raw_table", "position", "error", "demand"]:
            raise ValueError(f"Unknown out_type {out_type}")

        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )

        if component in ["Cam", "Camera"]:
            component = "Cam"
            salIndex = 1
        elif component == "M2":
            salIndex = 2
        else:
            raise ValueError(f"Unknown component {component}")

        if compensated:
            df = self._get_compensated_hexapod_telemetry(record)
            df = df[df["salIndex"] == salIndex]
        else:
            df = self._get_uncompensated_hexapod_event(record, salIndex)
            df = df.to_frame().T

        if out_type == "raw":
            return df

        table = QTable.from_pandas(df)
        if out_type == "raw_table":
            return table

        new_names = [component + "_" + name for name in ["x", "y", "z", "rx", "ry"]]

        if compensated:
            original_names = [f"{out_type}{i}" for i in range(5)]
        else:
            original_names = ["x", "y", "z", "u", "v"]
        table.rename_columns(original_names, new_names)

        time = Time(
            table["private_efdStamp"].value.astype(np.float64), format="unix"
        ).tai
        time.format = "iso"
        table["time"] = time

        table = table[["time"] + new_names]

        for i, name in enumerate(new_names):
            table[name] = table[name].astype(np.float64)
            if i < 3:
                table[name].unit = u.micron
            else:
                table[name].unit = u.degree

        if not do_mean:
            return table

        out = {}
        for name in ["x", "y", "z", "rx", "ry"]:
            k = f"{component}_{name}"
            out[k] = np.mean(table[k])

        return out

    def get_hexapods(self, **kwargs: Any) -> dict:
        """Return both hexapods components in one dictionary.

        Returns
        -------
        dict:
            Dictionary with both hexapod components.
        """
        kwargs.pop("component", None)
        m2_hex = self.get_hexapod(component="M2", **kwargs)
        cam_hex = self.get_hexapod(component="Cam", **kwargs)
        if isinstance(m2_hex, dict):
            out = m2_hex
            out.update(cam_hex)
            return out
        else:
            raise ValueError("Cannot combine hexapods")

    @lru_cache
    def _get_M2_telemetry(self, record: DimensionRecord) -> pd.DataFrame:
        return getEfdData(self.client, "lsst.sal.MTM2.axialForce", expRecord=record)

    def get_M2_forces(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
        do_mean: bool = True,
        topic: str = "applied",
    ) -> QTable | np.ndarray:
        """Get M2 actuator forces

        Parameters
        ----------
        day_obs : `int`, optional
            Observation day to query.
        seq_num : `int`, optional
            Sequence number to query.
        exp_id : `int`, optional
            Exposure id to query.
        record : `lsst.daf.butler.dimensions.DimensionRecord`, optional
            The exposure record containing the timespan to query.
        do_mean : `bool`, optional
            Return mean value over exposure timespan?  If true, then output is a
            astropy.units.Quantity of size 72 for the number of actuators
        topic : `str`, optional
            The kind of force to return.  Must be one of:
            measuredAOS, applied, measured, hardpointCorrection,
            lutGravity, lutTemperature

        Returns
        -------
        val : `astropy.table.QTable` or `ndarray`
            Either a table with columns for times and forces, or a numpy array with
            forces.
        """
        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )

        telemetry = QTable.from_pandas(self._get_M2_telemetry(record))
        nrow = len(telemetry)
        if topic == "measuredAOS":
            measured = np.empty((nrow, 72), dtype=np.float64)
            hardpointCorrection = np.empty((nrow, 72), dtype=np.float64)
            lutGravity = np.empty((nrow, 72), dtype=np.float64)
            lutTemperature = np.empty((nrow, 72), dtype=np.float64)

            for i in range(72):
                measured[:, i] = telemetry[f"measured{i}"]
                hardpointCorrection[:, i] = telemetry[f"hardpointCorrection{i}"]
                lutGravity[:, i] = telemetry[f"lutGravity{i}"]
                lutTemperature[:, i] = telemetry[f"lutTemperature{i}"]
            out = measured - hardpointCorrection - lutGravity - lutTemperature
        else:
            out = np.empty((nrow, 72), dtype=np.float64)
            for i in range(72):
                out[:, i] = telemetry[f"{topic}{i}"]

        table = QTable()
        time = Time(
            telemetry["private_efdStamp"].value.astype(np.float64), format="unix"
        ).tai
        time.format = "iso"
        table["time"] = time
        table[topic] = out * u.Newton

        if not do_mean:
            return table

        return np.mean(table[topic], axis=0)

    @lru_cache
    def _get_M1M3_telemetry(self, record: DimensionRecord, topic: str) -> pd.DataFrame:
        # for topics
        #   '', 'Acceleration', 'Azimuth', 'Balance', 'Elevation', 'Thermal', 'Velocity'
        return getEfdData(
            self.client, f"lsst.sal.MTM1M3.applied{topic}Forces", expRecord=record
        )

    @lru_cache
    def _get_M1M3_event(self, record: DimensionRecord, topic: str) -> pd.Series:
        # for topics: 'ActiveOptic', 'Offset', 'Static'
        return getMostRecentRowWithDataBefore(
            self.client,
            f"lsst.sal.MTM1M3.logevent_applied{topic}Forces",
            timeToLookBefore=record.timespan.end,
        )

    def get_M1M3_forces(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
        do_mean: bool = True,
        topic: str = "ActiveOptic",
    ) -> QTable | np.ndarray:
        """Get M1M3 forces.

        Parameters
        ----------
        day_obs : `int`, optional
            Observation day to query.
        seq_num : `int`, optional
            Sequence number to query.
        exp_id : `int`, optional
            Exposure id to query.
        record : `lsst.daf.butler.dimensions.DimensionRecord`, optional
            The exposure record containing the timespan to query.
        do_mean : `bool`, optional
            Return mean value over exposure timespan?  If true, then output is a
            astropy.units.Quantity of size 156 for the number of actuators
        topic : `str`
            The kind of force to return.  Must be one of:
            '', Acceleration, Azimuth, Balance, Elevation, Thermal, Velocity,
            ActiveOptic, Offset, Static.

        Returns
        -------
        val : `astropy.table.QTable` or `ndarray`
            Either a table with columns for times and forces,
            or a numpy array with forces.

        """
        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )

        if topic in ["ActiveOptic", "Offset", "Static"]:
            event = self._get_M1M3_event(record, topic)
            out = np.empty((156,))
            for i in range(156):
                out[i] = event[f"zForces{i}"]
            return out * u.Newton

        telemetry = QTable.from_pandas(self._get_M1M3_telemetry(record, topic))
        nrow = len(telemetry)
        out = np.empty((nrow, 156), dtype=np.float64)
        for i in range(156):
            out[:, i] = telemetry[f"zForces{i}"]

        table = QTable()
        time = Time(
            telemetry["private_efdStamp"].value.astype(np.float64), format="unix"
        ).tai
        time.format = "iso"
        table["time"] = time
        table[topic] = out * u.Newton

        if not do_mean:
            return table

        return np.mean(table[topic], axis=0)

    @lru_cache
    def get_M2_bending(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
    ) -> dict:
        """Return the M2 benind modes values from the forces.

        Parameters
        ----------
        day_obs: int
            Observation day
        seq_num: int
            Sequence number
        exp_id: int
            Exposure id
        record: DimensionRecord
            Exposure record from the butler

        Returns
        -------
        dict
            Dictionary with the value of each M2 bending mode
        """
        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )
        force = self.get_M2_forces(record=record)
        quantity = m2_force_to_bending_mode(force.to_value(u.Newton)) * u.micron
        out = {}
        for i in range(20):
            out[f"M2_B{i+1}"] = quantity[i]
        return out

    @lru_cache
    def get_M1M3_bending(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
    ) -> dict:
        """Return the M1M3 benind modes values from the forces.

        Parameters
        ----------
        day_obs: int
            Observation day
        seq_num: int
            Sequence number
        exp_id: int
            Exposure id
        record: DimensionRecord
            Exposure record from the butler

        Returns
        -------
        dict
            Dictionary with the value of each M1M3 bending mode
        """
        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )
        force = self.get_M1M3_forces(record=record)
        quantity = m1m3_force_to_bending_mode(force.to_value(u.Newton)) * u.micron
        out = {}
        for i in range(20):
            out[f"M1M3_B{i+1}"] = quantity[i]
        return out

    def get_hardware_state(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
        as_array: bool = False,
        clip_small: bool = False,
    ) -> dict | np.ndarray:
        """Return hardware state of each of the degrees of freedom
        as returned by the hardware itself through EFD.

        Parameters
        ----------
        day_obs: int
            Observation day
        seq_num: int
            Sequence number
        exp_id: int
            Exposure id
        record: DimensionRecord
            Exposure record from the butler
        as_array: bool
            Boolean to return state as an array or dict
        clip_small: bool
            Boolen to delete values smaller than threshold

        Returns
        -------
        dict or np.ndarray:
            Dict or array with the state of each degree of freedom
        """
        if as_array and clip_small:
            raise ValueError("Cannot use as_array with clip_small")

        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )
        out = self.get_hexapods(record=record)
        out.update(self.get_M2_bending(record=record))
        out.update(self.get_M1M3_bending(record=record))

        if as_array:
            arr = np.empty(50)
            arr[0] = out["M2_z"].to_value(u.micron)
            arr[1] = out["M2_x"].to_value(u.micron)
            arr[2] = out["M2_y"].to_value(u.micron)
            arr[3] = out["M2_rx"].to_value(u.degree)
            arr[4] = out["M2_ry"].to_value(u.degree)
            arr[5] = out["Cam_z"].to_value(u.micron)
            arr[6] = out["Cam_x"].to_value(u.micron)
            arr[7] = out["Cam_y"].to_value(u.micron)
            arr[8] = out["Cam_rx"].to_value(u.degree)
            arr[9] = out["Cam_ry"].to_value(u.degree)
            for i in range(20):
                arr[10 + i] = out[f"M1M3_B{i+1}"].to_value(u.micron)
            for i in range(20):
                arr[30 + i] = out[f"M2_B{i+1}"].to_value(u.micron)
            return arr

        if clip_small:
            small_thresholds = {
                "M2_z": 0.1 * u.micron,
                "M2_x": 10.0 * u.micron,
                "M2_y": 10.0 * u.micron,
                "M2_rx": 1.0 * u.arcsec,
                "M2_ry": 1.0 * u.arcsec,
                "Cam_z": 0.1 * u.micron,
                "Cam_x": 10.0 * u.micron,
                "Cam_y": 10.0 * u.micron,
                "Cam_rx": 1.0 * u.arcsec,
                "Cam_ry": 1.0 * u.arcsec,
            }
            for i in range(20):
                small_thresholds[f"M1M3_B{i+1}"] = 0.001 * u.micron
                small_thresholds[f"M2_B{i+1}"] = 0.001 * u.micron

            delete_me = []
            for k, v in out.items():
                if abs(v) < small_thresholds[k]:
                    delete_me.append(k)
            for k in delete_me:
                del out[k]

        return out

    @lru_cache
    def get_aggregated_state(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
    ) -> np.ndarray:
        """Get the aggregated state of MTAOS

        Parameters
        ----------
        day_obs : `int`, optional
            Observation day to query.
        seq_num : `int`, optional
            Sequence number to query.
        exp_id : `int`, optional
            Exposure id to query.
        record : `lsst.daf.butler.dimensions.DimensionRecord`, optional
            The exposure record containing the timespan to query.

        Returns
        -------
        val : `ndarray`
            The aggregated state inside MTAOS.  The order is:
                0-4 : M2 hexapod (micron)
                5-9 : Camera hexapod (degree)
                10-29 : M1M3 bending modes (micron)
                30-49 : M2 bending modes (micron)
        """
        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )
        event = getMostRecentRowWithDataBefore(
            self.client,
            "lsst.sal.MTAOS.logevent_degreeOfFreedom",
            timeToLookBefore=record.timespan.end,
        )
        out = np.empty(
            50,
        )
        for i in range(50):
            out[i] = event[f"aggregatedDoF{i}"]
        return out

    def _get_requested_output(
        self,
        out_type: str,
        telemetry: pd.DataFrame,
        do_mean: bool,
    ) -> np.ndarray | QTable:
        if out_type == "raw":
            return telemetry

        table = QTable.from_pandas(telemetry)
        if out_type == "raw_table":
            return table

        time = Time(
            table["private_efdStamp"].value.astype(np.float64), format="unix"
        ).tai
        time.format = "iso"
        table["time"] = time

        table[out_type] = table[out_type].astype(np.float64) * u.degree

        table = table[["time", out_type]]

        if do_mean:
            return np.mean(table[out_type])

        return table

    @lru_cache
    def _get_elevation_telemetry(self, record: DimensionRecord) -> pd.DataFrame:
        return getEfdData(self.client, "lsst.sal.MTMount.elevation", expRecord=record)

    def get_elevation(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
        do_mean: bool = True,
        out_type: str = "actualPosition",
    ) -> float | QTable:
        """Get the elevation of the telescope

        Parameters
        ----------
        day_obs : `int`, optional
            Observation day to query.
        seq_num : `int`, optional
            Sequence number to query.
        exp_id : `int`, optional
            Exposure id to query.
        record : `lsst.daf.butler.dimensions.DimensionRecord`, optional
            The exposure record containing the timespan to query.
        do_mean : `bool`, optional
            Return mean value over exposure timespan?  If true, the output is a single
            scalar astropy.units.Quantity, otherwise it is a QTable.
        out_type: {'raw', 'raw_table', 'actualPosition', 'demandPosition'}
            Adjusts the output.  Allowed values are:
                actualPosition : Returns delivered elevation
                demandPosition : Returns demanded elevation
                raw : Returns raw pandas dataframe with all columns
                raw_table : Returns astropy.table.Table with all columns

        Returns
        -------
        val : `float`
            The elevation angle in degrees
        """
        if out_type not in ["raw", "raw_table", "actualPosition", "demandPosition"]:
            raise ValueError(f"Unknown out_type {out_type}")

        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )
        telemetry = self._get_elevation_telemetry(record)

        val = self._get_requested_output(out_type, telemetry, do_mean)

        return val

    @lru_cache
    def _get_rotator_telemetry(self, record: DimensionRecord) -> pd.DataFrame:
        return getEfdData(self.client, "lsst.sal.MTRotator.rotation", expRecord=record)

    def get_rotator(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
        do_mean: bool = True,
        out_type: str = "actualPosition",
    ) -> float | QTable:
        """Get the rotation angle of the camera

        Parameters
        ----------
        day_obs : `int`, optional
            Observation day to query.
        seq_num : `int`, optional
            Sequence number to query.
        exp_id : `int`, optional
            Exposure id to query.
        record : `lsst.daf.butler.dimensions.DimensionRecord`, optional
            The exposure record containing the timespan to query.
        do_mean : `bool`, optional
            Return mean value over exposure timespan?  If true, the output is a single
            scalar astropy.units.Quantity, otherwise it is a QTable.
        out_type: {'raw', 'raw_table', 'actualPosition', 'demandPosition'}
            Adjusts the output.  Allowed values are:
                actualPosition : Returns delivered rotator position
                demandPosition : Returns demanded rotator position
                raw : Returns raw pandas dataframe with all columns
                raw_table : Returns astropy.table.Table with all columns

        Returns
        -------
        val : `float`
            The rotation angle in degrees
        """
        if out_type not in ["raw", "raw_table", "actualPosition", "demandPosition"]:
            raise ValueError(f"Unknown out_type {out_type}")

        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )
        telemetry = self._get_rotator_telemetry(record)

        val = self._get_requested_output(out_type, telemetry, do_mean)

        return val

    def _get_mount_telemetry(self, record: DimensionRecord) -> pd.DataFrame:
        return getEfdData(self.client, "lsst.sal.MTPtg.mountStatus", expRecord=record)

    def get_mount(
        self,
        *,
        day_obs: int | None = None,
        seq_num: int | None = None,
        exp_id: int | None = None,
        record: DimensionRecord | None = None,
        do_mean: bool = True,
        out_type: str = "position",
    ) -> QTable | dict:
        """
        Get the mount status

        Parameters
        ----------
        day_obs : `int`, optional
            Observation day to query.
        seq_num : `int`, optional
            Sequence number to query.
        exp_id : `int`, optional
            Exposure id to query.
        record : `lsst.daf.butler.dimensions.DimensionRecord`, optional
            The exposure record containing the timespan to query.
        do_mean : `bool`, optional
            Return mean value over exposure timespan?  If true, the output is a dict
            with scalar astropy.units.Quantity values, otherwise it is a QTable.
        out_type: {'raw', 'raw_table', 'position'}
            Adjusts the output.  Allowed values are:
                position : Returns delivered mount position.
                raw : Returns raw pandas dataframe with all columns
                raw_table : Returns astropy.table.Table with all columns

        Returns
        -------
        val : `dict` or `astropy.table.QTable`
            The mount status.  If do_mean is True, then the output is a dictionary
            with keys indicating the mount degree-of-freedom and values that are
            astropy.units.Quantity.  Otherwise, the output is a table with columns
            for time, azimuth, elevation, ra, dec, and RTP.
        """

        record = self.get_exp_record(
            day_obs=day_obs, seq_num=seq_num, exp_id=exp_id, record=record
        )
        telemetry = self._get_mount_telemetry(record)

        if out_type == "raw":
            return telemetry

        table = QTable.from_pandas(telemetry)
        if out_type == "raw_table":
            return table

        time = Time(
            table["private_sndStamp"].value.astype(np.float64), format="unix"
        ).tai
        time.format = "iso"
        table["time"] = time
        old_names = ["mountAz", "mountEl", "mountRA", "mountDec", "mountRot"]
        new_names = ["azimuth", "elevation", "ra", "dec", "RTP"]
        table.rename_columns(old_names, new_names)
        table = table[["time"] + new_names]
        for name in new_names:
            if name == "ra":
                unit = u.hourangle
            else:
                unit = u.degree
            table[name] = table[name].astype(np.float64) * unit

        if do_mean:
            out = {}
            for name in new_names:
                out[name] = np.mean(table[name])
            return out

        return table
