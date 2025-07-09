import logging

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from lsst.obs.lsst import LsstCam
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
from lsst.ts.ofc import BendModeToForce, OFCData, StateEstimator
from lsst.ts.wep.utils import convertZernikesToPsfWidth
from tqdm import tqdm as tqdm

__all__ = ["AOSDatabase"]


class AOSDatabase:
    table: pd.DataFrame

    def __init__(
        self,
        day_obs: int = 20250415,
        seq_min: int = 1,
        seq_max: int = 9999,
        consdb_url: str = "http://consdb-pq.consdb:8080/consdb",
    ) -> None:
        """Create fetcher.

        Parameters
        ----------
        seq_max : int, optional
            Maximum sequence number to fetch. Default is 9999.
        consdb_url : str, optional
            URL to create ConsDB client.
            The default is "http://consdb-pq.consdb:8080/consdb".
        """
        self.log = logging.getLogger(__name__)

        self.efd_client = makeEfdClient()
        self.cdb_client = ConsDbClient(consdb_url)
        self.seeing_monitor = RingssSeeingMonitor(self.efd_client)

        self.ofc_data = OFCData("lsst")
        self.ofc_data.controller["truncation_index"] = 6

        m2_hexapod = np.ones(5, dtype=bool)
        cam_hexapod = np.ones(5, dtype=bool)
        m1m3_bending = np.zeros(20, dtype=bool)
        m2_bending = np.zeros(20, dtype=bool)
        m1m3_bending[:5] = True
        m2_bending[:5] = True
        self.ofc_data.comp_dof_idx = dict(
            m2HexPos=m2_hexapod,
            camHexPos=cam_hexapod,
            M1M3Bend=m1m3_bending,
            M2Bend=m2_bending,
        )
        self.state_estimator = StateEstimator(self.ofc_data)

        self.m2_bmf = BendModeToForce("M2", self.ofc_data)
        self.m1m3_bmf = BendModeToForce("M1M3", self.ofc_data)

        self.det_order = (191, 195, 199, 203)
        camera = LsstCam().getCamera()
        self.detector_names = [
            camera.get(det_id).getName() for det_id in self.det_order
        ]

        self.day_obs = day_obs
        self.seq_max = seq_max
        self.seq_min = seq_min
        self.table = pd.DataFrame()

        self.time_window = TimeDelta(0.2, format="sec")
        self.temp_time_window = TimeDelta(0.2, format="sec")

    async def create(self, simplified=False):
        self.table = await self._fetch(
            self.day_obs, self.seq_min, self.seq_max, simplified
        )

    async def update(self, simplified: bool = False) -> None:
        """Update the database by grabbing more recent exposures.
        This will only grab new sequences, not re-fetch existing ones.
        """
        # First grab new sequences
        seq_min = self.table["seq"].max() + 1
        updated_table = await self._fetch(
            self.day_obs, seq_min, self.seq_max, simplified=simplified
        )
        self.table = pd.concat(
            [self.table, updated_table],
            ignore_index=True,
        )

    async def _fetch(
        self, day_obs: int, seq_min: int, seq_max: int, simplified: bool = False
    ) -> pd.DataFrame:
        query = f"""
            SELECT
            e.air_temp AS air_temp,
            e.airmass AS airmass,
            e.dimm_seeing AS dimm,
            e.altitude AS elevation,
            e.azimuth AS azimuth,
            e.exposure_id AS visit_id,
            e.physical_filter as band,
            e.day_obs AS day_obs,
            e.exp_midpt AS time,
            e.dimm_seeing AS seeing,
            e.humidity AS humidity,
            e.pressure AS pressure,
            e.seq_num AS seq,
            e.sky_rotation AS sky_rotation,
            e.wind_dir AS wind_dir,
            e.wind_speed AS wind_speed,
            ccdvisit1_quicklook.psf_sigma,
            ccdvisit1_quicklook.z4,
            ccdvisit1_quicklook.z5,
            ccdvisit1_quicklook.z6,
            ccdvisit1_quicklook.z7,
            ccdvisit1_quicklook.z8,
            ccdvisit1_quicklook.z9,
            ccdvisit1_quicklook.z10,
            ccdvisit1_quicklook.z11,
            ccdvisit1_quicklook.z12,
            ccdvisit1_quicklook.z13,
            ccdvisit1_quicklook.z14,
            ccdvisit1_quicklook.z15,
            ccdvisit1_quicklook.z16,
            ccdvisit1_quicklook.z17,
            ccdvisit1_quicklook.z18,
            ccdvisit1_quicklook.z19,
            ccdvisit1_quicklook.z20,
            ccdvisit1_quicklook.z21,
            ccdvisit1_quicklook.z22,
            ccdvisit1_quicklook.z23,
            ccdvisit1_quicklook.z24,
            ccdvisit1_quicklook.z25,
            ccdvisit1_quicklook.z26,
            ccdvisit1_quicklook.z27,
            ccdvisit1_quicklook.z28,
            ccdvisit1.detector as detector,
            q.sky_noise_median AS sky_noise,
            q.sky_noise_max AS sky_noise_max,
            q.sky_noise_min AS sky_noise_min,
            q.sky_bg_median AS sky_bg,
            q.sky_bg_max AS sky_bg_max,
            q.sky_bg_min AS sky_bg_min,
            q.psf_sigma_median AS psf_fwhm,
            q.psf_sigma_min AS psf_fwhm_min,
            q.psf_sigma_max AS psf_fwhm_max,
            q.psf_ixx_median AS psf_ixx_median,
            q.psf_ixx_max AS psf_ixx_max,
            q.psf_ixx_min AS psf_ixx_min,
            q.psf_iyy_median AS psf_iyy_median,
            q.psf_iyy_max AS psf_iyy_max,
            q.psf_iyy_min AS psf_iyy_min,
            q.psf_ixy_median AS psf_ixy_median,
            q.psf_ixy_max AS psf_ixy_max,
            q.psf_ixy_min AS psf_ixy_min,
            q.psf_area_max AS psf_area_max,
            q.psf_area_median AS psf_area_median,
            q.psf_area_min AS psf_area_min,
            e.obs_end,
            e.obs_start
            FROM
            cdb_lsstcam.ccdvisit1_quicklook AS ccdvisit1_quicklook,
            cdb_lsstcam.ccdvisit1 AS ccdvisit1,
            cdb_lsstcam.visit1 AS visit1,
            cdb_lsstcam.visit1_quicklook AS q,
            cdb_lsstcam.exposure AS e
            WHERE
            ccdvisit1.detector IN (191, 192, 195, 196, 199, 200, 203, 204)
            AND ccdvisit1.ccdvisit_id = ccdvisit1_quicklook.ccdvisit_id
            AND ccdvisit1.visit_id = visit1.visit_id
            AND ccdvisit1.visit_id = q.visit_id
            AND ccdvisit1.visit_id = e.exposure_id
            AND (e.img_type = 'science' or e.img_type = 'acq')
            AND e.day_obs = {day_obs}
            AND (e.seq_num BETWEEN {seq_min} AND {seq_max})
            AND e.airmass > 0
            AND e.band != 'none'
        """
        self.table = self.cdb_client.query(query).to_pandas()

        # Convert PSF sigma to FWHM
        sig2fwhm = 2 * np.sqrt(2 * np.log(2))
        pixel_scale = 0.2  # arcsec / pixel
        self.table["psf_fwhm"] = self.table["psf_fwhm"] * sig2fwhm * pixel_scale

        self.table["fwhm_zenith_500nm"] = [
            fwhm
            * getAirmassSeeingCorrection(airmass)
            * getBandpassSeeingCorrection(band)
            for fwhm, band, airmass in zip(
                self.table["psf_fwhm"], self.table["band"], self.table["airmass"]
            )
        ]

        zernike_columns = [f"z{i}" for i in range(4, 29)]
        self.table["zernikes"] = self.table[zernike_columns].apply(
            lambda row: np.array(row.fillna(0.0).values, dtype=float), axis=1
        )
        self.table["zernikes_fwhm"] = self.table["zernikes"].apply(
            convertZernikesToPsfWidth
        )
        self.table["aos_fwhm"] = 1.06 * np.log(
            1
            + np.sqrt(
                np.sum(np.square(np.vstack(self.table["zernikes_fwhm"].values)), axis=1)
            )
        )

        if not simplified:
            unique_day_seq = (
                self.table[["day_obs", "seq", "obs_end", "obs_start"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            (
                inside_air_temp,
                days,
                fc_45,
                ringss,
                seqs,
                lut,
                inside_temp1,
                inside_temp2,
                rots,
                inside_temp3,
                camera_hex_temp,
                states,
                above_temp,
                outside_temp,
                sonic_temp,
                m2_top_temp,
                m2_bottom_temp,
                sonic_temp_dev,
            ) = ([] for _ in range(18))
            for idx, row in tqdm(unique_day_seq.iterrows(), total=len(unique_day_seq)):
                day_obs = int(row["day_obs"])
                seq = int(row["seq"])

                rec_end = row["obs_end"]
                rec_start = row["obs_start"]

                try:
                    ringss_data = self.seeing_monitor.getSeeingAtTime(
                        time=Time(rec_start, scale="utc")
                    )
                    ringss_val = ringss_data.fwhmSector
                except Exception:
                    ringss_val = np.nan

                # ---------- Environment variables -------------
                # ----------------------------------------------
                # Get state
                try:
                    total_lut_gravity = [f"lutGravity{i}" for i in range(72)]
                    total_lut_temperature = [f"lutTemperature{i}" for i in range(72)]
                    m2_actuator_data = await self.efd_client.select_time_series(
                        "lsst.sal.MTM2.axialForce",
                        ["*"],
                        Time(rec_start, scale="utc"),
                        Time(rec_start, scale="utc") + self.time_window,
                    )

                    m2_combined_lut = (
                        m2_actuator_data[total_lut_gravity].values
                        + m2_actuator_data[total_lut_temperature].values
                    )
                    m2_combined_lut = m2_combined_lut.mean(axis=0)
                    m2_dofs_lut = self.m2_bmf.bending_mode(m2_combined_lut)
                except Exception:
                    m2_dofs_lut = np.full(20, np.nan)

                try:
                    z_cols = [f"zForces{i}" for i in range(156)]
                    m1m3_el_lut = await self.efd_client.select_time_series(
                        "lsst.sal.MTM1M3.appliedElevationForces",
                        z_cols,
                        Time(rec_start, scale="utc"),
                        Time(rec_start, scale="utc") + self.time_window,
                    )
                    m1m3_el_lut = m1m3_el_lut[z_cols].values.mean(axis=0)

                    m1m3_az_lut = await self.efd_client.select_time_series(
                        "lsst.sal.MTM1M3.appliedAzimuthForces",
                        z_cols,
                        Time(rec_start, scale="utc"),
                        Time(rec_start, scale="utc") + self.time_window,
                    )
                    m1m3_az_lut = m1m3_az_lut[z_cols].values.mean(axis=0)

                    m1m3_temp_lut = await self.efd_client.select_time_series(
                        "lsst.sal.MTM1M3.appliedThermalForces",
                        z_cols,
                        Time(rec_start, scale="utc"),
                        Time(rec_start, scale="utc") + self.time_window,
                    )
                    m1m3_temp_lut = m1m3_temp_lut[z_cols].values.mean(axis=0)

                    # Align the three DataFrames
                    # (assumes same shape/timestamps)
                    m1m3_combined_lut = m1m3_el_lut + m1m3_az_lut + m1m3_temp_lut
                    m1m3_combined_lut = m1m3_combined_lut
                    m1m3_dofs_lut = self.m1m3_bmf.bending_mode(m1m3_combined_lut)
                except Exception:
                    m1m3_dofs_lut = np.full(20, np.nan)

                try:
                    cam_hexapod_data = getMostRecentRowWithDataBefore(
                        self.efd_client,
                        "lsst.sal.MTHexapod.logevent_compensationOffset",
                        Time(rec_end, scale="utc"),
                        maxSearchNMinutes=10,
                        where=lambda df: df["salIndex"] == 1,
                    )

                    m2_hexapod_data = getMostRecentRowWithDataBefore(
                        self.efd_client,
                        "lsst.sal.MTHexapod.logevent_compensationOffset",
                        Time(rec_end, scale="utc"),
                        maxSearchNMinutes=10,
                        where=lambda df: df["salIndex"] == 2,
                    )

                    hexapod_val = np.array(
                        [
                            m2_hexapod_data["z"],
                            m2_hexapod_data["x"],
                            m2_hexapod_data["y"],
                            m2_hexapod_data["u"],
                            m2_hexapod_data["v"],
                            cam_hexapod_data["z"],
                            cam_hexapod_data["x"],
                            cam_hexapod_data["y"],
                            cam_hexapod_data["u"],
                            cam_hexapod_data["v"],
                        ]
                    )
                    lut_val = np.concatenate([hexapod_val, m1m3_dofs_lut, m2_dofs_lut])
                except Exception:
                    lut_val = np.full(50, np.nan)

                event = getMostRecentRowWithDataBefore(
                    self.efd_client,
                    "lsst.sal.MTAOS.logevent_degreeOfFreedom",
                    timeToLookBefore=Time(rec_start, scale="utc"),
                )
                out = np.empty(
                    50,
                )
                for i in range(50):
                    out[i] = event[f"aggregatedDoF{i}"]
                states_val = out

                # Get outsie temperature
                temp_outside_data = await self.efd_client.select_time_series(
                    "lsst.sal.ESS.temperature",
                    ["*"],
                    Time(rec_start, scale="utc"),
                    Time(rec_end, scale="utc") + self.temp_time_window,
                    index=301,
                )
                if "temperatureItem0" in temp_outside_data:
                    outside_temp_val = temp_outside_data["temperatureItem0"].mean()
                else:
                    outside_temp_val = np.nan

                # Get sonic temperature
                temp_sonic_data = await self.efd_client.select_time_series(
                    "lsst.sal.ESS.airTurbulence",
                    ["sonicTemperature", "sonicTemperatureStdDev"],
                    Time(rec_start, scale="utc"),
                    Time(rec_end, scale="utc") + self.temp_time_window,
                )
                sonic_temp_val = temp_sonic_data["sonicTemperature"].mean()
                sonic_temp_dev_val = temp_sonic_data["sonicTemperatureStdDev"].mean()

                # Get M2 temperature
                m2_temp_data = await self.efd_client.select_time_series(
                    "lsst.sal.MTM2.temperature",
                    ["*"],
                    Time(rec_start, scale="utc"),
                    Time(rec_end, scale="utc") + self.temp_time_window,
                )
                m2_top_temp_val = m2_temp_data["ring5"].mean()
                m2_bottom_temp_val = m2_temp_data["ring6"].mean()

                # Get hex temperatures
                temp_cam_hex_data = await self.efd_client.select_time_series(
                    "lsst.sal.ESS.temperature",
                    ["*"],
                    Time(rec_start, scale="utc"),
                    Time(rec_end, scale="utc") + self.temp_time_window,
                    index=1,
                )
                temps_array = np.array(
                    [
                        temp_cam_hex_data["temperatureItem5"].mean(),
                        temp_cam_hex_data["temperatureItem4"].mean(),
                        temp_cam_hex_data["temperatureItem3"].mean(),
                        temp_cam_hex_data["temperatureItem2"].mean(),
                        temp_cam_hex_data["temperatureItem1"].mean(),
                        temp_cam_hex_data["temperatureItem0"].mean(),
                    ]
                )
                camera_hex_temp_val = np.mean(temps_array)

                # Get temperatures
                temp_data = getEfdData(
                    self.efd_client,
                    "lsst.sal.MTM1M3TS.glycolLoopTemperature",
                    begin=Time(rec_start, scale="utc"),
                    end=Time(rec_end, scale="utc"),
                )
                if "insideCellTemperature1" in temp_data:
                    inside_temp1_val = temp_data["insideCellTemperature1"].mean()
                    inside_temp2_val = temp_data["insideCellTemperature2"].mean()
                    inside_temp3_val = temp_data["insideCellTemperature3"].mean()
                    above_temp_val = temp_data["aboveMirrorTemperature"].mean()
                else:
                    inside_temp1_val = np.nan
                    inside_temp2_val = np.nan
                    inside_temp3_val = np.nan
                    above_temp_val = np.nan

                # Inside dome air temperature
                inside_air_data = await self.efd_client.select_time_series(
                    "lsst.sal.ESS.temperature",
                    ["*"],
                    Time(rec_start, scale="utc"),
                    Time(rec_end, scale="utc") + self.temp_time_window,
                    index=112,
                )
                inside_air_temp_val = inside_air_data["temperatureItem0"].mean()

                fc_data = getEfdData(
                    self.efd_client,
                    "lsst.sal.MTM1M3TS.thermalData",
                    begin=Time(rec_start, scale="utc"),
                    end=Time(rec_end, scale="utc"),
                )
                if "absoluteTemperature45" in fc_data:
                    fc_45_val = fc_data["absoluteTemperature45"].mean()
                else:
                    fc_45_val = np.nan

                rot_data = getEfdData(
                    self.efd_client,
                    "lsst.sal.MTRotator.rotation",
                    begin=Time(rec_start, scale="utc"),
                    end=Time(rec_end, scale="utc"),
                )
                rot_val = rot_data["actualPosition"].mean()

                lut.append(lut_val)
                inside_temp1.append(inside_temp1_val)
                inside_temp2.append(inside_temp2_val)
                inside_temp3.append(inside_temp3_val)
                above_temp.append(above_temp_val)
                inside_air_temp.append(inside_air_temp_val)
                fc_45.append(fc_45_val)
                camera_hex_temp.append(camera_hex_temp_val)
                m2_top_temp.append(m2_top_temp_val)
                m2_bottom_temp.append(m2_bottom_temp_val)
                sonic_temp.append(sonic_temp_val)
                sonic_temp_dev.append(sonic_temp_dev_val)
                outside_temp.append(outside_temp_val)
                states.append(states_val)
                ringss.append(ringss_val)
                days.append(day_obs)
                rots.append(rot_val)
                seqs.append(seq)

            # Calculate FWHM Zernike contributions
            efd_table = pd.DataFrame(
                {
                    "day_obs": days,
                    "seq": seqs,
                    "ringss": ringss,
                    "rotation_angle": rots,
                    "m1m3_inside_temp1": inside_temp1,
                    "m1m3_inside_temp2": inside_temp2,
                    "m1m3_inside_temp3": inside_temp3,
                    "inside_dome_air_temp": inside_air_temp,
                    "cam_hex_temp": camera_hex_temp,
                    "sonic_temp": sonic_temp,
                    "sonic_temp_dev": sonic_temp_dev,
                    "m2_top_temp": m2_top_temp,
                    "m2_bottom_temp": m2_bottom_temp,
                    "above_temp": above_temp,
                    "outside_temp": outside_temp,
                    "lut_state": lut,
                    "fc_45": fc_45,
                    "dof_state": states,
                }
            )

            self.table = pd.merge(
                self.table, efd_table, how="left", on=["seq", "day_obs"]
            )
            self.table["m1m3_delta_t"] = (
                self.table["above_temp"] - self.table["m1m3_inside_temp3"]
            )
            self.table["m2_delta_t"] = (
                self.table["m2_top_temp"] - self.table["m2_bottom_temp"]
            )
            self.table["dome_delta_t"] = (
                self.table["outside_temp"] - self.table["inside_dome_air_temp"]
            )
            self.table["cam_hex_m1m3_delta_t"] = (
                self.table["cam_hex_temp"] - self.table["above_temp"]
            )

            state_rows = []
            grouped = self.table.groupby(["seq", "day_obs"])
            for (seq, day_obs), group in tqdm(grouped):
                full_state = np.full(50, np.nan)
                try:
                    group_sorted = (
                        group.set_index("detector").loc[self.det_order].reset_index()
                    )
                    wfe = np.array(group_sorted["zernikes"].to_list())
                    filter_name = group_sorted["band"].iloc[0]
                    rotation_angle = group_sorted["rotation_angle"].iloc[0]

                    state = self.state_estimator.dof_state(
                        filter_name.split("_")[0].upper(),
                        wfe,
                        self.detector_names,
                        rotation_angle,
                    )
                    full_state[self.ofc_data.dof_idx] = state
                except Exception:
                    self.log.warning(
                        "Failed to estimate the residual state from "
                        "the wavefront error, continuing with NaN state"
                    )

                state_dict = {"seq": seq, "day_obs": day_obs}
                state_dict.update({"residual_dof_state": full_state})
                state_rows.append(state_dict)

            state_df = pd.DataFrame(state_rows)
            self.table = self.table.merge(state_df, on=["seq", "day_obs"], how="left")
        return self.table
