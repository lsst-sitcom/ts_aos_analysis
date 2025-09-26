import logging

import numpy as np
from lsst.daf.butler import Butler
from lsst.ts.aos.analysis import StateFetcher
from lsst.ts.mtaos import Model
from lsst.ts.ofc import OFC, OFCData
from lsst.ts.wep.utils import makeDense
from lsst_efd_client import EfdClient

__all__ = ["AOSFetcher", "parse_dof_str"]


def parse_dof_str(dof_str: str) -> np.ndarray:
    """Parse a string representation of integer ranges
    into a sorted list of integers.

    The input string may contain comma-separated integers
    and/or ranges of the form "start-end".
    For example:
        "0-4,10-14" -> [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
        "3,7,9-11"  -> [3, 7, 9, 10, 11]

    Parameters
    ----------
    dof_str : str
        A string containing integers and/or integer ranges,
        separated by commas.

    Returns
    -------
    numpy.ndarray
        A sorted array of integers expanded from the given ranges.

    Raises
    ------
    ValueError
        If the string cannot be parsed into integers or ranges of integers.
    """
    dof_str = dof_str.strip()
    use_dof = []
    for part in dof_str.split(","):
        if "-" in part:
            start, end = [int(p) for p in part.split("-")]
            use_dof.extend(range(start, end + 1))
        else:
            use_dof.append(int(part))
    use_dof = np.sort(use_dof)
    return use_dof


class AOSFetcher:
    """Class to fetch AOS data from the Butler
    and calculate OFC corrections.
    """

    def __init__(
        self,
        butler: Butler,
        efd_client: EfdClient,
        config_dir: str,
        instrument: str = "lsst",
    ) -> None:
        """Initialize the AOSFetcher object.

        Parameters
        ----------
        butler : Butler
            The Butler object to access the data. It requires a
            default collection provided, where at least the
            zernike estimates are stored.
        config_dir : str
            Configuration path used to initialize OFCData.
        efdClient: EfdClient
            The EfdClient that will be used to retrieve relevant data from EFD.
        instrument: str
            The instrument being used.
        """
        self.butler = butler
        self.ofc_data = OFCData(instrument, config_dir=config_dir)
        self.config_dir = config_dir
        self.efd_client = efd_client
        self.instrument = instrument
        self.state_fetcher = StateFetcher(butler, efd_client)

    async def get_mtaos_correction(
        self, visit_id: int, use_dof: np.ndarray | str, nkeep: int, log: logging.Logger
    ) -> np.ndarray:
        """Compute the MTAOS (Active Optics) correction for a given visit.

        This method builds an optical model for the LSST telescope using
        measured Zernike data and the current system state, then calculates
        corrections in the selected degrees of freedom (DOF). The results
        can be restricted by truncating the controller basis.

        Parameters
        ----------
        visit_id : int
            The exposure/visit ID for which to compute the correction.
        use_dof : numpy.ndarray or str
            Degrees of freedom to include in the correction. Can be provided
            either as:
            - a NumPy array of integer indices, or
            - a string encoding integer ranges (e.g. ``"0-4,10-14"``),
              which will be parsed by `parse_dof_str`.

            DOF indices are mapped to telescope subsystems as follows:

            - 0–4   : M2 hexapod translations/rotations
            - 5–9   : Camera hexapod translations/rotations
            - 10–29 : M1M3 bending modes
            - 30–49 : M2 bending modes


        nkeep : int
            Truncation index for the optical feedback controller (OFC),
            limiting the number of modes kept in the correction.
        log : logging.Logger
            Logger instance for model output and debugging.

        Returns
        -------
        numpy.ndarray
            The calculated correction vector in DOF space.

        Notes
        -----
        - This method is asynchronous because it queries external OCS/OFC
          data sources.
        - Internally, it uses `OFCData` and `Model` to set telescope state,
          fetch Zernike measurements from the Butler, and compute
          corrections.
        - Temporary OFC configuration is restored after correction
          calculation.
        """
        if isinstance(use_dof, str):
            use_dof = parse_dof_str(use_dof)

        ofc_data = OFCData("lsst", config_dir=self.config_dir)
        model = Model(
            instrument="lsst",
            data_path="LSSTCam",
            ofc_data=ofc_data,
            run_name="LSSTCam/runs/quickLook",
            log=log,
        )

        state = self.state_fetcher.get_aggregated_state(exp_id=visit_id)
        model.set_dof_aggr(state)

        # Initialize component DOF masks
        new_comp_dof_idx = dict(
            m2HexPos=np.full(5, False, dtype=bool),  # M2 hexapod (0–4)
            camHexPos=np.full(5, False, dtype=bool),  # Camera hexapod (5–9)
            M1M3Bend=np.full(20, False, dtype=bool),  # M1M3 bending modes (10–29)
            M2Bend=np.full(20, False, dtype=bool),  # M2 bending modes (30–49)
        )

        # Mark active DOFs
        for idof in use_dof:
            if idof < 5:
                # M2 hexapod (x, y, z, tip, tilt)
                new_comp_dof_idx["m2HexPos"][idof] = True
            elif 5 <= idof < 10:
                # Camera hexapod (x, y, z, tip, tilt)
                new_comp_dof_idx["camHexPos"][idof - 5] = True
            elif 10 <= idof < 30:
                # M1M3 bending modes (low-order figure control)
                # These modes correct deformations of the
                # primary/tertiary mirror
                new_comp_dof_idx["M1M3Bend"][idof - 10] = True
            elif 30 <= idof < 50:
                # M2 bending modes (low-order figure control)
                # These modes correct deformations of the secondary mirror
                new_comp_dof_idx["M2Bend"][idof - 30] = True

        config = {
            "filter_name": "z_20",  # overwritten by the actual filter below
            "rotation_angle": 0.0,  # overwritten by the actual rotation angle below
            "comp_dof_idx": new_comp_dof_idx,
            "truncation_index": 12,
        }
        # Update OFCData with the chosen DOFs and truncation setting
        ofc_data.comp_dof_idx = new_comp_dof_idx
        ofc_data.controller["truncation_index"] = nkeep

        # Load Zernike table (aggregated across detectors for this visit)
        # Used to extract telescope rotation angle
        zkTable = self.butler.get("aggregateZernikesAvg", visit=visit_id)
        config["rotation_angle"] = np.rad2deg(zkTable.meta["rotTelPos"])

        # Load single-detector Zernikes (e.g. detector 191)
        # Used to determine which filter was in use
        zks_info = self.butler.get("zernikes", visit=visit_id, detector=191)
        config["filter_name"] = zks_info.meta["intra"]["band"].upper()

        # Assign visit IDs (intra/extra exposures) to the model
        model.set_visit_ids(intra_id=visit_id, extra_id=None)

        # Query OCS/OFC pipeline results (async call)
        await model.query_ocps_results(model.instrument)

        # Apply the temporary OFC configuration for correction calculation
        old_config = await model.set_ofc_data_values(**config)

        # Calculate corrections in the chosen DOFs
        # - Large defocus will not trigger an exception
        # - Execution time dict is placeholder
        model.calculate_corrections(
            raise_on_large_defocus=False,
            execution_time=dict(),
            filter_name=config["filter_name"],
            rotation_angle=config["rotation_angle"],
        )

        # Restore the original OFC configuration (undo temp changes)
        await model.set_ofc_data_values(**old_config)

        # Return the final DOF correction vector
        return model.get_dof_lv()

    def get_zernikes(
        self,
        visit_name: int,
        coordinate_system: str = "CCS",
        instrument: str = "LSSTCam",
    ) -> tuple:
        """Return the Zernike coefficients for the given visit.

        Parameters
        ----------
        visit_name : int
            The visit name to use for the data.
        coordinate_system : str
            Coordinate system in which to get the Zernikes, either CCS or OCS.
        instrument : str
            Instrument to be used when querying the zernikes,
            either LSSTComCam or LSSTCam.

        Returns
        -------
        tuple
            A tuple containing the Zernike coefficients,
            Zernike array, Noll indices, and sensor names.
        """
        if coordinate_system not in ["OCS", "CCS"]:
            raise ValueError("Not an allowed coordinate system.")

        zernikes_data = self.butler.get(
            "aggregateAOSVisitTableAvg",
            dataId={"instrument": instrument, "visit": visit_name},
        )[f"zk_{coordinate_system}", "detector"]

        noll_indices_data = self.butler.get(
            "zernikes",
            dataId={
                "instrument": instrument,
                "visit": visit_name,
                "detector": zernikes_data["detector"][0],
            },
        )

        noll_indices = np.array(
            [int(col[1:]) for col in noll_indices_data.colnames if col.startswith("Z")]
        )

        zernikes = {}
        sensor_names = []
        for detector in zernikes_data["detector"]:
            zernikes[detector] = makeDense(
                zernikes_data[zernikes_data["detector"] == detector][
                    f"zk_{coordinate_system}"
                ][0],
                noll_indices,
            )
            sensor_names.append(detector)

        num_detectors = len(sensor_names)
        num_zernikes = len(next(iter(zernikes.values())))
        zernike_array = np.zeros((num_detectors, num_zernikes))
        for idx, (detector, zk_values) in enumerate(zernikes.items()):
            zernike_array[idx, :] = zk_values

        return zernikes, zernike_array, noll_indices, sensor_names

    def return_corrections_ofc_from_zernikes(
        self,
        zernike_array: np.ndarray,
        band: str,
        noll_indices: list,
        sensor_names: list,
        m2_hexapod: np.ndarray = np.ones(5, dtype=bool),
        cam_hexapod: np.ndarray = np.ones(5, dtype=bool),
        m2_bending: np.ndarray = np.ones(20, dtype=bool),
        m1m3_bending: np.ndarray = np.ones(20, dtype=bool),
        rot_angle: float = 0.0,
        trunc: float = 1e-7,
    ) -> tuple:
        """Return the OFC corrections for the given Zernike coefficients.

        Parameters
        ----------
        zernike_array : dict
            The Zernike coefficients for each detector.
        band : str
            Band of the exposure being handled.
        noll_indices : list
            The Noll indices used for the Zernike coefficients.
        sensor_names : list
            The names of the sensors used.
        m2_hexapod : np.ndarray, optional
            The M2 hexapod dofs used, by default all enabled
        cam_hexapod : np.ndarray, optional
            The camera hexapod dofs used, by default all enabled
        m2_bending : np.ndarray, optional
            The M2 bending modes used, by default all enabled
        m1m3_bending : np.ndarray, optional
            The M1M3 bending modes used, by default all enabled
        rot_angle : float, optional
            The rotation angle, by default 0.0
        trunc : float, optional
            The truncation threshold, by default 1e-7

        Returns
        -------
        tuple
            A tuple containing the LV DOF and the OFC corrections.
        """
        self.ofc_data.controller["truncation_threshold"] = trunc
        ofc = OFC(self.ofc_data)

        ofc.ofc_data.comp_dof_idx = dict(
            m2HexPos=m2_hexapod,
            camHexPos=cam_hexapod,
            M1M3Bend=m1m3_bending,
            M2Bend=m2_bending,
        )

        ofc.controller.reset_history()
        ofc.ofc_data.zn_selected = noll_indices

        name_to_id = {
            v: k for k, v in self.ofc_data.sensor_id_to_name[self.instrument].items()
        }
        sensor_ids = np.array(
            [name_to_id[sensor_name] for sensor_name in sensor_names], dtype=int
        )

        corrections = ofc.calculate_corrections(
            wfe=zernike_array,
            sensor_ids=sensor_ids,
            filter_name=band,
            rotation_angle=rot_angle,
        )

        return ofc.lv_dof, corrections
