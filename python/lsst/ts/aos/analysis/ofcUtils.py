import numpy as np
from lsst.daf.butler import Butler
from lsst.ts.ofc import OFC, OFCData
from lsst.ts.wep.utils import makeDense

__all__ = ["AOSFetcher"]


class AOSFetcher:
    """Class to fetch AOS data from the Butler and calculate OFC corrections."""

    def __init__(
        self, butler: Butler, band: str = "r", instrument: str = "comcam"
    ) -> None:
        """Initialize the AOSFetcher object.

        Parameters
        ----------
        butler : Butler
            The Butler object to access the data. It requires a default collection
            provided, where at least the zernike estimates are stored.
        band : str, optional
            The band to use for the data, by default 'r'.
        instrument : str, optional
            The instrument to use for the data, by default 'comcam'.
        """
        self.butler = butler
        self.ofc_data = OFCData(instrument)
        self.band = band
        self.instrument = instrument

    def return_zernikes(self, visit_name: int) -> tuple:
        """Return the Zernike coefficients for the given visit.

        Parameters
        ----------
        visit_name : int
            The visit name to use for the data.

        Returns
        -------
        tuple
            A tuple containing the Zernike coefficients,
            Zernike array, Noll indices, and sensor names.
        """
        zernikes_data = self.butler.get(
            "aggregateAOSVisitTableAvg",
            dataId={"instrument": "LSSTComCam", "visit": visit_name},
        )["zk_OCS", "detector"]

        noll_indices_data = self.butler.get(
            "zernikes",
            dataId={
                "instrument": "LSSTComCam",
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
                zernikes_data[zernikes_data["detector"] == detector]["zk_OCS"][0],
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
            filter_name=self.band,
            rotation_angle=rot_angle,
        )

        return ofc.lv_dof, corrections
