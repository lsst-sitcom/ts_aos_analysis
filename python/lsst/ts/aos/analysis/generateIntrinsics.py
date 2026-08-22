#!/usr/bin/env python3
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

import multiprocessing as mp
import os

import batoid
import numpy as np
from astropy import units as u
from astropy.table import Table
from batoid_rubin import LSSTBuilder
from lsst.afw.cameraGeom import FIELD_ANGLE
from lsst.obs.lsst import LsstCam
from tqdm import tqdm

# ------------- CONFIG -------------
band_mapping = {
    "u": 0.365e-6,
    "g": 0.480e-6,
    "r": 0.620e-6,
    "i": 0.754e-6,
    "z": 0.868e-6,
    "y": 0.973e-6,
}
band = "u"
telescope_type = "asbuilt"

# Grid sampling per detector (n x n points, sampled half a grid length from edges)
N_GRID = 50

# Zernike evaluation parameters
JMAX = 78
EPS = 0.612
NX = 63
PROJECTION = "gnomonic"  # DM's FIELD_ANGLE convention

camera = LsstCam().getCamera()
detectors = [det for det in camera]
# ----------------------------------

_telescope_cache = None


def init_worker(band: str, telescope_type: str) -> None:
    global _telescope_cache
    if telescope_type == "design":
        fid = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
    else:
        fid = batoid.Optic.fromYaml(f"Rubin_v3.14_{band}.yaml")

    _telescope_cache = LSSTBuilder(
        fid,
        dof_coord_system="OCS",
        flip_m2_bending_modes=False,
        dof_angle_units="degree",
    )


def process_detector(args) -> str:
    """
    Multiprocessing worker.
    args = (band, telescope_type, det_name, det_id,
    corners, completed, lock, total_detectors)
    """
    (
        band,
        telescope_type,
        det_name,
        det_id,
        corners,
        completed,
        lock,
        total_detectors,
    ) = args
    wavelength = band_mapping[band]

    global _telescope_cache
    telescope = _telescope_cache.build_det(det_id)

    # Build per-detector grid (transpose DVCS -> CCS)
    xmin = min(c.y for c in corners)
    xmax = max(c.y for c in corners)
    ymin = min(c.x for c in corners)
    ymax = max(c.x for c in corners)
    xx = np.linspace(xmin, xmax, N_GRID)
    yy = np.linspace(ymin, ymax, N_GRID)
    xx, yy = np.meshgrid(xx, yy)

    R_deg = np.sqrt(xx**2 + yy**2) * (180.0 / np.pi)
    mask = R_deg <= 1.9
    xx = xx[mask]
    yy = yy[mask]

    x_flat = xx.ravel()
    y_flat = yy.ravel()
    n_points = x_flat.size
    print(f"Starting {det_name} {det_id} {band} with {n_points} pts")

    n_coeffs = JMAX + 1
    intr = np.zeros((n_points, n_coeffs))
    for idx, (x, y) in enumerate(zip(x_flat, y_flat)):
        try:
            intr[idx, :] = (
                batoid.zernike(
                    telescope,
                    theta_x=x,
                    theta_y=y,
                    wavelength=wavelength,
                    projection=PROJECTION,
                    jmax=JMAX,
                    eps=EPS,
                    nx=NX,
                ) * wavelength
            )
        except Exception:
            intr[idx, :] = np.nan
            continue

    # Build Table: Z4..ZJMAX
    zidx = np.arange(4, n_coeffs)
    names = ["x", "y"] + [f"Z{j}" for j in zidx]
    units = [u.rad, u.rad] + [u.m] * len(zidx)
    data = np.column_stack([x_flat, y_flat, intr[:, 4:n_coeffs]])

    table = Table(data=data, names=names, units=units)

    outpath = (
        f"/sdf/data/rubin/repo/aos_imsim/gmegias/intrinsic_maps_builddet/{band}/"
        f"{det_name}_{det_id}_{band}_{telescope_type}_intrinsics_{JMAX}.fits"
    )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    table.write(outpath, overwrite=True)
    with lock:
        completed.value += 1
        print(f"Finished {det_name} {band}  " f"({completed.value}/{total_detectors})")

    return outpath


def main() -> None:
    tasks = []
    manager = mp.Manager()

    completed = manager.Value("i", 0)
    lock = manager.Lock()

    total_detectors = len(detectors)
    for det in detectors:
        name = det.getName()
        det_id = det.getId()
        corners = det.getCorners(FIELD_ANGLE)
        tasks.append(
            (
                band,
                telescope_type,
                name,
                det_id,
                corners,
                completed,
                lock,
                total_detectors,
            )
        )

    nproc = 64  # safe default on SDF

    with mp.get_context("spawn").Pool(
        processes=nproc,
        initializer=init_worker,
        initargs=(band, telescope_type),
    ) as pool:
        results = list(pool.imap_unordered(process_detector, tasks))

    print("Finished:")
    for r in results:
        print("  ", r)


if __name__ == "__main__":
    main()
