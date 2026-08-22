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
import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

__all__ = ["cp_profile", "AtmZkStat"]


def cp_profile(
    n_gl: int = 10,
    gl_quality: str = "typical",
    fa_quality: str = "typical",
    gl_hmin: float = 30,
    gl_hmax: float = 400,
) -> np.ndarray:
    """Generate integrated Cn2 profile for Cerro Pachon.

    Generates a discrete layer turbulence profile with a variable 
    number of ground layers. These are chosen to match the models 
    from Tokovinin 2006:
    https://academic.oup.com/mnras/article/365/4/1235/992742

    Parameters
    ----------
    n_gl: int, default=10
        The number of ground layers.
    gl_quality: str, default="typical"
        Quality of seeing in the ground layer.
        Can be "typical", "good", or "bad".
    fa_quality: str, default="typical"
        Quality of seeing in the free air above the ground layer.
        Can be "typical", "good", or "bad".
    gl_hmin: float, default=30
        The minimum height of the ground layer.
        The default 30m is about the height of the dome.
    gl_hmax: float, default=400
        The maximum height of the ground layer. The default 400m is
        set to be just under the free-air layers, which start at 500m.

    Returns
    -------
    np.ndarray
        2D array where the first row is altitudes in km, and
        the second is the integrated Cn2 value for each layer,
        in units of 1e-13 m^(1/3).
    """
    # set parameters for the ground layer model
    if gl_quality == "good":
        A = 70
        h0 = 15
        B = 0.4
        h1 = 700
    elif gl_quality == "typical":
        A = 70
        h0 = 20
        B = 1.4
        h1 = 900
    elif gl_quality == "bad":
        A = 60
        h0 = 100
        B = 2.0
        h1 = 1500

    # setup bins for integration
    gl_h_bins = np.linspace(gl_hmin, gl_hmax, n_gl + 1)
    gl_h = (gl_h_bins[:-1] + gl_h_bins[1:]) / 2

    # analytic integration
    gl_cn2_edges = h0 * A * np.exp(-gl_h_bins / h0)
    gl_cn2_edges += h1 * B * np.exp(-gl_h_bins / h1)
    gl_cn2 = 1e-3 * (gl_cn2_edges[:-1] - gl_cn2_edges[1:])

    # now the free air
    fa_h = 1e3 * np.array([0.5, 1, 2, 4, 8, 16])
    if fa_quality == "good":
        fa_cn2 = np.array([0.2, 0.03, 0.02, 0.2, 0.15, 0.25])
    elif fa_quality == "typical":
        fa_cn2 = np.array([0.4, 0.1, 0.1, 0.4, 0.2, 0.3])
    elif fa_quality == "bad":
        fa_cn2 = np.array([0.7, 0.2, 0.4, 0.6, 0.3, 0.3])

    # combine ground layers and free air into same profile
    h = np.append(gl_h, fa_h) / 1e3
    cn2 = np.append(gl_cn2, fa_cn2)

    return np.vstack((h, cn2))


class AtmZkStat:
    """Class for calculating atmospheric Zernike statistics."""

    def __init__(
        self,
        seeing: float = 0.67,
        seeing_type: str = "fwhm",
        zenith: float | None = 30,
        airmass: float | None = None,
        wavelength: float | str = "r",
        L0: float = 30,
        v_wind: float = 10,
        t_exp: float = 15,
        theta: float = 0,
        Cn2: np.ndarray = cp_profile(10),
        jmax: int = 22,
        pupil_D: float = 8.36,
        pupil_eps: float = 0.61,
        pupil_N: int = 125,
    ) -> None:
        """
        Parameters
        ----------
        seeing: float, default=0.67
            The 500 nm seeing at zenith, in arcseconds. The meaning
            of this value is determined by the seeing_type parameter.
        seeing_type: str, default="fwhm"
            If "fwhm", the seeing parameter is interpreted as the
            delivered PSF FWHM at zenith. If "dimm", the seeing 
            parameter is interpreted as the angle corresponding to
            the Fried parameter measured by a DIMM, assuming
            Kolmogorov turbulence. In other words, if the seeing
            value is measured from images, you should use "fwhm",
            and if it's measured using a DIMM, you should use "dimm".
        zenith: float or None, default=30
            Zenith angle, in degrees. You must provide only one of
            zenith or airmass.
        airmass: float, default=None
            Airmass of the observation. You must provide only one
            of zenith or airmass.
        wavelength: float or string, default="r"
            Effective wavelength of the observation, in meters.
            You can also supply a string specifying an LSST band,
            in which case the effective wavelength of that band 
            will be used.
        L0: float, default=30
            Outer scale of turbulence, in meters. If infinite or
            non-positive, Kolmogorov turbulence is assumed.
            Otherwise, von Karman turbulence.
        v_wind: float, default=10
            Wind velocity in the dominant layer, in meters per second.
        t_exp: float, default=15
            Exposure time, in seconds.
        theta: float, default=0
            Separation angle between the two sources, in degrees.
        Cn2: np.ndarray, optional
            Turbulence profile in a 2D array where the first row is 
            altitude in km, and the second is the integrated Cn^2 value 
            for each layer, in units of 1e-13 m^(1/3). The overall
            normalization doesn't matter, as the profile will be
            normalized to sum to 1. Note these values are only relevant
            when theta != 0. Default values are taken from Tokovinin 2006:
            https://academic.oup.com/mnras/article/365/4/1235/992742
        jmax: int, default=22
            The maximum Noll index for the Zernikes.
        pupil_D: float, default=8.36
            Diameter of the pupil, in meters.
        pupil_eps: float, default=0.61
            Fractional obscuration of the pupil.
        pupil_N: int, default=125
            The pupil is discretized into a grid of pupil_N x pupil_N
            points. Increasing this number increases accuracy at the
            expense of computation time. The standard deviation of
            Zernikes 4-28 has mostly converged by 125. Lower numbers
            may be desired for faster calculation, for a small
            sacrifice in accuracy.
        """
        if zenith is None and airmass is None:
            raise ValueError("You must provide either zenith or airmass")
        elif zenith is not None and airmass is not None:
            raise ValueError("Provide only one of zenith or airmass")
            
        self.seeing = seeing
        self.seeing_type = seeing_type
        self.zenith = zenith
        self.airmass = airmass
        self.wavelength = wavelength
        self.L0 = L0
        self.v_wind = v_wind
        self.t_exp = t_exp
        self.theta = theta
        self.Cn2 = Cn2
        self.jmax = jmax
        self.pupil_D = pupil_D
        self.pupil_eps = pupil_eps
        self.pupil_N = pupil_N

    def _check_zenith_airmass(self) -> None:
        """Check that either zenith or airmass is set."""
        if self._zenith is None and self._airmass is None:
            raise ValueError(
                "zenith and airmass are both None. "
                "Please set one."
            )

    @property
    def zenith(self) -> None:
        """Zenith angle in degrees"""
        self._check_zenith_airmass()        
        if self._zenith is not None:
            return self._zenith
        else:
            return np.rad2deg(np.arccos(1 / self._airmass))

    @zenith.setter
    def zenith(self, value: float | None) -> None:
        """Set the zenith angle.

        Parameters
        ----------
        value : float or None.
            Zenith angle in degrees, or None.
        """
        self._zenith = value
        if value is not None:
            self._airmass = None

    @property
    def airmass(self) -> None:
        """Airmass"""
        self._check_zenith_airmass()
        if self._airmass is not None:
            return self._airmass
        else:
            return 1 / np.cos(np.deg2rad(self._zenith))

    @airmass.setter
    def airmass(self, value: float | None) -> None:
        """Set the airmass.

        Parameters
        ----------
        value : float or None.
            Airmass, or None.
        """
        self._airmass = value
        if value is not None:
            self._zenith = None
        
    @property
    def wavelength(self) -> float:
        """Effective wavelength of the observation in meters."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float | str) -> None:
        """Set effective wavelength.

        Parameters
        ----------
        value : float or str
            Effective wavelength of the observation, in meters.
            You can also supply a string specifying an LSST band,
            in which case the effective wavelength of that band 
            will be used.
        """
        if isinstance(value, str):
            value = (
                galsim.Bandpass(
                    f"LSST_{value}.dat",
                    wave_type="nm",
                ).effective_wavelength
                * 1e-9
            )
        self._wavelength = value

    @property
    def L0(self) -> float:
        """Outer scale of turbulence, in meters"""
        return self._L0

    @L0.setter
    def L0(self, value: float) -> None:
        """Set L0.

        Parameters
        ----------
        value : float
            Outer scale of turbulence, in meters. If infinite or
            non-positive, Kolmogorov turbulence is assumed.
            Otherwise, von Karman turbulence.
        """
        value = np.inf if value < 0 else value
        self._L0 = value

    @property
    def Cn2(self) -> np.ndarray[float]:
        """Turbulence profile.
        
        Format is a 2D array where the first row is altitude in km,
        and the second is the integrated Cn^2 value for each layer,
        in units of 1e-13 m^(1/3). The profile is normalized, so the
        overall normalization doesn't matter.

        Note these values only matter when theta != 0.
        """
        return self._Cn2

    @Cn2.setter
    def Cn2(self, value: np.ndarray[float]) -> None:
        """Set Cn2.

        Parameters
        ----------
        Cn2 : np.ndarray
            Turbulence profile in a 2D array where the first row is 
            altitude in km, and the second is the integrated Cn^2 value 
            for each layer, in units of 1e-13 m^(1/3). The overall
            normalization doesn't matter, as the profile will be
            normalized to sum to 1. Note these values are only relevant
            when theta != 0.
        """
        h, cn2 = np.array(value)
        cn2 /= cn2.sum()
        self._Cn2 = np.array([h, cn2])

    @property
    def params(self) -> dict:
        """Return the parameter dictionary."""
        return {
            "seeing": self.seeing,
            "seeing_type": self.seeing_type,
            "zenith": self.zenith,
            "airmass": self.airmass,
            "wavelength": self.wavelength,
            "L0": self.L0,
            "v_wind": self.v_wind,
            "t_exp": self.t_exp,
            "theta": self.theta,
            "Cn2": self.Cn2,
            "jmax": self.jmax,
            "pupil_D": self.pupil_D,
            "pupil_eps": self.pupil_eps,
            "pupil_N": self.pupil_N,
        }

    @property
    def r0_ref(self) -> float:
        """Reference Fried parameter for 500nm at zenith."""
        # calculate r0_ref for Kolmogorov turbulence
        r0k = 0.976 * 500e-9 / np.deg2rad(self.seeing / 3600)

        # if the seeing comes from a DIMM,
        # this is the corresponding Fried parameter
        if self.seeing_type == "dimm":
            r0vk = r0k

        # if the seeing comes from the image FWHM,
        # we need to invert the formula from Tokovinin 2002
        elif self.seeing_type == "fwhm":
            result = optimize.root(
                lambda r0vk: (r0vk / r0k) ** 2
                + 2.183 * (r0vk / self.L0) ** 0.356
                - 1,
                r0k,
            )
            if not result.success:
                raise RuntimeError(
                    "Failed to invert Tokovinin formula for FWHM "
                    "as a function of Fried parameter."
                )

            r0vk = result.x[0]

        return r0vk

    @property
    def r0(self) -> float:
        """Fried parameter at the target airmass and wavelength."""
        return (
            self.r0_ref
            * (self.wavelength / 500e-9) ** 1.2
            / self.airmass**0.6
        )

    @property
    def psf_fwhm(self) -> float:
        """Target PSF FWHM in arcseconds."""
        fwhm_rad = 0.976 * self.wavelength / self.r0
        fwhm_rad *= np.sqrt(1 - 2.183 * (self.r0 / self.L0) ** 0.356)

        return 3600 * np.rad2deg(fwhm_rad)

    @property
    def t0(self) -> float:
        """Coherence time of the atmosphere, in seconds."""
        return 0.31 * self.r0 / self.v_wind

    @property
    def N(self) -> float:
        """Effective number of independent atmosphere realizations."""
        return np.clip(self.t_exp / self.t0, 1, None)

    def correlation(self, rho: np.ndarray | float) -> np.ndarray | float:
        """Calculate projected correlation function.

        Note this includes the offset from a non-zero difference in
        field angle, corresponding to self.theta.

        Note the correlation function for Kolmogorov turbulence is
        formally infinite, so for Kolmogorov turbulence, this returns
        only the finite piece of the variance, which is negative.

        Parameters
        ----------
        rho: np.ndarray or float
            The pupil distance in meters.

        Returns
        -------
        np.ndarray or float
            Values of the correlation function at distances rho.
        """
        # pull out the atmosphere structure constants
        h, cn2 = self.Cn2
        h = 1e3 * h  # km -> m

        # calculate the argument
        rho = np.atleast_1d(rho)
        r = (
            rho[..., None]
            + np.deg2rad(self.theta) * h[None, :] * self.airmass
        )

        # no turbulence
        if np.isclose(self.L0, 0):
            integrand = 0 * r

        # Kolmogorov turbulence
        elif self.L0 == np.inf:
            # calculate the integrand
            integrand = (r / self.r0) ** (5 / 3)

            # multiply in the constants
            integrand *= -6.88 / 2

        # von Karman turbulence
        else:
            # scale the distances
            r *= 2 * np.pi / self.L0

            # to avoid problems with kv(5/6, 0),
            # we will fill an array with lim_r->0 B(r)
            # and then for non-zero values of r, replace with B(r)
            integrand = np.full_like(r, gamma(5 / 6) / 2 ** (1 / 6))
            mask = np.nonzero(r)
            integrand[mask] = r[mask] ** (5 / 6) * kv(5 / 6, r[mask])

            # multiply in the constants
            integrand *= 0.0858 * (self.L0 / self.r0) ** (5 / 3)

        # evaluate integral to calculate projected correlation function
        B = np.sum(cn2 * integrand, axis=-1)

        return B.squeeze()

    def structure(self, rho: np.ndarray | float) -> np.ndarray | float:
        """Calculate projected structure function.

        Note this includes the offset from a non-zero difference in
        field angle, corresponding to self.theta.

        Parameters
        ----------
        rho: np.ndarray or float
            The pupil distance in meters.

        Returns
        -------
        np.ndarray float
            Values of the structure function at distances rho.
        """
        return 2 * (self.correlation(0) - self.correlation(rho))

    def zk_cov(self) -> np.ndarray:
        """Calculate covariance of the Zernike coefficients.

        Returns
        -------
        np.ndarray
            The covariance matrix for the Zernikes, in meters^2.
        """
        # create the pupil grid
        N = self.pupil_N
        yPupil, xPupil = np.mgrid[-1 : 1 : 1j * N, -1 : 1 : 1j * N]

        # create the Zernike basis
        zk = galsim.zernike.zernikeBasis(
            self.jmax,
            xPupil,
            yPupil,
            R_inner=self.pupil_eps,
        )[4:]

        # mask outside the pupil
        rPupil = np.sqrt(xPupil**2 + yPupil**2)
        zk *= (rPupil > self.pupil_eps) & (rPupil < 1)

        # normalize the Zernikes
        zk /= np.diag(np.einsum("jab,kab->jk", zk, zk))[:, None, None]
        self._zk = zk

        # now create a grid that is twice as large as the pupil
        x, y = np.mgrid[
            -2 : 2 : 1j * (2 * N - 1),
            -2 : 2 : 1j * (2 * N - 1),
        ]

        # calculate distance from center of grid
        rho = cdist(
            [[0, 0]],
            np.vstack((x.flatten(), y.flatten())).T,
        ).reshape(x.shape)
        rho *= self.pupil_D / 2  # scale by mirror radius

        # calculate correlation function on this grid
        Bphi = self.correlation(rho)

        # Create sliding window over the gridded correlation function.
        # Result is an (N x N x N x N) tensor, which contains the
        # correlation function between every pair of points on the pupil
        Bphi = np.lib.stride_tricks.sliding_window_view(Bphi, (N, N))

        # the numpy function doesn't actually return these windows in
        # the order we want, so we need to reverse the order of the 
        # first two dimensions
        Bphi = Bphi[::-1, ::-1, ...]  # type: ignore

        # trace over the pixels and calculate covariance
        cov = (
            np.einsum(
                "jab,kcd,abcd->jk",
                zk,
                zk,
                Bphi,
                optimize="optimal",
            )
            / self.N
        )

        # Convert wavelengths -> meters
        cov *= self.wavelength ** 2

        return cov

    def zk_std(self) -> np.ndarray:
        """Calculate the standard deviation of the Zernike coefficients.

        Returns
        -------
        np.ndarray
            Standard deviations for Zernike coefficients, in meters.
        """
        cov = self.zk_cov()
        return np.sqrt(np.diag(cov))