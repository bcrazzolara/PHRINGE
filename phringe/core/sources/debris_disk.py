import warnings
from typing import Any, Union

import numpy as np
import torch
from astropy import units as u
from astropy.units import Quantity
from pydantic import field_validator
from pydantic_core.core_schema import ValidationInfo
from scipy.constants import sigma
from torch import Tensor

from phringe.core.sources.base_source import BaseSource
from phringe.io.validation import validate_quantity_units
from phringe.util.grid import get_meshgrid
from phringe.util.spectrum import get_blackbody_spectrum_si_units

AU = 1.495978707e11  # m


class DebrisDisk(BaseSource):
    """Class representation of an optically thin, thermally emitting dust ring, belt or disk around the host star.

    Several components (e.g. an inner warm belt and an outer ring) can be modelled by adding several DebrisDisk
    sources to the scene.

    Radial profile of the vertical (normal) optical depth tau(r), selected with ``radial_profile``:

    - ``'gaussian'``: tau ~ exp(-(r - r0)^2 / (2 sigma^2)), a ring/belt centred on r0 with standard deviation
      ``radial_width``. This is the profile used to fit the resolved debris discs of the REASONS survey
      (Matra et al. 2025, A&A).
    - ``'two_power_law'``: tau ~ [(r/r0)^(-2 alpha_in) + (r/r0)^(-2 alpha_out)]^(-1/2), with alpha_in > 0 and
      alpha_out < 0, i.e. tau ~ r^alpha_in inside and ~ r^alpha_out outside r0 with a smooth turnover. This is the
      density profile introduced by Augereau et al. 1999 (A&A 348, 557) for HR 4796A and widely used for
      scattered-light modelling of debris discs.
    - ``'power_law'``: tau ~ (r/r0)^(-p) between ``inner_radius`` and ``outer_radius``, a classical extended disk.

    All profiles can additionally be truncated with ``inner_radius`` / ``outer_radius``, and are always cut where
    tau drops below ``truncation`` times its peak. This matters in the near/mid-IR: on the Wien side of the dust
    spectrum the emission rises exponentially with temperature, so even a tiny inner tail of hot dust (e.g. the inner
    wing of a Gaussian fitted to mm data, extended all the way to the star) can outshine the ring itself. Set
    ``inner_radius`` explicitly whenever the inner edge is known. The profile is normalised such that its maximum
    vertical optical depth equals ``optical_depth``.

    Emission: grains are treated as blackbodies (absorption efficiency 1) at the equilibrium temperature
    T(r) = temperature_factor * 278.3 K * (L / L_sun)^(1/4) * (r / 1 au)^(-1/2) (e.g. Wyatt 2008, ARA&A 46, 339),
    the same temperature law as the Exozodi source. ``temperature_factor`` > 1 mimics superheated small grains.
    The disk is optically thin, so the surface brightness of a face-on disk is tau(r) * B_lambda(T(r)).
    Scattered starlight is not included.

    Geometry: ``inclination`` and ``raan`` follow the same convention as Planet and Exozodi (inclination 0 = face-on,
    90 deg = edge-on; raan = position angle of the line of nodes measured from the +x towards the +y sky axis), so a
    disk and a planet with the same inclination and raan are coplanar.

    - ``scale_height=None``: geometrically thin disk, projected analytically (surface brightness / cos(i)). Not
      valid close to edge-on.
    - ``scale_height=h``: vertically Gaussian disk with scale height H(r) = h * r, integrated numerically along the
      line of sight. Works for all inclinations including edge-on. Aspect ratios h of a few per cent are typical for
      debris discs (see e.g. the ARKS survey, vertical structure paper, 2026).

    Parameters
    ----------
    optical_depth : float
        Peak vertical (normal) geometrical optical depth of the dust, tau_max.
    radial_profile : str
        One of 'gaussian', 'two_power_law', 'power_law'.
    reference_radius : str or float or Quantity
        Ring centre (gaussian) or reference radius r0 (power laws), in units of length (e.g. '50 au').
    radial_width : str or float or Quantity, optional
        Gaussian standard deviation sigma of the ring (gaussian profile only).
    alpha_in : float, optional
        Inner power-law index (> 0) of the two-power-law profile.
    alpha_out : float, optional
        Outer power-law index (< 0) of the two-power-law profile.
    power_law_index : float, optional
        Index p of the power-law profile, tau ~ r^-p.
    inner_radius : str or float or Quantity, optional
        Inner truncation radius (required for the power-law profile).
    outer_radius : str or float or Quantity, optional
        Outer truncation radius (required for the power-law profile).
    inclination : str or float or Quantity
        Inclination of the disk plane.
    raan : str or float or Quantity
        Position angle of the line of nodes.
    scale_height : float, optional
        Vertical aspect ratio h = H / r. None for a geometrically thin disk.
    temperature_factor : float
        Dust temperature in units of the blackbody equilibrium temperature. Default 1.
    n_line_of_sight : int, optional
        Number of line-of-sight samples for the vertically extended disk. Chosen automatically if None.
    truncation : float
        Relative optical depth (w.r.t. the peak) below which the profile is set to zero. Default 1e-4.
    """
    optical_depth: float
    radial_profile: str
    reference_radius: Union[str, float, Quantity]
    radial_width: Union[str, float, Quantity] = None
    alpha_in: float = None
    alpha_out: float = None
    power_law_index: float = None
    inner_radius: Union[str, float, Quantity] = None
    outer_radius: Union[str, float, Quantity] = None
    inclination: Union[str, float, Quantity] = 0
    raan: Union[str, float, Quantity] = 0
    scale_height: Union[float, None] = None
    temperature_factor: float = 1.0
    n_line_of_sight: Union[int, None] = None
    truncation: float = 1e-4
    _extent: Any = None
    _norm: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._check_profile_parameters()
        self._extent = self._compute_radial_extent()
        self._norm = self._compute_normalisation()

    @field_validator('radial_profile')
    def _validate_radial_profile(cls, value: Any, info: ValidationInfo) -> str:
        if value not in ('gaussian', 'two_power_law', 'power_law'):
            raise ValueError(f"radial_profile must be 'gaussian', 'two_power_law' or 'power_law', got '{value}'.")
        return value

    @field_validator('reference_radius', 'radial_width', 'inner_radius', 'outer_radius')
    def _validate_length(cls, value: Any, info: ValidationInfo) -> float:
        """Validate a length input and return it in units of au."""
        if value is None:
            return None
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.m,)) / AU

    @field_validator('inclination', 'raan')
    def _validate_angle(cls, value: Any, info: ValidationInfo) -> float:
        """Validate an angle input and return it in units of radians."""
        return validate_quantity_units(value=value, field_name=info.field_name, unit_equivalency=(u.deg,))

    def _check_profile_parameters(self):
        required = {
            'gaussian': ['radial_width'],
            'two_power_law': ['alpha_in', 'alpha_out'],
            'power_law': ['power_law_index', 'inner_radius', 'outer_radius'],
        }[self.radial_profile]
        missing = [p for p in required if getattr(self, p) is None]
        if missing:
            raise ValueError(f"radial_profile='{self.radial_profile}' requires {missing}.")
        if self.radial_profile == 'two_power_law' and not (self.alpha_in > 0 > self.alpha_out):
            raise ValueError("two_power_law requires alpha_in > 0 and alpha_out < 0.")
        if self.scale_height is None and np.cos(self.inclination) < 0.1:
            warnings.warn(
                f"DebrisDisk '{self.name}': thin-disk projection at inclination {np.degrees(self.inclination):.1f} "
                f"deg (1/cos(i) = {1 / max(np.cos(self.inclination), 1e-12):.0f}) is inaccurate; set scale_height "
                f"(e.g. 0.05) to integrate along the line of sight."
            )

    def _radial_profile_unnormalised(self, r: Tensor) -> Tensor:
        """Return the (unnormalised) vertical optical depth profile at radius r in au."""
        r0 = self.reference_radius
        if self.radial_profile == 'gaussian':
            tau = torch.exp(-(r - r0) ** 2 / (2 * self.radial_width ** 2))
        elif self.radial_profile == 'two_power_law':
            x = r / r0
            tau = (x ** (-2 * self.alpha_in) + x ** (-2 * self.alpha_out)) ** -0.5
        else:
            tau = (r / r0) ** (-self.power_law_index)

        if self.inner_radius is not None:
            tau = torch.where(r >= self.inner_radius, tau, torch.zeros_like(tau))
        if self.outer_radius is not None:
            tau = torch.where(r <= self.outer_radius, tau, torch.zeros_like(tau))
        return tau

    def _compute_radial_extent(self) -> tuple[float, float]:
        """Return (r_min, r_max) in au outside which tau < truncation * peak."""
        r = torch.logspace(-3, 4, 20000, dtype=torch.float64)
        tau = self._radial_profile_unnormalised(r)
        significant = r[tau > self.truncation * tau.max()]
        return significant.min().item(), significant.max().item()

    def _compute_normalisation(self) -> float:
        r_min, r_max = self._extent
        r = torch.linspace(r_min, r_max, 20000, dtype=torch.float64)
        return self.optical_depth / self._radial_profile_unnormalised(r).max().item()

    @property
    def _radial_extent(self) -> tuple[float, float]:
        return self._extent

    def _optical_depth(self, r: Tensor) -> Tensor:
        """Return the vertical optical depth tau(r), r in au, truncated to the significant radial extent."""
        r_min, r_max = self._extent
        tau = self._norm * self._radial_profile_unnormalised(r)
        return torch.where((r >= r_min) & (r <= r_max), tau, torch.zeros_like(tau))

    def _temperature(self, r: Tensor) -> Tensor:
        """Return the dust temperature in K at radius r in au."""
        return self.temperature_factor * 278.3 * (self._host_star_luminosity / 3.86e26) ** 0.25 * r ** -0.5

    @property
    def _host_star_luminosity(self) -> float:
        if self._phringe._scene.star is not None:
            return self._phringe._scene.star.luminosity
        obs = self._phringe._observation
        return 4 * np.pi * obs.host_star_radius ** 2 * sigma * obs.host_star_temperature ** 4

    @property
    def _host_star_distance(self) -> float:
        if self._phringe._scene.star is not None:
            return self._phringe._scene.star.distance
        return self._phringe._observation.host_star_distance

    def _surface_brightness(self) -> Tensor:
        """Return the sky surface brightness of shape n_wavelengths x n_grid x n_grid in units of ph s-1 m-3 sr-1,
        evaluated on the same sky grid as sky_coordinates."""
        x_rad, y_rad = self.sky_coordinates[:, :, 0]  # each n_wavelengths x n_grid x n_grid, in rad
        return self.get_surface_brightness(
            x_rad * self._host_star_distance / AU,
            y_rad * self._host_star_distance / AU,
            self._phringe._instrument.wavelength_bin_centers,
        )

    def get_surface_brightness(self, x: Tensor, y: Tensor, wavelengths: Tensor) -> Tensor:
        """Return the sky surface brightness at arbitrary projected sky offsets from the star, e.g. on a finer grid
        than the simulation's for plotting.

        Parameters
        ----------
        x, y : torch.Tensor
            Projected sky offsets from the star in au, of shape n_wavelengths x ... (one coordinate set per
            wavelength; use x[None] to evaluate the same coordinates for a single wavelength).
        wavelengths : torch.Tensor
            Wavelengths in m, of shape n_wavelengths.

        Returns
        -------
        torch.Tensor
            Surface brightness tau * B_lambda(T) integrated along the line of sight, same shape as x, in units of
            ph s-1 m-3 sr-1.
        """
        wavelengths = torch.as_tensor(wavelengths, dtype=torch.float32, device=x.device)
        wl_shape = (-1,) + (1,) * (x.dim() - 1)

        cos_i, sin_i = np.cos(self.inclination), np.sin(self.inclination)
        cos_o, sin_o = np.cos(self.raan), np.sin(self.raan)

        # Sky coordinates rotated so that x' lies along the line of nodes
        x_n = cos_o * x + sin_o * y
        y_n = -sin_o * x + cos_o * y

        r_floor = 1e-6  # au, avoids the T -> inf singularity at the star's position

        if self.scale_height is None:
            # Geometrically thin disk: deproject (same as Exozodi) and scale by 1/cos(i)
            r = torch.sqrt(x_n ** 2 + (y_n / cos_i) ** 2).clamp(min=r_floor)
            bb = get_blackbody_spectrum_si_units(self._temperature(r), wavelengths.reshape(wl_shape))
            return self._optical_depth(r) * bb / abs(cos_i)

        # Vertically Gaussian disk, H = h * r, integrated along the line of sight s (towards the observer).
        # Disk-frame coordinates: u along the nodes, v in the disk plane, z along the disk normal:
        #   u = x_n,  v = y_n cos(i) + s sin(i),  z = -y_n sin(i) + s cos(i)
        # In the thin limit (z = 0) this reduces to the projection used above and by Exozodi.
        r_min, r_max = self._radial_extent
        s_max = r_max * (1 + 3 * self.scale_height)
        n_s = self.n_line_of_sight or self._auto_n_line_of_sight(r_min, r_max, s_max, cos_i)
        s = torch.linspace(-s_max, s_max, n_s, device=x.device, dtype=torch.float32)
        ds = 2 * s_max / (n_s - 1)

        brightness = torch.zeros_like(x)
        for il in range(len(wavelengths)):  # loop over wavelengths to bound memory (n_pixels x n_s per step)
            u_ = x_n[il][..., None]
            v_ = y_n[il][..., None] * cos_i + s * sin_i
            z_ = -y_n[il][..., None] * sin_i + s * cos_i
            r = torch.sqrt(u_ ** 2 + v_ ** 2).clamp(min=r_floor)
            height = self.scale_height * r
            density = self._optical_depth(r) / (np.sqrt(2 * np.pi) * height) * torch.exp(-z_ ** 2 / (2 * height ** 2))
            bb = get_blackbody_spectrum_si_units(self._temperature(r), wavelengths[il])
            brightness[il] = torch.sum(density * bb, dim=-1) * ds
        return brightness

    def _auto_n_line_of_sight(self, r_min: float, r_max: float, s_max: float, cos_i: float) -> int:
        """Choose the number of line-of-sight samples so that both the vertical Gaussian (seen through the disk
        plane) and the radial structure along the line of sight are resolved."""
        h_min = self.scale_height * max(r_min, 1e-3)
        ds_vertical = h_min / max(abs(cos_i), 1e-6) / 3  # LOS crosses the layer over ~H/cos(i)
        radial_scale = self.radial_width if self.radial_profile == 'gaussian' else r_min
        ds_radial = radial_scale / 3
        n = int(np.ceil(2 * s_max / min(ds_vertical, ds_radial))) + 1
        return int(np.clip(n, 64, 4096))

    @property
    def n_grid_points(self) -> int:
        return self._phringe._grid_size ** 2

    @property
    def sky_brightness_distribution(self) -> Tensor:
        # Pixel brightness: surface brightness times full FoV solid angle; PHRINGE divides by n_grid_points.
        # Broadcast to time dimension
        return self.spectral_energy_distribution[:, None, :, :]

    @property
    def sky_coordinates(self) -> Tensor:
        sky_coordinates = get_meshgrid(
            self._phringe._instrument._field_of_view,
            self._phringe._grid_size,
            self._phringe._device,
        )

        # Broadcast to time dimension
        return sky_coordinates[:, :, None, :, :]

    @property
    def solid_angle(self) -> Union[float, Tensor]:
        return self._phringe._instrument._field_of_view ** 2

    @property
    def spectral_energy_distribution(self) -> Tensor:
        """Return the spectral energy distribution map of shape n_wavelengths x n_grid x n_grid (as for Exozodi, the
        surface brightness times the full FoV solid angle). Summing over the grid and dividing by n_grid_points gives
        the disk's total spectral flux in ph s-1 m-3."""
        return self._surface_brightness() * self.solid_angle[:, None, None]
