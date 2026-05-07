"""Core image-processing primitives for AMyGDA."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[Any]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
PointArray = NDArray[np.int_]


@dataclass(frozen=True)
class PlateLayout:
    """Configuration describing the drug layout for a plate design.

    Attributes
    ----------
    drug_names : numpy.ndarray
        Drug identifier for each well.
    concentrations : numpy.ndarray
        Drug concentration for each well.
    dilutions : numpy.ndarray
        Dilution index for each well.
    """

    drug_names: NDArray[np.str_]
    concentrations: FloatArray
    dilutions: IntArray


@dataclass(frozen=True)
class MicResult:
    """Inferred MIC state for a single drug strip.

    Attributes
    ----------
    mic_concentration : float
        Inferred MIC concentration.
    mic_dilution : int
        Inferred MIC dilution index.
    inconsistent_growth : bool, default=False
        Whether the growth pattern was internally inconsistent.
    """

    mic_concentration: float
    mic_dilution: int
    inconsistent_growth: bool = False


def infer_mic(
    growth: FloatArray,
    concentrations: FloatArray,
    dilutions: IntArray,
    growth_threshold_percentage: float,
    controls_valid: bool,
) -> MicResult:
    """Infer the MIC for one drug from ordered well measurements.

    Parameters
    ----------
    growth : numpy.ndarray
        Measured growth percentages ordered from low to high concentration.
    concentrations : numpy.ndarray
        Drug concentrations ordered from low to high concentration.
    dilutions : numpy.ndarray
        Dilution numbers aligned with ``growth`` and ``concentrations``.
    growth_threshold_percentage : float
        Threshold above which a well is considered to contain growth.
    controls_valid : bool
        Whether the positive control wells showed acceptable growth.

    Returns
    -------
    MicResult
        The inferred MIC state for the drug.

    Notes
    -----
    Parameters are expected in low-to-high concentration order after dilution
    reordering has been applied by the caller.
    """

    if not controls_valid:
        return MicResult(mic_concentration=-2.0, mic_dilution=-2)

    mic_conc: float | None = None
    mic_dilution: int | None = None
    seen_growth = False
    seen_no_growth = False
    inconsistent_growth = False

    for growth_value, concentration, dilution_value in zip(
        growth, concentrations, dilutions, strict=False
    ):
        if growth_value > growth_threshold_percentage:
            seen_growth = True
            if seen_no_growth:
                inconsistent_growth = True
        else:
            if mic_conc is None and seen_growth and not seen_no_growth:
                mic_conc = float(concentration)
                mic_dilution = int(dilution_value)
            seen_no_growth = True

    if seen_growth and not seen_no_growth and mic_conc is None:
        mic_conc = float(np.max(concentrations) * 2)
        mic_dilution = int(np.max(dilutions) + 1)

    if seen_no_growth and not seen_growth and mic_conc is None:
        mic_conc = float(np.min(concentrations))
        mic_dilution = 1

    if inconsistent_growth:
        return MicResult(
            mic_concentration=-1.0,
            mic_dilution=-1,
            inconsistent_growth=True,
        )

    if mic_conc is None or mic_dilution is None:
        raise RuntimeError("Could not infer an MIC from the provided growth values.")

    return MicResult(mic_concentration=mic_conc, mic_dilution=mic_dilution)


class PlateMeasurement:
    """Analyse a photograph of a multi-well plate to infer MIC values.

    Parameters
    ----------
    plate_image : str or pathlib.Path
        Directory containing the image files, or an image path whose parent
        directory should be used as the working location.
    new : bool, default=False
        Deprecated compatibility argument with no effect.
    categories : dict of str to Any, optional
        Metadata associated with the image and analysis.
    tags : str or list of str, default="PlateMeasurement"
        Deprecated compatibility argument with no effect.
    well_dimensions : tuple of int, default=(8, 12)
        Plate dimensions as ``(rows, columns)``.
    configuration_path : str or pathlib.Path, default="config"
        Deprecated compatibility argument with no effect.
    plate_design : str, optional
        Name of the bundled plate layout to use.
    pixel_intensities : bool, default=False
        Whether to retain raw pixel samples from measured well centres.

    Notes
    -----
    The public API intentionally stays close to the historic implementation so
    older scripts can keep working, while the internals use standard Python
    data structures, ``pathlib``, type hints, and explicit resource loading.
    """

    def __init__(
        self,
        plate_image: str | Path,
        new: bool = False,
        categories: dict[str, Any] | None = None,
        tags: str | list[str] = "PlateMeasurement",
        well_dimensions: tuple[int, int] = (8, 12),
        configuration_path: str | Path = "config",
        plate_design: str | None = None,
        pixel_intensities: bool = False,
    ) -> None:
        if new:
            warnings.warn(
                "'new' is deprecated and no longer affects PlateMeasurement.",
                DeprecationWarning,
                stacklevel=2,
            )
        if tags != "PlateMeasurement":
            warnings.warn(
                "'tags' is deprecated and no longer affects PlateMeasurement.",
                DeprecationWarning,
                stacklevel=2,
            )
        if str(configuration_path) != "config":
            warnings.warn(
                "'configuration_path' is deprecated; packaged layouts are loaded "
                "from amygda/config.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.base_path = Path(plate_image).expanduser()
        if self.base_path.suffix:
            inferred_name = self.base_path.stem
            self.base_path = self.base_path.parent
        else:
            inferred_name = self.base_path.name

        self.base_path.mkdir(parents=True, exist_ok=True)
        self.abspath = f"{self.base_path.resolve()}/"
        self.categories: dict[str, Any] = dict(categories or {})

        self.image_name = (
            self.categories.get("ImageFileName")
            or self.categories.get("IMAGEFILENAME")
            or inferred_name
        )
        self.name = Path(str(self.image_name)).stem

        self.well_dimensions = well_dimensions
        self.number_of_wells = well_dimensions[0] * well_dimensions[1]
        self.configuration_path = Path(configuration_path)
        self.plate_design = plate_design
        self.pixel_intensities = pixel_intensities

        self.image_path: Path | None = None
        self.image: ImageArray | None = None
        self.image_colour = False
        self.image_dimensions: tuple[int, ...] | None = None
        self.scaling_factor = 1.0

        self.well_index: FloatArray = np.zeros(self.well_dimensions, dtype=np.float64)
        self.well_radii: FloatArray = np.zeros(self.well_dimensions, dtype=np.float64)
        self.well_centre: PointArray = np.zeros((*self.well_dimensions, 2), dtype=int)
        self.well_top_left: PointArray = np.zeros((*self.well_dimensions, 2), dtype=int)
        self.well_bottom_right: PointArray = np.zeros((*self.well_dimensions, 2), dtype=int)
        self.well_growth: FloatArray = np.zeros(self.well_dimensions, dtype=np.float64)

        self.well_drug_name: NDArray[np.str_] | None = None
        self.well_drug_conc: FloatArray | None = None
        self.well_drug_dilution: IntArray | None = None
        self.well_positive_controls: list[tuple[int, int]] = []
        self.well_positive_controls_number = 0
        self.drug_names: list[str] = []
        self.drug_orientation: dict[str, str] = {}
        self.well_pixel_intensities: dict[tuple[int, int], list[int]] = {}

        self.threshold_pixel = 130
        self.threshold_percentage = 3.0
        self.sensitivity = 4.0

    def load_image(self, file_ending: str) -> None:
        """Load an image and initialize analysis state.

        Parameters
        ----------
        file_ending : str
            Filename suffix appended to ``image_name`` when resolving the image.
        """

        image_filename = self._resolve_image_filename(file_ending)
        image = cv2.imread(str(image_filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_filename}")

        self.image_path = image_filename
        self.image = image
        self.image_colour = True
        self.image_dimensions = image.shape
        self.scaling_factor = image.shape[1] / 1000.0

        self.well_index.fill(0.0)
        self.well_radii.fill(0.0)
        self.well_centre.fill(0)
        self.well_top_left.fill(0)
        self.well_bottom_right.fill(0)
        self.well_growth.fill(0.0)

        if self.pixel_intensities:
            self.well_pixel_intensities = {
                (i, j): []
                for i in range(self.well_dimensions[0])
                for j in range(self.well_dimensions[1])
            }

    def initialize_plate_layout(self) -> None:
        """Load plate layout metadata for commands that need drug information.

        Returns
        -------
        None
        """

        layout = self._load_plate_layout()
        self.well_drug_name = layout.drug_names
        self.well_drug_conc = layout.concentrations
        self.well_drug_dilution = layout.dilutions
        self.well_positive_controls = self._find_positive_controls()
        self.well_positive_controls_number = len(self.well_positive_controls)
        self.drug_names = np.unique(self.well_drug_name).tolist()
        self.drug_orientation = self._infer_drug_orientation()

    def _scale_value(self, value: int | float) -> int | float:
        """Scale a value relative to an image width of 1000 pixels.

        Parameters
        ----------
        value : int or float
            Value to scale.

        Returns
        -------
        int or float
            Scaled value, preserving integer-like rounding for integer inputs.
        """

        scaled = self.scaling_factor * value
        if isinstance(value, int):
            return round(scaled)
        return float(scaled)

    def save_arrays(self, file_ending: str) -> None:
        """Persist measured arrays to disk.

        Parameters
        ----------
        file_ending : str
            Output suffix for the saved ``.npz`` file.
        """

        output_path = self.base_path / f"{self.image_name}{file_ending}"
        if (
            self.well_drug_name is None
            or self.well_drug_conc is None
            or self.well_drug_dilution is None
        ):
            raise RuntimeError("Plate layout must be loaded before saving arrays.")
        np.savez(
            output_path,
            well_index=self.well_index,
            well_radii=self.well_radii,
            well_centre=self.well_centre,
            well_top_left=self.well_top_left,
            well_bottom_right=self.well_bottom_right,
            well_growth=self.well_growth,
            well_drug_name=self.well_drug_name,
            well_drug_conc=self.well_drug_conc,
            well_drug_dilution=self.well_drug_dilution,
            threshold_pixel=self.threshold_pixel,
            threshold_percentage=self.threshold_percentage,
            sensitivity=self.sensitivity,
        )

        if self.pixel_intensities:
            with (self.base_path / f"{self.image_name}-pixels.pkl").open("wb") as handle:
                pickle.dump(self.well_pixel_intensities, handle)

    def save_segment_arrays(self, file_ending: str) -> None:
        """Persist only the segmentation geometry to disk.

        Parameters
        ----------
        file_ending : str
            Output suffix for the saved ``.npz`` file.
        """

        output_path = self.base_path / f"{self.image_name}{file_ending}"
        np.savez(
            output_path,
            well_index=self.well_index,
            well_radii=self.well_radii,
            well_centre=self.well_centre,
            well_top_left=self.well_top_left,
            well_bottom_right=self.well_bottom_right,
        )

    def load_arrays(self, file_ending: str, pixel_intensities: bool = False) -> None:
        """Load previously saved analysis arrays from disk.

        Parameters
        ----------
        file_ending : str
            Input suffix for the saved ``.npz`` file.
        pixel_intensities : bool, default=False
            Whether to also restore the pickled raw pixel measurements.
        """

        npz_path = self.base_path / f"{self.image_name}{file_ending}"
        with np.load(npz_path, allow_pickle=False) as npzfile:
            self.well_index = npzfile["well_index"]
            self.well_radii = npzfile["well_radii"]
            self.well_centre = npzfile["well_centre"]
            self.well_top_left = npzfile["well_top_left"]
            self.well_bottom_right = npzfile["well_bottom_right"]
            self.well_growth = npzfile["well_growth"]
            self.well_drug_name = npzfile["well_drug_name"]
            self.well_drug_conc = npzfile["well_drug_conc"]
            self.well_drug_dilution = npzfile["well_drug_dilution"]
            self.threshold_pixel = int(npzfile["threshold_pixel"])
            self.threshold_percentage = float(npzfile["threshold_percentage"])
            self.sensitivity = float(npzfile["sensitivity"])

        self.drug_names = np.unique(self.well_drug_name).tolist()
        self.drug_orientation = self._infer_drug_orientation()
        self.well_positive_controls = self._find_positive_controls()
        self.well_positive_controls_number = len(self.well_positive_controls)

        if pixel_intensities:
            with (self.base_path / f"{self.image_name}-pixels.pkl").open("rb") as handle:
                self.well_pixel_intensities = pickle.load(handle)

    def load_segment_arrays(self, file_ending: str) -> None:
        """Load saved segmentation geometry from disk.

        Parameters
        ----------
        file_ending : str
            Input suffix for the saved ``.npz`` file.
        """

        npz_path = self.base_path / f"{self.image_name}{file_ending}"
        with np.load(npz_path, allow_pickle=False) as npzfile:
            self.well_index = npzfile["well_index"]
            self.well_radii = npzfile["well_radii"]
            self.well_centre = npzfile["well_centre"]
            self.well_top_left = npzfile["well_top_left"]
            self.well_bottom_right = npzfile["well_bottom_right"]

    def mean_shift_filter(self, spatial_radius: int = 10, colour_radius: int = 10) -> None:
        """Apply a mean-shift filter to reduce local noise.

        Parameters
        ----------
        spatial_radius : int, default=10
            Spatial radius passed to OpenCV.
        colour_radius : int, default=10
            Colour radius passed to OpenCV.
        """

        image = self._require_image()
        if not self.image_colour:
            self._convert_image_to_colour()
            image = self._require_image()
        self.image = cv2.pyrMeanShiftFiltering(image, spatial_radius, colour_radius)

    def equalise_histograms(self) -> None:
        """Apply a global histogram equalization."""

        if self.image_colour:
            self._convert_image_to_grey()
        self.image = cv2.equalizeHist(self._require_image())

    def plot_histogram(self, file_ending: str) -> None:
        """Plot and save a histogram for the current image.

        Parameters
        ----------
        file_ending : str
            Output suffix for the saved histogram image.
        """

        image = self._require_image()
        plt.tight_layout()
        fig = plt.figure(figsize=(4, 1.8))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_xlim((0.0, 255.0))
        axis.yaxis.set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        plt.hist(
            image.flatten(),
            bins=25,
            histtype="stepfilled",
            align="left",
            alpha=0.5,
            color="black",
            edgecolor="black",
            linewidth=1,
        )
        fig.savefig(self.base_path / f"{self.image_name}{file_ending}")
        plt.close(fig)

    def stretch_histogram(self, debug: bool = False) -> None:
        """Stretch image contrast around the modal intensity.

        Parameters
        ----------
        debug : bool, default=False
            If ``True``, print summary percentile information before and after
            stretching.
        """

        image = self._require_image().astype(np.int16)
        mode = self._image_mode(image)
        debug_line: str | None = None

        if debug:
            lower_debug = int(np.percentile(image, 5))
            upper_debug = int(np.percentile(image, 95))
            debug_line = f"{self.name},{lower_debug},{mode},{upper_debug}"

        image = image - mode
        lower = float(np.percentile(image, 5))
        upper = float(np.percentile(image, 95))
        pos_factor = 40.0 / upper if upper else 1.0
        neg_factor = -110.0 / lower if lower else 1.0

        stretched = np.multiply(image, np.where(image > 0, pos_factor, neg_factor)) + 180.0
        self.image = np.clip(stretched, 0, 255).astype(np.uint8)

        if debug:
            lower = int(np.percentile(self.image, 5))
            upper = int(np.percentile(self.image, 95))
            mode = self._image_mode(self.image)
            print(f"{debug_line},{lower},{mode},{upper}")

    def equalise_histograms_locally(self) -> None:
        """Apply CLAHE to better normalize lighting across the plate."""

        if self.image_colour:
            self._convert_image_to_grey()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=self.well_dimensions)
        self.image = clahe.apply(self._require_image())

    def save_image(self, file_ending: str) -> None:
        """Save the current image to disk.

        Parameters
        ----------
        file_ending : str
            Output suffix for the saved image.
        """

        cv2.imwrite(str(self.base_path / f"{self.image_name}{file_ending}"), self._require_image())

    def _convert_image_to_colour(self) -> None:
        """Convert a grayscale image into a 3-channel BGR image."""

        image = self._require_image()
        if image.ndim == 3:
            self.image = image
        else:
            self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.image_colour = True

    def _convert_image_to_grey(self) -> None:
        """Convert a color image into grayscale."""

        image = self._require_image()
        if image.ndim == 2:
            self.image = image
        else:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image_colour = False

    def annotate_well_circumference(
        self, color: tuple[int, int, int] = (0, 0, 0), linewidth: int = 1
    ) -> None:
        """Draw a circle around each identified well.

        Parameters
        ----------
        color : tuple of int, default=(0, 0, 0)
            BGR colour used for the annotation.
        linewidth : int, default=1
            Circle outline width.
        """

        if not self.image_colour:
            self._convert_image_to_colour()

        for iy in range(self.well_dimensions[0]):
            for ix in range(self.well_dimensions[1]):
                centre = tuple(int(value) for value in self.well_centre[iy, ix])
                radius = int(self.well_radii[iy, ix])
                cv2.circle(self._require_image(), centre, radius, color, linewidth)

    def annotate_well_centres(
        self, color: tuple[int, int, int] = (0, 0, 0), linewidth: int = 2
    ) -> None:
        """Draw a marker at the centre of each well.

        Parameters
        ----------
        color : tuple of int, default=(0, 0, 0)
            BGR colour used for the annotation.
        linewidth : int, default=2
            Marker outline width.
        """

        if not self.image_colour:
            self._convert_image_to_colour()

        for iy in range(self.well_dimensions[0]):
            for ix in range(self.well_dimensions[1]):
                centre = tuple(int(value) for value in self.well_centre[iy, ix])
                cv2.circle(self._require_image(), centre, 1, color, linewidth)

    def annotate_well_drugs_concs(
        self,
        color: tuple[int, int, int] = (0, 0, 0),
        fontsize: float = 0.4,
        fontemphasis: int = 1,
    ) -> None:
        """Label each well with its drug name and concentration.

        Parameters
        ----------
        color : tuple of int, default=(0, 0, 0)
            BGR colour used for the text.
        fontsize : float, default=0.4
            Relative font scale.
        fontemphasis : int, default=1
            OpenCV text thickness.
        """

        if self.well_drug_name is None or self.well_drug_conc is None:
            raise RuntimeError("Plate layout must be loaded before annotation.")
        if not self.image_colour:
            self._convert_image_to_colour()

        for iy in range(self.well_dimensions[0]):
            for ix in range(self.well_dimensions[1]):
                label1 = str(self.well_drug_name[iy, ix])
                label2 = str(self.well_drug_conc[iy, ix])
                x, y = (int(value) for value in self.well_centre[iy, ix])
                cv2.putText(
                    self._require_image(),
                    label1,
                    (x - round(15 * self.scaling_factor), y - round(20 * self.scaling_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.scaling_factor * fontsize,
                    color,
                    fontemphasis,
                )
                cv2.putText(
                    self._require_image(),
                    label2,
                    (x - round(15 * self.scaling_factor), y + round(30 * self.scaling_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.scaling_factor * fontsize,
                    color,
                    fontemphasis,
                )

    def annotate_well_analysed_region(
        self,
        growth_color: tuple[int, int, int] = (0, 0, 0),
        region: float = 0.4,
        thickness: int = 1,
    ) -> None:
        """Highlight the central region of wells classified as growing.

        Parameters
        ----------
        growth_color : tuple of int, default=(0, 0, 0)
            BGR colour used for highlighted wells.
        region : float, default=0.4
            Relative radius of the analysed region inside each well.
        thickness : int, default=1
            Annotation line width.
        """

        if not self.image_colour:
            self._convert_image_to_colour()

        growth_threshold_percentage = self._effective_growth_threshold()
        for iy in range(self.well_dimensions[0]):
            for ix in range(self.well_dimensions[1]):
                x, y = (int(value) for value in self.well_centre[iy, ix])
                radius = int(self.well_radii[iy, ix] * region)
                if self.well_growth[iy, ix] > growth_threshold_percentage:
                    color = (
                        growth_color if self.categories.get("IM_POS_GROWTH", True) else (0, 0, 0)
                    )
                    cv2.circle(self._require_image(), (x, y), radius, color, thickness=thickness)

    def delete_mics(self) -> None:
        """Remove previously stored MIC values from the metadata."""

        for drug in self.drug_names:
            self.categories.pop(f"IM_{drug.upper()}MIC", None)
            self.categories.pop(f"IM_{drug.upper()}DILUTION", None)

    def measure_growth(
        self,
        threshold_pixel: int = 130,
        threshold_percentage: float = 3,
        region: float = 0.4,
        sensitivity: float = 4.0,
    ) -> None:
        """Measure growth in each well and infer MIC values.

        Parameters
        ----------
        threshold_pixel : int, default=130
            Pixel intensity threshold below which pixels are treated as growth.
        threshold_percentage : float, default=3
            Percentage threshold above which a well is treated as growing.
        region : float, default=0.4
            Relative radius of the measured central region within each well.
        sensitivity : float, default=4.0
            Adaptive threshold factor applied when control wells are very dark.
        """

        if not 0 <= threshold_pixel <= 255:
            raise ValueError("threshold_pixel must take a value between 0 and 255")
        if not 0 <= threshold_percentage <= 100:
            raise ValueError("threshold_percentage must take a value between 0 and 100")
        if not 0 <= region <= 1:
            raise ValueError("region must take a value between 0 and 1")
        if (
            self.well_drug_name is None
            or self.well_drug_conc is None
            or self.well_drug_dilution is None
        ):
            raise RuntimeError("Plate layout must be loaded before measuring growth.")

        if self.image_colour:
            self._convert_image_to_grey()
        image = self._require_image()

        self.threshold_pixel = threshold_pixel
        self.threshold_percentage = threshold_percentage
        self.sensitivity = sensitivity

        y0, x0 = np.ogrid[: image.shape[0], : image.shape[1]]

        for iy in range(self.well_dimensions[0]):
            for ix in range(self.well_dimensions[1]):
                x = int(self.well_centre[iy, ix][0])
                y = int(self.well_centre[iy, ix][1])
                radius = float(self.well_radii[iy, ix]) * region
                circular_mask = (x0 - x) ** 2 + (y0 - y) ** 2 < radius**2
                rect_pixels = image[circular_mask].flatten()
                if rect_pixels.size == 0:
                    raise RuntimeError(f"No pixels were selected for well {(iy, ix)}.")

                if self.pixel_intensities:
                    self.well_pixel_intensities.setdefault((iy, ix), []).extend(
                        int(value) for value in rect_pixels
                    )

                self.well_growth[iy, ix] = float(
                    np.sum(rect_pixels < self.threshold_pixel, dtype=np.float64)
                    / rect_pixels.size
                    * 100.0
                )

        self._record_positive_controls()
        growth_threshold_percentage = self._effective_growth_threshold()

        number_drugs_inconsistent_growth = 0
        for drug in self.drug_names:
            growth = self.well_growth[self.well_drug_name == drug][::-1]
            conc = self.well_drug_conc[self.well_drug_name == drug][::-1]
            dilution = self.well_drug_dilution[self.well_drug_name == drug][::-1]

            growth = growth[dilution - 1]
            conc = conc[dilution - 1]
            dilution = np.arange(len(dilution), dtype=int) + 1
            mic_result = infer_mic(
                growth=growth,
                concentrations=conc,
                dilutions=dilution,
                growth_threshold_percentage=growth_threshold_percentage,
                controls_valid=bool(self.categories["IM_POS_GROWTH"]),
            )
            if mic_result.inconsistent_growth:
                number_drugs_inconsistent_growth += 1

            self.categories[f"IM_{drug.upper()}MIC"] = mic_result.mic_concentration
            self.categories[f"IM_{drug.upper()}DILUTION"] = mic_result.mic_dilution

        self.categories["IM_DRUGS_INCONSISTENT_GROWTH"] = number_drugs_inconsistent_growth

    def write_mics(self, file_ending: str) -> None:
        """Write the metadata summary to a human-readable text file.

        Parameters
        ----------
        file_ending : str
            Output suffix for the text report.
        """

        output_path = self.base_path / f"{self.image_name}{file_ending}"
        with output_path.open("w", encoding="utf-8") as handle:
            for field in sorted(self.categories):
                handle.write(f"{field:>28} {self.categories[field]:>20}\n")

    def identify_wells(
        self,
        hough_param1: int = 20,
        hough_param2: int = 25,
        radius_tolerance: float = 0.005,
        verbose: bool = False,
    ) -> bool:
        """Locate wells using a Hough circle transform.

        Parameters
        ----------
        hough_param1 : int, default=20
            First Hough transform parameter passed to OpenCV.
        hough_param2 : int, default=25
            Second Hough transform parameter passed to OpenCV.
        radius_tolerance : float, default=0.005
            Increment used when widening the allowable well-radius search range.
        verbose : bool, default=False
            If ``True``, print circle-detection progress information.

        Returns
        -------
        bool
            ``True`` if exactly one circle is found for every well.
        """

        if self.image_dimensions is None or self.image_path is None:
            raise RuntimeError("An image must be loaded before identifying wells.")

        estimate_well_y = float(self.image_dimensions[0]) / self.well_dimensions[0]
        estimate_well_x = float(self.image_dimensions[1]) / self.well_dimensions[1]

        if max(estimate_well_x, estimate_well_y) > 1.05 * min(estimate_well_x, estimate_well_y):
            return False

        estimated_radius = (estimate_well_x + estimate_well_y) / 4.0
        radius_multiplier = 1.0 + radius_tolerance
        grey_image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if grey_image is None:
            raise FileNotFoundError(f"Could not read image at {self.image_path}")

        circles: NDArray[Any] | None = None
        while True:
            circles = None
            while circles is None:
                circles = cast(
                    NDArray[Any] | None,
                    cv2.HoughCircles(
                        grey_image,
                        cv2.HOUGH_GRADIENT,
                        1,
                        50,
                        param1=hough_param1,
                        param2=hough_param2,
                        minRadius=int(estimated_radius / radius_multiplier),
                        maxRadius=int(estimated_radius * radius_multiplier),
                    ),
                )
                radius_multiplier += radius_tolerance
                if verbose and circles is None:
                    print(
                        "No circles, "
                        f"{int(estimated_radius / radius_multiplier)} < radius < "
                        f"{int(estimated_radius * radius_multiplier)}"
                    )

            number_of_circles = len(circles[0])
            if verbose:
                print(
                    f"{number_of_circles} circles, "
                    f"{int(estimated_radius / radius_multiplier)} < radius < "
                    f"{int(estimated_radius * radius_multiplier)}"
                )

            if number_of_circles >= self.number_of_wells or radius_multiplier > 2:
                break
            radius_multiplier += radius_tolerance

        assert circles is not None
        well_counter = 0
        one_circle_per_well = True

        for ix in range(self.well_dimensions[1]):
            for iy in range(self.well_dimensions[0]):
                top_left = (int(ix * estimate_well_x), int(iy * estimate_well_y))
                bottom_right = (int((ix + 1) * estimate_well_x), int((iy + 1) * estimate_well_y))

                matched_circle: NDArray[Any] | None = None
                number_of_circles_in_well = 0
                for circle in circles[0]:
                    if (
                        top_left[0] < circle[0] < bottom_right[0]
                        and top_left[1] < circle[1] < bottom_right[1]
                    ):
                        number_of_circles_in_well += 1
                        matched_circle = circle

                if number_of_circles_in_well == 1 and matched_circle is not None:
                    well_centre = (int(matched_circle[0]), int(matched_circle[1]))
                    well_radius = float(matched_circle[2])
                    well_extent = 1.2 * well_radius

                    x1 = max(0, int(well_centre[0] - well_extent))
                    x2 = min(self.image_dimensions[1], int(well_centre[0] + well_extent))
                    y1 = max(0, int(well_centre[1] - well_extent))
                    y2 = min(self.image_dimensions[0], int(well_centre[1] + well_extent))

                    self.well_index[iy, ix] = well_counter
                    self.well_centre[iy, ix] = well_centre
                    self.well_radii[iy, ix] = well_radius
                    self.well_top_left[iy, ix] = (x1, y1)
                    self.well_bottom_right[iy, ix] = (x2, y2)
                    well_counter += 1
                else:
                    if verbose:
                        print(number_of_circles_in_well)
                    one_circle_per_well = False

        return well_counter == self.number_of_wells and one_circle_per_well

    def _resolve_image_filename(self, file_ending: str) -> Path:
        """Resolve the concrete filename for the current image stem.

        Parameters
        ----------
        file_ending : str
            Suffix used to construct the image filename.

        Returns
        -------
        pathlib.Path
            Resolved image path.
        """

        image_name = str(self.image_name)
        if file_ending and image_name.endswith(file_ending):
            filename = image_name
        else:
            filename = f"{image_name}{file_ending}"
        return self.base_path / filename

    def _load_plate_layout(self) -> PlateLayout:
        """Load the configured plate layout from package resources.

        Returns
        -------
        PlateLayout
            Loaded plate layout.
        """

        if self.plate_design is None:
            raise ValueError("plate_design must be provided before loading an image.")

        prefix = f"{self.plate_design}"
        drug_names = self._load_matrix(f"config/{prefix}-drug-matrix.txt", dtype=str)
        concentrations = self._load_matrix(f"config/{prefix}-conc-matrix.txt", dtype=float)
        dilutions = self._load_matrix(f"config/{prefix}-dilution-matrix.txt", dtype=int)
        return PlateLayout(
            drug_names=np.asarray(drug_names, dtype=str),
            concentrations=np.asarray(concentrations, dtype=np.float64),
            dilutions=np.asarray(dilutions, dtype=int),
        )

    def _load_matrix(self, resource_name: str, dtype: type[Any]) -> NDArray[Any]:
        """Load a CSV-like matrix from package resources.

        Parameters
        ----------
        resource_name : str
            Resource path relative to the package root.
        dtype : type
            Data type passed to ``numpy.loadtxt``.

        Returns
        -------
        numpy.ndarray
            Loaded matrix.
        """

        resource = resources.files("amygda").joinpath(resource_name)
        with resource.open("r", encoding="utf-8") as handle:
            return np.loadtxt(handle, delimiter=",", dtype=dtype)

    def _infer_drug_orientation(self) -> dict[str, str]:
        """Infer the geometric orientation of each drug strip on the plate.

        Returns
        -------
        dict of str to str
            Mapping from drug name to inferred orientation label.
        """

        if self.well_drug_name is None:
            return {}

        drug_orientation: dict[str, str] = {}
        for drug in self.drug_names:
            positions = np.argwhere(self.well_drug_name == drug)
            n_rows = len(np.unique(positions[:, 0]))
            n_cols = len(np.unique(positions[:, 1]))
            if n_rows == 1 and n_cols > 1:
                drug_orientation[drug] = "horizontal"
            elif n_rows > 1 and n_cols == 1:
                drug_orientation[drug] = "vertical"
            elif n_rows > 1 and n_cols == 2:
                cols = np.unique(positions[:, 1])
                column0 = positions[positions[:, 1] == cols[0]]
                column1 = positions[positions[:, 1] == cols[1]]
                if column0[0][0] == column1[0][0] and column0[1][0] == column1[1][0]:
                    drug_orientation[drug] = "P-shape"
                elif column0[0][-1] == column1[0][0]:
                    drug_orientation[drug] = "L-shape"
                else:
                    drug_orientation[drug] = "unknown"
            elif n_rows == 1 and n_cols == 2:
                drug_orientation[drug] = "double-column"
            else:
                drug_orientation[drug] = "unknown"
        return drug_orientation

    def _find_positive_controls(self) -> list[tuple[int, int]]:
        """Locate positive control wells from the loaded layout.

        Returns
        -------
        list of tuple of int
            Indices of wells that act as positive controls.
        """

        if self.well_drug_name is None or self.well_drug_conc is None:
            return []
        return [
            (iy, ix)
            for iy in range(self.well_dimensions[0])
            for ix in range(self.well_dimensions[1])
            if self.well_drug_conc[iy, ix] == 0.0 and self.well_drug_name[iy, ix] == "POS"
        ]

    def _record_positive_controls(self) -> None:
        """Store positive-control summary statistics in ``categories``."""

        positive_control_growth_total = 0.0
        self.categories["IM_POS_GROWTH"] = True

        for counter, control_well in enumerate(self.well_positive_controls, start=1):
            positive_control_growth = float(self.well_growth[control_well])
            positive_control_growth_total += positive_control_growth
            self.categories[f"IM_POS{counter}GROWTH"] = float(f"{positive_control_growth:.2f}")
            has_growth = positive_control_growth > self.threshold_percentage
            self.categories[f"IM_POS{counter}"] = has_growth
            if not has_growth:
                self.categories["IM_POS_GROWTH"] = False

        if self.well_positive_controls_number == 0:
            raise RuntimeError("No positive control wells were defined for this plate.")

        self.categories["IM_POS_AVERAGE"] = float(
            f"{positive_control_growth_total / self.well_positive_controls_number:.2f}"
        )

    def _effective_growth_threshold(self) -> float:
        """Calculate the effective growth threshold for this plate.

        Returns
        -------
        float
            Effective percentage threshold used to classify growth.
        """

        positive_average = float(self.categories.get("IM_POS_AVERAGE", 0.0))
        if self.sensitivity == 0:
            return float(self.threshold_percentage)
        if positive_average > self.sensitivity * self.threshold_percentage:
            return positive_average / self.sensitivity
        return float(self.threshold_percentage)

    def _require_image(self) -> ImageArray:
        """Return the current image or raise if no image has been loaded.

        Returns
        -------
        numpy.ndarray
            Loaded image data.
        """

        if self.image is None:
            raise RuntimeError("An image must be loaded before this operation.")
        return self.image

    @staticmethod
    def _image_mode(image: NDArray[np.integer[Any]]) -> int:
        """Return the modal pixel intensity for the given image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        Returns
        -------
        int
            Modal pixel intensity.
        """

        flattened = np.clip(image.astype(int).ravel(), 0, 255)
        return int(np.bincount(flattened, minlength=256).argmax())
