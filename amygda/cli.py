"""Command line interface for AMyGDA."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from . import PlateMeasurement

VALID_PLATE_DESIGNS = {
    "UKMYC5",
    "UKMYC6",
    "GPALL1F",
    "CHNMCMM2",
    "GB1ECSDP",
}

PINK = (138, 41, 231)
YELLOW = (51, 255, 255)
BLACK = (0, 0, 0)
STAGE_SUFFIXES = ("filtered", "segmented", "growth")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """

    parser = argparse.ArgumentParser(prog="amygda")
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter", help="Filter an input image.")
    _add_common_image_arguments(filter_parser)
    filter_parser.set_defaults(handler=handle_filter)

    segment_parser = subparsers.add_parser("segment", help="Identify wells in an input image.")
    _add_common_image_arguments(segment_parser)
    _add_segment_arguments(segment_parser)
    segment_parser.set_defaults(handler=handle_segment)

    measure_parser = subparsers.add_parser("measure", help="Measure growth from an input image.")
    _add_common_image_arguments(measure_parser)
    _add_plate_design_argument(measure_parser)
    _add_measure_arguments(measure_parser)
    measure_parser.set_defaults(handler=handle_measure)

    strips_parser = subparsers.add_parser(
        "strips",
        help="Automatically save individual PNG panel images for each drug.",
    )
    _add_common_image_arguments(strips_parser)
    _add_plate_design_argument(strips_parser)
    strips_parser.set_defaults(handler=handle_save_strips)

    panels_parser = subparsers.add_parser(
        "panels",
        help="Automatically save individual PNG panel images with positive controls.",
    )
    _add_common_image_arguments(panels_parser)
    _add_plate_design_argument(panels_parser)
    panels_parser.set_defaults(handler=handle_save_panels)

    run_parser = subparsers.add_parser(
        "run",
        help="Run filtering, segmentation, and measurement in sequence.",
    )
    _add_common_image_arguments(run_parser)
    _add_plate_design_argument(run_parser)
    _add_segment_arguments(run_parser)
    _add_measure_arguments(run_parser)
    run_parser.set_defaults(handler=handle_run)

    return parser


def handle_filter(options: argparse.Namespace) -> None:
    """Run the ``filter`` command.

    Parameters
    ----------
    options : argparse.Namespace
        Parsed command-line arguments.
    """

    image_info = ImageInfo.from_path(options.image)
    plate = _build_plate(image_info)
    run_filter_stage(plate, image_info)


def handle_segment(options: argparse.Namespace) -> None:
    """Run the ``segment`` command.

    Parameters
    ----------
    options : argparse.Namespace
        Parsed command-line arguments.
    """

    image_info = ImageInfo.from_path(options.image)
    plate = _build_plate(image_info)
    run_segment_stage(plate, image_info, options)


def handle_measure(options: argparse.Namespace) -> None:
    """Run the ``measure`` command.

    Parameters
    ----------
    options : argparse.Namespace
        Parsed command-line arguments.
    """

    image_info = ImageInfo.from_path(options.image)
    plate = _build_plate(image_info, options.plate_design)
    run_measure_stage(plate, image_info, options)


def handle_run(options: argparse.Namespace) -> None:
    """Run all pipeline stages with minimal reloads.

    Parameters
    ----------
    options : argparse.Namespace
        Parsed command-line arguments.
    """

    raw_info = ImageInfo.from_path(options.image)
    plate = _build_plate(raw_info, options.plate_design)

    filtered_info = run_filter_stage(plate, raw_info)
    segmented_info = run_segment_stage(plate, filtered_info, options, reuse_loaded_image=True)
    run_measure_stage(plate, segmented_info, options, reuse_loaded_state=True)


def handle_save_strips(options: argparse.Namespace) -> None:
    """Run the ``strips`` command.

    Parameters
    ----------
    options : argparse.Namespace
        Parsed command-line arguments.
    """

    image_info = ImageInfo.from_path(options.image)
    plate = _build_plate(image_info, options.plate_design)
    plate.load_image(image_info.suffix)
    segment_info = image_info.with_stage("segmented")
    plate.image_name = segment_info.stem
    plate.load_segment_arrays("-arrays.npz")
    plate.image_name = image_info.stem
    plate.initialize_plate_layout()
    save_drug_images(plate, image_info, include_controls=False)


def handle_save_panels(options: argparse.Namespace) -> None:
    """Run the ``panels`` command.

    Parameters
    ----------
    options : argparse.Namespace
        Parsed command-line arguments.
    """

    image_info = ImageInfo.from_path(options.image)
    plate = _build_plate(image_info, options.plate_design)
    plate.load_image(image_info.suffix)
    segment_info = image_info.with_stage("segmented")
    plate.image_name = segment_info.stem
    plate.load_segment_arrays("-arrays.npz")
    plate.image_name = image_info.stem
    plate.initialize_plate_layout()
    save_drug_images(plate, image_info, include_controls=True)


def run_filter_stage(plate: PlateMeasurement, image_info: ImageInfo) -> ImageInfo:
    """Execute the filtering stage.

    Parameters
    ----------
    plate : PlateMeasurement
        Plate object to update.
    image_info : ImageInfo
        Exact input image information.

    Returns
    -------
    ImageInfo
        Output image information for the filtered image.
    """

    plate.load_image(image_info.suffix)
    plate.categories["IM_IMAGE_DOWNLOADED"] = True
    plate.mean_shift_filter()
    plate.equalise_histograms_locally()
    plate.stretch_histogram(debug=False)

    output_info = image_info.with_stage("filtered")
    plate.image_name = output_info.stem
    plate.save_image(output_info.suffix)
    plate.categories["IM_IMAGE_FILTERED"] = True
    return output_info


def run_segment_stage(
    plate: PlateMeasurement,
    image_info: ImageInfo,
    options: argparse.Namespace,
    *,
    reuse_loaded_image: bool = False,
) -> ImageInfo:
    """Execute the segmentation stage.

    Parameters
    ----------
    plate : PlateMeasurement
        Plate object to update.
    image_info : ImageInfo
        Exact input image information.
    options : argparse.Namespace
        Parsed command-line arguments.
    reuse_loaded_image : bool, default=False
        If ``True``, reuse the in-memory image instead of reloading it.

    Returns
    -------
    ImageInfo
        Output image information for the segmented image.
    """

    if not reuse_loaded_image:
        plate.load_image(image_info.suffix)

    wells_identified = plate.identify_wells(
        hough_param1=options.hough_param1,
        hough_param2=options.hough_param2,
        radius_tolerance=options.radius_tolerance,
        verbose=options.verbose,
    )
    if not wells_identified:
        raise RuntimeError("Failed to identify the expected set of wells.")

    plate.categories["IM_WELLS_IDENTIFIED"] = True
    output_info = image_info.with_stage("segmented")
    plate.image_name = output_info.stem
    plate.save_segment_arrays("-arrays.npz")
    if options.save_segmented_image:
        plate.annotate_well_circumference(color=PINK, linewidth=2)
        plate.save_image(output_info.suffix)
    return output_info


def run_measure_stage(
    plate: PlateMeasurement,
    image_info: ImageInfo,
    options: argparse.Namespace,
    *,
    reuse_loaded_state: bool = False,
) -> ImageInfo:
    """Execute the measurement stage.

    Parameters
    ----------
    plate : PlateMeasurement
        Plate object to update.
    image_info : ImageInfo
        Exact input image information.
    options : argparse.Namespace
        Parsed command-line arguments.
    reuse_loaded_state : bool, default=False
        If ``True``, reuse the in-memory image and well metadata.

    Returns
    -------
    ImageInfo
        Output image information for the growth image.
    """

    if not reuse_loaded_state:
        plate.load_image(image_info.suffix)
        segment_info = image_info.with_stage("segmented")
        plate.image_name = segment_info.stem
        plate.load_segment_arrays("-arrays.npz")
        plate.image_name = image_info.stem

    plate.initialize_plate_layout()

    plate.measure_growth(
        region=options.measured_region,
        threshold_pixel=options.growth_pixel_threshold,
        threshold_percentage=options.growth_percentage,
        sensitivity=options.sensitivity,
    )
    plate.save_arrays("-arrays.npz")
    plate.annotate_well_circumference(color=PINK, linewidth=2)
    plate.annotate_well_drugs_concs(color=BLACK, fontsize=0.5)
    plate.annotate_well_analysed_region(
        growth_color=YELLOW,
        region=options.measured_region,
        thickness=3,
    )
    output_info = image_info.with_stage("growth")
    plate.image_name = output_info.stem
    plate.save_image(output_info.suffix)
    plate.write_mics("-mics.txt")
    return output_info


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI.

    Parameters
    ----------
    argv : sequence of str, optional
        Explicit argument vector used instead of ``sys.argv``.
    """

    parser = build_parser()
    options = parser.parse_args(argv)
    options.handler(options)


def save_drug_images(
    plate: PlateMeasurement, image_info: ImageInfo, *, include_controls: bool
) -> None:
    """Write one strip or panel image for each drug on the plate.

    Parameters
    ----------
    plate : PlateMeasurement
        Plate measurement with image, segment arrays, and layout loaded.
    image_info : ImageInfo
        Exact input image metadata.
    include_controls : bool
        Whether to prepend positive-control wells above each drug strip.
    """

    if plate.image is None:
        raise RuntimeError("An image must be loaded before saving panels.")
    if (
        plate.well_drug_name is None
        or plate.well_drug_conc is None
        or plate.well_drug_dilution is None
    ):
        raise RuntimeError("Plate layout must be initialized before saving panels.")
    well_drug_name = plate.well_drug_name
    well_drug_conc = plate.well_drug_conc
    control_panel = _build_positive_control_panel(plate) if include_controls else None

    for drug in plate.drug_names:
        if drug == "POS":
            continue

        positions = np.argwhere(well_drug_name == drug)
        groups = _group_drug_positions(positions)
        ordered_groups = sorted(
            groups,
            key=lambda group: float(np.min(well_drug_conc[group[:, 0], group[:, 1]])),
        )
        panels = [_extract_oriented_drug_panel(plate, group) for group in ordered_groups]
        panel = _concatenate_panels_horizontally(panels)
        if control_panel is not None:
            panel = _stack_panels_vertically_left_aligned([control_panel, panel])
        output_info = image_info.with_suffix_label(f"{drug.lower()}-panel")
        cv2.imwrite(str(output_info.path), panel)


class ImageInfo:
    """Exact input/output image naming information.

    Parameters
    ----------
    directory : pathlib.Path
        Directory containing the image.
    stem : str
        Exact filename stem.
    suffix : str
        Exact filename suffix.
    """

    def __init__(self, directory: Path, stem: str, suffix: str) -> None:
        self.directory = directory
        self.stem = stem
        self.suffix = suffix
        self.base_stem = _strip_stage_suffix(stem)

    @classmethod
    def from_path(cls, image: str) -> ImageInfo:
        """Build image naming metadata from a CLI image argument.

        Parameters
        ----------
        image : str
            Exact image path.

        Returns
        -------
        ImageInfo
            Parsed image metadata.
        """

        image_path = Path(image).expanduser()
        image_dir = image_path.parent if image_path.parent != Path("") else Path(".")
        return cls(directory=image_dir, stem=image_path.stem, suffix=image_path.suffix)

    def with_stage(self, stage: str) -> ImageInfo:
        """Return a new image info with an appended stage suffix.

        Parameters
        ----------
        stage : str
            Stage label to append to the filename stem.

        Returns
        -------
        ImageInfo
            Derived image metadata for the stage output.
        """

        return ImageInfo(self.directory, f"{self.base_stem}-{stage}", self.suffix)

    def with_suffix_label(self, label: str) -> ImageInfo:
        """Return a new image info with a custom suffix label.

        Parameters
        ----------
        label : str
            Label appended to the canonical base stem.

        Returns
        -------
        ImageInfo
            Derived image metadata.
        """

        return ImageInfo(self.directory, f"{self.base_stem}-{label}", self.suffix)

    @property
    def path(self) -> Path:
        """Return the concrete filesystem path for this image.

        Returns
        -------
        pathlib.Path
            Concrete file path.
        """

        return self.directory / f"{self.stem}{self.suffix}"


def _build_plate(
    image_info: ImageInfo, plate_design: str | None = None
) -> PlateMeasurement:
    """Create a ``PlateMeasurement`` for the given image.

    Parameters
    ----------
    image_info : ImageInfo
        Exact image metadata.
    plate_design : str, optional
        Selected bundled plate layout.

    Returns
    -------
    PlateMeasurement
        Configured plate measurement object.
    """

    return PlateMeasurement(
        image_info.directory,
        categories={"ImageFileName": image_info.stem},
        configuration_path="config",
        pixel_intensities=False,
        plate_design=plate_design,
    )


def _add_common_image_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach shared image arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to update.
    """

    parser.add_argument("image", metavar="path", help="Exact path to the input image")


def _add_plate_design_argument(parser: argparse.ArgumentParser) -> None:
    """Attach the plate-design argument to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to update.
    """

    parser.add_argument(
        "--plate_design",
        default="UKMYC5",
        choices=sorted(VALID_PLATE_DESIGNS),
        help=(
            "Plate design to use for drug, concentration, and dilution maps "
            "(default: UKMYC5)"
        ),
    )


def _add_segment_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach segmentation arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to update.
    """

    parser.add_argument(
        "--hough_param1",
        default=20,
        type=int,
        metavar="value",
        help="First Hough transform parameter passed to OpenCV.",
    )
    parser.add_argument(
        "--hough_param2",
        default=25,
        type=int,
        metavar="value",
        help="Second Hough transform parameter passed to OpenCV.",
    )
    parser.add_argument(
        "--radius_tolerance",
        default=0.005,
        type=float,
        metavar="value",
        help="Increment used when widening the radius search range.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while identifying wells.",
    )
    parser.add_argument(
        "--save-segmented-image",
        action="store_true",
        help="Write the optional segmented overlay image.",
    )


def _add_measure_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach measurement arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to update.
    """

    parser.add_argument(
        "--growth_pixel_threshold",
        default=130,
        type=int,
        metavar="value",
        help="Pixels below this intensity are treated as growth (default: 130)",
    )
    parser.add_argument(
        "--growth_percentage",
        default=2,
        type=float,
        metavar="value",
        help="Percentage of dark pixels required to classify growth (default: 2)",
    )
    parser.add_argument(
        "--measured_region",
        default=0.5,
        type=float,
        metavar="value",
        help="Radius of the central measured region relative to the well (default: 0.5)",
    )
    parser.add_argument(
        "--sensitivity",
        default=4,
        type=float,
        metavar="value",
        help="Sensitivity factor applied when control wells are very dark (default: 4)",
    )


def _strip_stage_suffix(stem: str) -> str:
    """Remove a known pipeline-stage suffix from a filename stem.

    Parameters
    ----------
    stem : str
        Input filename stem.

    Returns
    -------
    str
        Filename stem with any terminal stage suffix removed.
    """

    for stage in STAGE_SUFFIXES:
        suffix = f"-{stage}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _group_drug_positions(positions: np.ndarray) -> list[np.ndarray]:
    """Group drug wells into strips for later rotation and concatenation.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of ``(row, column)`` indices for wells belonging to one drug.

    Returns
    -------
    list of numpy.ndarray
        Position groups, typically one per occupied column when the drug spans
        multiple columns.
    """

    unique_rows = np.unique(positions[:, 0])
    unique_columns = np.unique(positions[:, 1])
    if len(unique_rows) == 1 or len(unique_columns) == 1:
        return [positions]
    if len(unique_columns) > 1:
        return [positions[positions[:, 1] == column] for column in unique_columns]
    return [positions]


def _extract_oriented_drug_panel(
    plate: PlateMeasurement, positions: np.ndarray
) -> np.ndarray:
    """Extract and orient a drug strip so low concentration is on the left.

    Parameters
    ----------
    plate : PlateMeasurement
        Plate measurement with image, segmentation geometry, and layout loaded.
    positions : numpy.ndarray
        Array of ``(row, column)`` indices for one strip of drug wells.

    Returns
    -------
    numpy.ndarray
        Cropped and oriented drug panel.
    """

    image = plate.image
    well_drug_conc = plate.well_drug_conc
    assert image is not None
    assert well_drug_conc is not None
    top_left = plate.well_top_left[positions[:, 0], positions[:, 1]]
    bottom_right = plate.well_bottom_right[positions[:, 0], positions[:, 1]]
    x1 = int(np.min(top_left[:, 0]))
    y1 = int(np.min(top_left[:, 1]))
    x2 = int(np.max(bottom_right[:, 0]))
    y2 = int(np.max(bottom_right[:, 1]))
    panel = image[y1:y2, x1:x2]
    centres = plate.well_centre[positions[:, 0], positions[:, 1]]
    concentrations = well_drug_conc[positions[:, 0], positions[:, 1]]

    x_span = float(np.max(centres[:, 0]) - np.min(centres[:, 0]))
    y_span = float(np.max(centres[:, 1]) - np.min(centres[:, 1]))

    if y_span > x_span:
        top_index = int(np.argmin(centres[:, 1]))
        bottom_index = int(np.argmax(centres[:, 1]))
        if float(concentrations[top_index]) <= float(concentrations[bottom_index]):
            panel = cv2.rotate(panel, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            panel = cv2.rotate(panel, cv2.ROTATE_90_CLOCKWISE)
    else:
        left_index = int(np.argmin(centres[:, 0]))
        right_index = int(np.argmax(centres[:, 0]))
        if float(concentrations[left_index]) > float(concentrations[right_index]):
            panel = cv2.flip(panel, 1)

    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    return panel


def _concatenate_panels_horizontally(panels: list[np.ndarray]) -> np.ndarray:
    """Join multiple oriented panels into one contiguous horizontal image.

    Parameters
    ----------
    panels : list of numpy.ndarray
        Oriented per-strip panels ordered from low to high concentration.

    Returns
    -------
    numpy.ndarray
        Combined horizontal panel image.
    """

    if not panels:
        raise ValueError("At least one panel is required.")

    max_height = max(panel.shape[0] for panel in panels)
    total_width = sum(panel.shape[1] for panel in panels)
    combined = np.full((max_height, total_width, 3), 255, dtype=np.uint8)

    x_offset = 0
    for panel in panels:
        height, width = panel.shape[:2]
        y_offset = (max_height - height) // 2
        combined[y_offset : y_offset + height, x_offset : x_offset + width] = panel
        x_offset += width
    return combined


def _build_positive_control_panel(plate: PlateMeasurement) -> np.ndarray:
    """Build a horizontal positive-control strip.

    Parameters
    ----------
    plate : PlateMeasurement
        Plate measurement with layout and segmentation loaded.

    Returns
    -------
    numpy.ndarray
        Horizontal positive-control panel.
    """

    control_positions = np.array(plate.well_positive_controls, dtype=int)
    if control_positions.size == 0:
        raise RuntimeError("No positive control wells were found for this plate design.")

    groups = _group_drug_positions(control_positions)
    ordered_groups = sorted(groups, key=lambda group: int(np.min(group[:, 1])))
    panels = [_extract_oriented_control_panel(plate, group) for group in ordered_groups]
    return _add_border(_concatenate_panels_horizontally(panels))


def _extract_oriented_control_panel(
    plate: PlateMeasurement, positions: np.ndarray
) -> np.ndarray:
    """Extract and orient a positive-control strip horizontally.

    Parameters
    ----------
    plate : PlateMeasurement
        Plate measurement with image and segmentation geometry loaded.
    positions : numpy.ndarray
        Array of ``(row, column)`` indices for one positive-control strip.

    Returns
    -------
    numpy.ndarray
        Cropped and oriented positive-control panel.
    """

    image = plate.image
    assert image is not None
    top_left = plate.well_top_left[positions[:, 0], positions[:, 1]]
    bottom_right = plate.well_bottom_right[positions[:, 0], positions[:, 1]]
    x1 = int(np.min(top_left[:, 0]))
    y1 = int(np.min(top_left[:, 1]))
    x2 = int(np.max(bottom_right[:, 0]))
    y2 = int(np.max(bottom_right[:, 1]))
    panel = image[y1:y2, x1:x2]
    centres = plate.well_centre[positions[:, 0], positions[:, 1]]

    x_span = float(np.max(centres[:, 0]) - np.min(centres[:, 0]))
    y_span = float(np.max(centres[:, 1]) - np.min(centres[:, 1]))
    if y_span > x_span:
        panel = cv2.rotate(panel, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    return panel


def _stack_panels_vertically_left_aligned(panels: list[np.ndarray]) -> np.ndarray:
    """Stack panels vertically with left edges aligned.

    Parameters
    ----------
    panels : list of numpy.ndarray
        Panels ordered from top to bottom.

    Returns
    -------
    numpy.ndarray
        Combined vertical panel image.
    """

    if not panels:
        raise ValueError("At least one panel is required.")

    max_width = max(panel.shape[1] for panel in panels)
    total_height = sum(panel.shape[0] for panel in panels)
    combined = np.full((total_height, max_width, 3), 255, dtype=np.uint8)

    y_offset = 0
    for panel in panels:
        height, width = panel.shape[:2]
        combined[y_offset : y_offset + height, :width] = panel
        y_offset += height
    return combined


def _add_border(panel: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Add a thin black border around a panel image.

    Parameters
    ----------
    panel : numpy.ndarray
        Panel image to decorate.
    thickness : int, default=1
        Border thickness in pixels.

    Returns
    -------
    numpy.ndarray
        Bordered panel image.
    """

    bordered = panel.copy()
    cv2.rectangle(
        bordered,
        (0, 0),
        (bordered.shape[1] - 1, bordered.shape[0] - 1),
        (0, 0, 0),
        thickness=thickness,
    )
    return bordered


if __name__ == "__main__":
    main()
