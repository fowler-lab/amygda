"""Command line interface for AMyGDA."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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
        plate.load_segment_arrays("-arrays.npz")

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
        help="Plate design to use for drug, concentration, and dilution maps",
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


if __name__ == "__main__":
    main()
