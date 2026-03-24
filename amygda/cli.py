"""Command line interface for AMyGDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import PlateMeasurement

VALID_PLATE_DESIGNS = {
    "UKMYC5",
    "UKMYC6",
    "GPALL1F",
    "CHNMCMM2",
    "GB1ECSDP",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--growth_pixel_threshold",
        default=130,
        type=int,
        help="Pixels below this intensity are treated as growth (default: 130)",
    )
    parser.add_argument(
        "--growth_percentage",
        default=2,
        type=float,
        help="Percentage of dark pixels required to classify growth (default: 2)",
    )
    parser.add_argument(
        "--measured_region",
        default=0.5,
        type=float,
        help="Radius of the central measured region relative to the well (default: 0.5)",
    )
    parser.add_argument(
        "--sensitivity",
        default=4,
        type=float,
        help="Sensitivity factor applied when control wells are very dark (default: 4)",
    )
    parser.add_argument(
        "--file_ending",
        default="-raw",
        type=str,
        help="Suffix stripped from the input filename stem (default: -raw)",
    )
    parser.add_argument(
        "--plate_design",
        default="UKMYC5",
        choices=sorted(VALID_PLATE_DESIGNS),
        help="Plate design to use for drug, concentration, and dilution maps",
    )
    return parser


def main() -> None:
    """Run plate analysis from the command line."""

    options = build_parser().parse_args()
    image_path = Path(options.image).expanduser()
    image_dir = image_path.parent if image_path.parent != Path("") else Path(".")
    image_name = image_path.name
    if options.file_ending in image_name:
        image_name = image_name.split(options.file_ending)[0]
    else:
        image_name = image_path.stem

    plate = PlateMeasurement(
        image_dir,
        categories={"ImageFileName": image_name},
        configuration_path="config",
        pixel_intensities=False,
        plate_design=options.plate_design,
    )

    pink = (138, 41, 231)
    yellow = (51, 255, 255)
    black = (0, 0, 0)

    plate.load_image(f"{options.file_ending}{image_path.suffix}")
    plate.categories["IM_IMAGE_DOWNLOADED"] = True

    plate.mean_shift_filter()
    plate.save_image("-msf.jpg")

    plate.equalise_histograms_locally()
    plate.save_image("-clahe.jpg")

    plate.stretch_histogram(debug=False)
    plate.save_image("-filtered.jpg")
    plate.categories["IM_IMAGE_FILTERED"] = True

    plate.load_image("-filtered.jpg")
    wells_identified = plate.identify_wells(
        hough_param1=20,
        hough_param2=25,
        radius_tolerance=0.005,
        verbose=False,
    )
    if wells_identified:
        plate.categories["IM_WELLS_IDENTIFIED"] = True
        plate.measure_growth(
            region=options.measured_region,
            threshold_pixel=options.growth_pixel_threshold,
            threshold_percentage=options.growth_percentage,
            sensitivity=options.sensitivity,
        )
        plate.save_arrays("-arrays.npz")
        plate.annotate_well_circumference(color=pink, linewidth=2)
        plate.annotate_well_drugs_concs(color=black, fontsize=0.5)
        plate.annotate_well_analysed_region(
            growth_color=yellow,
            region=options.measured_region,
            thickness=3,
        )
        plate.save_image("-growth.jpg")
        plate.write_mics("-mics.txt")


if __name__ == "__main__":
    main()
