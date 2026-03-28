from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from amygda import PlateMeasurement, infer_mic
from amygda.cli import _group_drug_positions, main


def test_scale_value_preserves_int_and_float_behavior(tmp_path: Path) -> None:
    plate = PlateMeasurement(
        tmp_path,
        categories={"ImageFileName": "plate"},
        plate_design="UKMYC5",
    )
    plate.scaling_factor = 0.5

    assert plate._scale_value(3) == 2
    assert plate._scale_value(3.0) == 1.5


def test_load_image_populates_layout_metadata(tmp_path: Path) -> None:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    image_path = tmp_path / "plate-raw.png"
    assert cv2.imwrite(str(image_path), image)

    plate = PlateMeasurement(
        tmp_path,
        categories={"ImageFileName": "plate"},
        plate_design="UKMYC5",
    )
    plate.load_image("-raw.png")
    plate.initialize_plate_layout()

    assert plate.image_dimensions == (80, 120, 3)
    assert plate.well_drug_name is not None
    assert plate.well_drug_name.shape == (8, 12)
    assert plate.well_positive_controls_number == 2
    assert "POS" in plate.drug_names
    assert plate.drug_orientation["BDQ"] == "vertical"


def test_measure_growth_infers_expected_mic_from_image_pixels(tmp_path: Path) -> None:
    plate = PlateMeasurement(
        tmp_path,
        categories={"ImageFileName": "synthetic"},
        plate_design="UKMYC5",
        well_dimensions=(2, 2),
    )

    image = np.full((40, 40), 220, dtype=np.uint8)
    centres = {
        (0, 0): (10, 10),
        (0, 1): (30, 10),
        (1, 0): (10, 30),
        (1, 1): (30, 30),
    }
    for well, centre in centres.items():
        radius = 5
        color = (40, 40, 40) if well in {(0, 0), (1, 0), (1, 1)} else (220, 220, 220)
        cv2.circle(image, centre, radius, color, thickness=-1)

    plate.image = image
    plate.image_colour = False
    plate.image_dimensions = image.shape
    plate.well_centre = np.array(
        [
            [[10, 10], [30, 10]],
            [[10, 30], [30, 30]],
        ],
        dtype=int,
    )
    plate.well_radii = np.full((2, 2), 5.0, dtype=np.float64)
    plate.well_drug_name = np.array([["DRUG", "DRUG"], ["POS", "POS"]], dtype=str)
    plate.well_drug_conc = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    plate.well_drug_dilution = np.array([[1, 2], [1, 2]], dtype=int)
    plate.drug_names = ["DRUG", "POS"]
    plate.well_positive_controls = [(1, 0), (1, 1)]
    plate.well_positive_controls_number = 2

    plate.measure_growth(threshold_pixel=130, threshold_percentage=10, region=0.8, sensitivity=4.0)

    assert plate.categories["IM_POS_GROWTH"] is True
    assert plate.categories["IM_DRUGMIC"] == pytest.approx(2.0)
    assert plate.categories["IM_DRUGDILUTION"] == 2
    assert plate.well_growth[0, 0] > 90
    assert plate.well_growth[0, 1] < 10


def test_measure_growth_marks_plate_invalid_without_control_growth(tmp_path: Path) -> None:
    plate = PlateMeasurement(
        tmp_path,
        categories={"ImageFileName": "synthetic"},
        plate_design="UKMYC5",
        well_dimensions=(2, 2),
    )

    image = np.full((40, 40), 220, dtype=np.uint8)
    cv2.circle(image, (10, 10), 5, (40, 40, 40), thickness=-1)
    cv2.circle(image, (30, 10), 5, (220, 220, 220), thickness=-1)
    cv2.circle(image, (10, 30), 5, (220, 220, 220), thickness=-1)
    cv2.circle(image, (30, 30), 5, (220, 220, 220), thickness=-1)

    plate.image = image
    plate.image_colour = False
    plate.image_dimensions = image.shape
    plate.well_centre = np.array(
        [
            [[10, 10], [30, 10]],
            [[10, 30], [30, 30]],
        ],
        dtype=int,
    )
    plate.well_radii = np.full((2, 2), 5.0, dtype=np.float64)
    plate.well_drug_name = np.array([["DRUG", "DRUG"], ["POS", "POS"]], dtype=str)
    plate.well_drug_conc = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    plate.well_drug_dilution = np.array([[1, 2], [1, 2]], dtype=int)
    plate.drug_names = ["DRUG", "POS"]
    plate.well_positive_controls = [(1, 0), (1, 1)]
    plate.well_positive_controls_number = 2

    plate.measure_growth(threshold_pixel=130, threshold_percentage=10, region=0.8, sensitivity=4.0)

    assert plate.categories["IM_POS_GROWTH"] is False
    assert plate.categories["IM_DRUGMIC"] == pytest.approx(-2.0)
    assert plate.categories["IM_DRUGDILUTION"] == -2


def test_infer_mic_flags_inconsistent_growth() -> None:
    result = infer_mic(
        growth=np.array([90.0, 0.0, 70.0], dtype=np.float64),
        concentrations=np.array([1.0, 2.0, 4.0], dtype=np.float64),
        dilutions=np.array([1, 2, 3], dtype=int),
        growth_threshold_percentage=10.0,
        controls_valid=True,
    )

    assert result.inconsistent_growth is True
    assert result.mic_concentration == -1.0
    assert result.mic_dilution == -1


def test_infer_mic_returns_invalid_when_controls_fail() -> None:
    result = infer_mic(
        growth=np.array([90.0, 0.0], dtype=np.float64),
        concentrations=np.array([1.0, 2.0], dtype=np.float64),
        dilutions=np.array([1, 2], dtype=int),
        growth_threshold_percentage=10.0,
        controls_valid=False,
    )

    assert result.inconsistent_growth is False
    assert result.mic_concentration == -2.0
    assert result.mic_dilution == -2


def test_identify_wells_finds_synthetic_grid(tmp_path: Path) -> None:
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    rows, cols = 2, 3
    centers: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            center = (60 + col * 120, 60 + row * 120)
            centers.append(center)
            cv2.circle(image, center, 32, (0, 0, 0), thickness=3)

    image_path = tmp_path / "plate-raw.png"
    assert cv2.imwrite(str(image_path), image)

    plate = PlateMeasurement(
        tmp_path,
        categories={"ImageFileName": "plate"},
        plate_design="UKMYC5",
        well_dimensions=(2, 3),
    )
    plate.load_image("-raw.png")

    assert plate.identify_wells(hough_param1=100, hough_param2=15, radius_tolerance=0.02) is True
    assert np.count_nonzero(plate.well_radii) == 6


def test_legacy_arguments_emit_deprecation_warnings(tmp_path: Path) -> None:
    with pytest.deprecated_call():
        PlateMeasurement(
            tmp_path,
            new=True,
            tags=["legacy"],
            configuration_path="custom-config",
            categories={"ImageFileName": "plate"},
            plate_design="UKMYC5",
        )


def test_stage_commands_save_and_load_pipeline_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full((120, 180, 3), 200, dtype=np.uint8)
    image_path = tmp_path / "plate-raw.png"
    assert cv2.imwrite(str(image_path), image)

    def fake_identify_wells(self: PlateMeasurement, **_: object) -> bool:
        self.well_centre[:] = (20, 20)
        self.well_radii[:] = 8
        return True

    def fake_measure_growth(self: PlateMeasurement, **_: object) -> None:
        self.well_growth[:] = 0.0
        self.categories["IM_POS_GROWTH"] = True
        self.categories["IM_POS_AVERAGE"] = 20.0
        self.categories["IM_DRUGS_INCONSISTENT_GROWTH"] = 0
        self.categories["IM_BDQMIC"] = 0.12
        self.categories["IM_BDQDILUTION"] = 1

    monkeypatch.setattr(PlateMeasurement, "identify_wells", fake_identify_wells)
    monkeypatch.setattr(PlateMeasurement, "measure_growth", fake_measure_growth)

    main(["filter", str(image_path)])
    filtered_path = tmp_path / "plate-raw-filtered.png"
    assert filtered_path.exists()

    main(["segment", str(filtered_path), "--save-segmented-image"])
    segmented_path = tmp_path / "plate-raw-segmented.png"
    arrays_path = tmp_path / "plate-raw-segmented-arrays.npz"
    assert segmented_path.exists()
    assert arrays_path.exists()

    main(["measure", str(segmented_path), "--plate_design", "UKMYC5"])
    assert (tmp_path / "plate-raw-growth.png").exists()
    assert (tmp_path / "plate-raw-growth-mics.txt").exists()


def test_run_command_executes_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full((120, 180, 3), 200, dtype=np.uint8)
    image_path = tmp_path / "plate-raw.png"
    assert cv2.imwrite(str(image_path), image)

    def fake_identify_wells(self: PlateMeasurement, **_: object) -> bool:
        self.well_centre[:] = (20, 20)
        self.well_radii[:] = 8
        return True

    def fake_measure_growth(self: PlateMeasurement, **_: object) -> None:
        self.well_growth[:] = 0.0
        self.categories["IM_POS_GROWTH"] = True
        self.categories["IM_POS_AVERAGE"] = 20.0
        self.categories["IM_DRUGS_INCONSISTENT_GROWTH"] = 0
        self.categories["IM_BDQMIC"] = 0.12
        self.categories["IM_BDQDILUTION"] = 1

    monkeypatch.setattr(PlateMeasurement, "identify_wells", fake_identify_wells)
    monkeypatch.setattr(PlateMeasurement, "measure_growth", fake_measure_growth)

    main(["run", str(image_path), "--plate_design", "UKMYC5"])

    assert (tmp_path / "plate-raw-filtered.png").exists()
    assert (tmp_path / "plate-raw-segmented-arrays.npz").exists()
    assert (tmp_path / "plate-raw-growth.png").exists()
    assert (tmp_path / "plate-raw-growth-mics.txt").exists()
    assert not (tmp_path / "plate-raw-segmented.png").exists()


def test_filter_parser_does_not_accept_plate_design() -> None:
    with pytest.raises(SystemExit):
        main(["filter", "plate.png", "--plate_design", "UKMYC5"])


def test_group_drug_positions_keeps_single_row_strip_together() -> None:
    positions = np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=int)
    groups = _group_drug_positions(positions)

    assert len(groups) == 1
    assert np.array_equal(groups[0], positions)


def test_save_panels_writes_low_to_high_drug_panels(tmp_path: Path) -> None:
    rows, cols = 8, 12
    cell_height = 20
    cell_width = 20
    image = np.full((rows * cell_height, cols * cell_width, 3), 255, dtype=np.uint8)
    filtered_path = tmp_path / "plate-filtered.png"
    assert cv2.imwrite(str(filtered_path), image)

    plate = PlateMeasurement(
        tmp_path,
        categories={"ImageFileName": "plate-filtered"},
        plate_design="UKMYC6",
    )
    plate.load_image(".png")
    plate.initialize_plate_layout()
    assert plate.well_drug_name is not None
    assert plate.well_drug_conc is not None
    assert plate.well_drug_dilution is not None
    assert plate.image is not None

    target_drug = "RIF"
    drug_positions = np.argwhere(plate.well_drug_name == target_drug)
    ranked_concs = sorted(
        {float(plate.well_drug_conc[row, col]) for row, col in drug_positions.tolist()}
    )
    intensity_map = {conc: 30 + (index * 20) for index, conc in enumerate(ranked_concs)}
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            plate.well_top_left[row, col] = (x1, y1)
            plate.well_bottom_right[row, col] = (x2, y2)
            plate.well_centre[row, col] = ((x1 + x2) // 2, (y1 + y2) // 2)
            plate.well_radii[row, col] = cell_width // 2
            intensity = 220
            if plate.well_drug_name[row, col] == target_drug:
                intensity = intensity_map[float(plate.well_drug_conc[row, col])]
            plate.image[y1:y2, x1:x2] = intensity

    plate.image_name = "plate-segmented"
    plate.save_segment_arrays("-arrays.npz")
    assert cv2.imwrite(str(filtered_path), plate.image)

    main(["strips", str(filtered_path), "--plate_design", "UKMYC6"])

    panel_path = tmp_path / f"plate-{target_drug.lower()}-panel.png"
    assert panel_path.exists()

    panel = cv2.imread(str(panel_path), cv2.IMREAD_GRAYSCALE)
    assert panel is not None
    assert panel.shape[1] > panel.shape[0]
    left_mean = float(panel[:, : cell_width].mean())
    right_mean = float(panel[:, -cell_width:].mean())
    assert left_mean < right_mean


def test_panels_include_positive_controls_above_drug_strip(tmp_path: Path) -> None:
    rows, cols = 8, 12
    cell_height = 20
    cell_width = 20
    image = np.full((rows * cell_height, cols * cell_width, 3), 255, dtype=np.uint8)
    filtered_path = tmp_path / "plate-filtered.png"
    assert cv2.imwrite(str(filtered_path), image)

    plate = PlateMeasurement(
        tmp_path,
        categories={"ImageFileName": "plate-filtered"},
        plate_design="UKMYC5",
    )
    plate.load_image(".png")
    plate.initialize_plate_layout()
    assert plate.well_drug_name is not None
    assert plate.well_drug_conc is not None
    assert plate.image is not None

    target_drug = "RIF"
    control_positions = np.array(plate.well_positive_controls, dtype=int)

    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            plate.well_top_left[row, col] = (x1, y1)
            plate.well_bottom_right[row, col] = (x2, y2)
            plate.well_centre[row, col] = ((x1 + x2) // 2, (y1 + y2) // 2)
            plate.well_radii[row, col] = cell_width // 2
            intensity = 220
            if plate.well_drug_name[row, col] == target_drug:
                intensity = 100
            plate.image[y1:y2, x1:x2] = intensity

    for row, col in control_positions.tolist():
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        plate.image[y1:y2, x1:x2] = 40

    plate.image_name = "plate-segmented"
    plate.save_segment_arrays("-arrays.npz")
    assert cv2.imwrite(str(filtered_path), plate.image)

    main(["panels", str(filtered_path), "--plate_design", "UKMYC5"])

    panel_path = tmp_path / "plate-rif-panel.png"
    assert panel_path.exists()

    panel = cv2.imread(str(panel_path), cv2.IMREAD_GRAYSCALE)
    assert panel is not None
    assert panel.shape[1] > panel.shape[0]
    top_band_mean = float(panel[:cell_height, : 2 * cell_width].mean())
    bottom_band_mean = float(panel[-cell_height:, :cell_width].mean())
    assert top_band_mean < bottom_band_mean
    assert panel[0, 0] == 0
    assert panel[0, min(cell_width, panel.shape[1] - 1)] == 0
