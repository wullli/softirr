
from __future__ import annotations

import csv

import pytest

from soft_irr.experiments.load_annotations import (
    _derm1_wanted_members,
    load_reflacx_annotation_rows,
)

_PHASE_LABELS = {1: "Emphysema", 2: "High lung volume / emphysema"}


def write_reflacx_phase(reflacx_dir, phase: int, readings: list[tuple[str, str, str]]) -> None:
    main_data = reflacx_dir / "main_data"
    main_data.mkdir(parents=True, exist_ok=True)
    label = _PHASE_LABELS[phase]
    with open(main_data / f"metadata_phase_{phase}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "dicom_id", "eye_tracking_data_discarded", label])
        writer.writeheader()
        for reading_id, dicom_id, transcription in readings:
            writer.writerow({
                "id": reading_id, "dicom_id": dicom_id,
                "eye_tracking_data_discarded": "False", label: "3",
            })
            (main_data / reading_id).mkdir(parents=True, exist_ok=True)
            (main_data / reading_id / "transcription.txt").write_text(transcription, encoding="utf-8")


def test_reflacx_disjoint_phases_merge(tmp_path):
    write_reflacx_phase(tmp_path, 1, [("r1", "xrayA", "left base opacity"), ("r2", "xrayA", "basal opacity")])
    write_reflacx_phase(tmp_path, 2, [("r3", "xrayB", "clear lungs"), ("r4", "xrayB", "no acute findings")])

    rows_by_dicom = load_reflacx_annotation_rows(reflacx_dir=tmp_path)

    assert set(rows_by_dicom) == {"xrayA", "xrayB"}
    assert len(rows_by_dicom["xrayA"]) == 2
    assert {r[2] for r in rows_by_dicom["xrayB"]} == {"r3", "r4"}


def test_reflacx_overlapping_dicom_id_across_phases_raises(tmp_path):
    write_reflacx_phase(tmp_path, 1, [("r1", "shared", "left base opacity"), ("r2", "shared", "basal opacity")])
    write_reflacx_phase(tmp_path, 2, [("r3", "shared", "clear lungs"), ("r4", "shared", "no findings")])

    with pytest.raises(ValueError, match="shared"):
        load_reflacx_annotation_rows(reflacx_dir=tmp_path)


def test_reflacx_drops_single_reading_xrays(tmp_path):
    write_reflacx_phase(tmp_path, 1, [("r1", "xrayA", "only reading"), ("r2", "xrayB", "a"), ("r3", "xrayB", "b")])
    assert set(load_reflacx_annotation_rows(reflacx_dir=tmp_path, phases=(1,))) == {"xrayB"}


def test_derm1_wanted_members_maps_both_full_path_and_stripped_name():
    wanted = _derm1_wanted_members(["src/a.jpg", "src/b.jpg"], hash_cache={})
    assert wanted == {"src/a.jpg": "src/a.jpg", "a.jpg": "src/a.jpg",
                      "src/b.jpg": "src/b.jpg", "b.jpg": "src/b.jpg"}


def test_derm1_wanted_members_never_shadows_a_full_path_with_a_stripped_name():
    wanted = _derm1_wanted_members(["src/src/b.jpg", "src/b.jpg"], hash_cache={})
    assert wanted["src/b.jpg"] == "src/b.jpg"
    assert wanted["src/src/b.jpg"] == "src/src/b.jpg"


def test_derm1_wanted_members_skips_already_hashed():
    wanted = _derm1_wanted_members(["src/a.jpg", "src/b.jpg"], hash_cache={"src/a.jpg": "ff00"})
    assert wanted == {"src/b.jpg": "src/b.jpg", "b.jpg": "src/b.jpg"}
