from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.five_finger_model import HAND_PREFIX, build_five_finger_model

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
OPPOSED_THUMB_YAW = 1.308
TARGET_DIAMETER_M = 0.071
MAX_HEIGHT_OFFSET_M = 0.06


def _rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array(((np.cos(angle), 0.0, np.sin(angle)), (0.0, 1.0, 0.0), (-np.sin(angle), 0.0, np.cos(angle))))


def _set_hand_closure(model: mujoco.MjModel, data: mujoco.MjData, fraction: float) -> None:
    for actuator_id in range(model.nu):
        name = model.actuator(actuator_id).name or ""
        if not name.startswith(f"{HAND_PREFIX}right_"):
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        lower, upper = model.actuator_ctrlrange[actuator_id]
        value = OPPOSED_THUMB_YAW if name.endswith("thumb_yaw") else lower + fraction * (upper - lower)
        data.qpos[model.jnt_qposadr[joint_id]] = np.clip(value, *model.jnt_range[joint_id])

    for equality_id in range(model.neq):
        if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        driven_id = int(model.eq_obj1id[equality_id])
        source_id = int(model.eq_obj2id[equality_id])
        if not (model.joint(driven_id).name or "").startswith(f"{HAND_PREFIX}right_"):
            continue
        source = data.qpos[model.jnt_qposadr[source_id]]
        coefficients = model.eq_data[equality_id, :5]
        value = sum(float(coefficient) * source**power for power, coefficient in enumerate(coefficients))
        data.qpos[model.jnt_qposadr[driven_id]] = np.clip(value, *model.jnt_range[driven_id])


def _tip_positions_in_wrist(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    wrist = model.site("right_ee_control_point").id
    origin = data.site_xpos[wrist]
    rotation = data.site_xmat[wrist].reshape(3, 3)
    tips = [data.site_xpos[model.site(f"{HAND_PREFIX}right_right_{finger}_tip").id] for finger in FINGERS]
    return np.asarray([(rotation.T @ (tip - origin)) for tip in tips])


def measure_sweep() -> dict:
    model = build_five_finger_model()
    data = mujoco.MjData(model)
    samples, summary = [], []
    closure_fractions = np.linspace(0.0, 1.0, 21)

    for tilt_degrees in range(0, 91, 15):
        tilt = _rotation_y(tilt_degrees)
        tilt_samples = []
        for fraction in closure_fractions:
            _set_hand_closure(model, data, float(fraction))
            mujoco.mj_forward(model, data)
            tips_wrist = _tip_positions_in_wrist(model, data)
            thumb, fingers = tilt @ tips_wrist[0], tilt @ np.mean(tips_wrist[1:], axis=0)
            jaw_vector = thumb - fingers
            aperture = float(np.linalg.norm(jaw_vector))
            sample = {
                "tilt_degrees": tilt_degrees,
                "closure_fraction": round(float(fraction), 2),
                "aperture_m": aperture,
                "height_offset_m": abs(float(thumb[2] - fingers[2])),
                "jaw_axis": (jaw_vector / aperture).tolist(),
                "jaw_midpoint_from_wrist_m": (0.5 * (thumb + fingers)).tolist(),
                "tips_wrist_m": tips_wrist.tolist(),
            }
            samples.append(sample)
            tilt_samples.append(sample)

        nearest = min(tilt_samples, key=lambda item: abs(item["aperture_m"] - TARGET_DIAMETER_M))
        apertures = [item["aperture_m"] for item in tilt_samples]
        summary.append(
            {
                "tilt_degrees": tilt_degrees,
                "min_aperture_m": min(apertures),
                "max_aperture_m": max(apertures),
                "target_closure_fraction": nearest["closure_fraction"],
                "target_aperture_m": nearest["aperture_m"],
                "height_offset_m": nearest["height_offset_m"],
                "jaw_axis": nearest["jaw_axis"],
                "wrist_offset_for_object_center_m": (-np.asarray(nearest["jaw_midpoint_from_wrist_m"])).tolist(),
                "suitable": min(apertures) <= TARGET_DIAMETER_M <= max(apertures)
                and nearest["height_offset_m"] < MAX_HEIGHT_OFFSET_M,
            }
        )
    return {"target_diameter_m": TARGET_DIAMETER_M, "summary": summary, "samples": samples}


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=report["summary"][0].keys())
        writer.writeheader()
        writer.writerows(report["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Inspire Hand fingertip closure sweep using forward kinematics")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hand_sweep.json"))
    args = parser.parse_args()
    report = measure_sweep()
    write_report(report, args.output)
    for row in report["summary"]:
        print(
            f"{row['tilt_degrees']:>2} deg | aperture {row['min_aperture_m']*100:.1f}-"
            f"{row['max_aperture_m']*100:.1f} cm | target height offset "
            f"{row['height_offset_m']*100:.1f} cm | {'PASS' if row['suitable'] else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
