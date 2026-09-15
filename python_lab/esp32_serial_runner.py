"""Send arithmetic benchmark requests to the ESP32 serial backend."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data" / "esp32"
SUITE_OPERATIONS = ["empty_loop", "add", "subtract", "multiply", "divide"]
SUITE_NUMERIC_TYPES = ["float32", "float64"]


def parse_esp32_json(line: str) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        repaired = re.sub(r'("result_checksum"\s*:\s*)(ovf|inf|-inf|nan)(\s*[,}])', r'\1"\2"\3', line)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--operation", default="divide")
    parser.add_argument("--numeric-type", default="float32")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--passes", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--suite", action="store_true", help="Run every operation/type pair. This is the default unless --profile or --single is used.")
    parser.add_argument("--single", action="store_true", help="Run only --operation with --numeric-type.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_request(args: argparse.Namespace, is_suite_request: bool) -> dict:
    if args.profile:
        return {"op": "profile"}
    if is_suite_request:
        return {
            "op": "suite",
            "samples": args.samples,
            "passes": args.passes,
            "seed": args.seed,
        }
    return {
        "op": args.operation,
        "numeric_type": args.numeric_type,
        "samples": args.samples,
        "passes": args.passes,
        "seed": args.seed,
    }


def main() -> int:
    try:
        import serial
    except ModuleNotFoundError:
        print("Missing dependency: pyserial")
        print("Install with: python -m pip install pyserial")
        return 1

    args = parse_args()
    if args.trials < 1:
        print("--trials must be at least 1")
        return 1

    is_suite_request = args.suite or not args.single

    if args.profile:
        is_suite_request = False
        args.trials = 1

    request = build_request(args, is_suite_request)

    rows: list[dict] = []
    run_id = str(uuid.uuid4())
    with serial.Serial(args.port, args.baud, timeout=0.25) as device:
        time.sleep(2.0)
        while device.in_waiting:
            line = device.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(f"ESP32: {line}")

        for trial_index in range(args.trials):
            encoded = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
            trial_label = f" trial {trial_index + 1}/{args.trials}" if args.trials > 1 else ""
            print(f"Sending{trial_label}: {encoded.decode().strip()}")
            device.write(encoded)
            device.flush()

            deadline = time.time() + args.timeout
            completed_trial = False
            while time.time() < deadline:
                line = device.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                print(f"ESP32: {line}")
                response = parse_esp32_json(line)
                if response is None:
                    continue
                response["host_timestamp"] = datetime.now(timezone.utc).isoformat()
                response["run_id"] = run_id
                response["trial_index"] = trial_index
                if response.get("status") == "ok" and response.get("op") in SUITE_OPERATIONS:
                    rows.append(response)
                if response.get("status") == "suite_end":
                    completed_trial = True
                    break
                if not is_suite_request and response.get("status") in {"ok", "error"}:
                    completed_trial = True
                    break
            if not completed_trial:
                print(f"Timed out waiting for ESP32 response during trial {trial_index + 1}.")
                return 1

        if rows:
            if is_suite_request:
                output = args.out or DATA_DIR / f"{run_id}_suite.csv"
                write_csv(output, rows)
                print(f"Wrote suite CSV: {output}")
            else:
                output = args.out or DATA_DIR / f"{run_id}_{rows[-1].get('op', 'result')}.csv"
                write_csv(output, rows)
                print(f"Wrote CSV: {output}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
