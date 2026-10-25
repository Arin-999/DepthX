
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "validation.csv"

REQUIRED_COLUMNS = {
    "participant_id",
    "measurement",
    "reference_cm",
    "estimated_cm",
}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Create the validation CSV first: {CSV_PATH}"
        )

    grouped = defaultdict(list)

    with CSV_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain: {sorted(REQUIRED_COLUMNS)}"
            )

        for row in reader:
            reference = float(row["reference_cm"])
            estimated = float(row["estimated_cm"])

            if not math.isfinite(reference) or not math.isfinite(estimated):
                raise ValueError("Measurements must be finite.")

            error = estimated - reference

            grouped[row["measurement"]].append({
                "participant": row["participant_id"],
                "reference": reference,
                "estimated": estimated,
                "error": error,
                "absolute_error": abs(error),
            })

    for measurement, rows in sorted(grouped.items()):
        errors = [row["error"] for row in rows]
        absolute_errors = [row["absolute_error"] for row in rows]

        print(f"\nMeasurement: {measurement}")
        print(f"Valid comparisons: {len(rows)}")
        print(f"MAE (cm): {statistics.mean(absolute_errors):.2f}")
        print(f"Mean signed error (cm): {statistics.mean(errors):.2f}")
        print(f"Maximum absolute error (cm): {max(absolute_errors):.2f}")


if __name__ == "__main__":
    main()
