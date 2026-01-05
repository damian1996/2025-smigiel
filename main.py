import argparse
from pathlib import Path
from typing import List

from binoculars import Binoculars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Binoculars model accuracy on a dataset"
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/train/data.tsv"),
        help="Path to input text data (.tsv)",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("data/train/labels.tsv"),
        help="Path to ground-truth labels (.tsv)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--model-name-1",
        type=str,
        default="speakleash/Bielik-11B-v2",
        help="Base model name",
    )
    parser.add_argument(
        "--model-name-2",
        type=str,
        default="speakleash/Bielik-11B-v2.3-Instruct",
        help="Instruct model name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.936,
        help="Decision threshold (currently informational)",
    )
    parser.add_argument(
        "--use-bfloat16",
        action="store_true",
        help="Enable bfloat16 precision",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="accuracy",
        choices=["accuracy", "low-fpr"],
        help="Binoculars inference mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample predictions",
    )

    return parser.parse_args()


def load_lines(path: Path, limit: int) -> List[str]:
    """Load up to `limit` lines from a text file."""
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            lines.append(line.rstrip("\n"))
            if i >= limit:
                break
    return lines


def label_to_int(label: str) -> int:
    """Convert label string to int (0=human, 1=AI)."""
    return 0 if label == "0" else 1


def pred_to_int(pred: str) -> int:
    """Convert model prediction to int (0=human, 1=AI)."""
    return 0 if pred.strip() == "human" else 1


def main() -> None:
    args = parse_args()

    texts = load_lines(args.data_path, args.n_samples)
    labels = load_lines(args.labels_path, args.n_samples)

    bino = Binoculars(
        args.model_name_1,
        args.model_name_2,
        use_bfloat16=args.use_bfloat16,
        mode=args.mode,
    )

    correct = 0

    for text, label in zip(texts, labels):
        pred_label, score, token_counts = bino.predict(text)

        pred = pred_to_int(pred_label)
        ground = label_to_int(label)
        score_value = round(float(score.tolist()), 3)

        if args.verbose:
            print(f"Predicted: {pred}")
            print(f"Ground:    {ground}")
            print(f"Score:     {score_value} | Tokens: {token_counts}\n")

        if pred == ground:
            correct += 1

    accuracy = correct / len(labels)
    print(f"ACC: {accuracy:.4f}")


if __name__ == "__main__":
    main()
