"""Command-line interface for evaluating Hebrew text readability."""

import argparse
import json

from inference.engine import ReadabilityInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Score Hebrew text readability from the CLI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw Hebrew paragraph string to evaluate.")
    group.add_argument("--file", type=str, help="Path to a UTF-8 .txt file to evaluate.")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Optional local artifact dir.")
    parser.add_argument("--json", action="store_true", help="Output raw JSON payload.")
    args = parser.parse_args()

    engine = ReadabilityInferenceEngine(artifact_dir=args.artifact_dir)

    if args.text:
        result = engine.evaluate_text(args.text, print_report=not args.json)
    else:
        result = engine.evaluate_txt_file(args.file, print_report=not args.json)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()