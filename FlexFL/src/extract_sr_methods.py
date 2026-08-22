"""
Extract top-5 suspicious methods from SR stage JSON outputs and write them
to the suspicious_methods input directory for the LR stage.

Usage:
    python extract_sr_methods.py --model Gemma4 --dataset Defects4J
"""
import json
import os
import argparse
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
PKG_ROOT = SRC_ROOT.parent
DATA_ROOT = PKG_ROOT / "data"
RES_ROOT = PKG_ROOT / "res"


def extract_methods(content: str) -> list[str]:
    methods = []
    for line in content.split("\n"):
        for i in range(1, 6):
            for pat in [f"Top_{i} : ", f"Top_{i}: ", f"Top {i}: ", f"Top {i} : "]:
                if pat in line:
                    methods.append(line.split(pat, 1)[1].strip())
                    break
    return methods


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Gemma4")
    parser.add_argument("--dataset", default="Defects4J")
    parser.add_argument("--sr-dir", default=None,
                        help="SR results dir (default: res/<model>_<dataset>_SR_br)")
    parser.add_argument("--rank", default="All",
                        help="Rank label used for the suspicious_methods subfolder")
    args = parser.parse_args()

    sr_dir = Path(args.sr_dir) if args.sr_dir else RES_ROOT / f"{args.model}_{args.dataset}_SR_br"
    out_dir = DATA_ROOT / "input" / "suspicious_methods" / args.dataset / f"{args.model}_{args.rank}"

    out_dir.mkdir(parents=True, exist_ok=True)

    bug_list_path = DATA_ROOT / "bug_list" / args.dataset / "bug_list.txt"
    bugs = [b.strip() for b in bug_list_path.read_text().splitlines() if b.strip()]

    ok, missing, empty = 0, 0, 0
    for bug in bugs:
        sr_path = sr_dir / f"{bug}.json"
        if not sr_path.exists():
            missing += 1
            continue
        result = json.loads(sr_path.read_text())
        last_content = result[-1]["content"] if result else ""
        methods = extract_methods(last_content)
        if not methods:
            empty += 1
        (out_dir / f"{bug}.txt").write_text("\n".join(methods))
        ok += 1

    print(f"Done: {ok} written, {missing} SR results missing, {empty} with no methods parsed")
    print(f"Output: {out_dir}")
