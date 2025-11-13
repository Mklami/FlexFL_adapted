import argparse
import json
import shutil
from csv import DictReader
from pathlib import Path

from function_call import get_code_snippet


def _find_sr_dir(res_root: Path, dataset: str, model: str) -> Path:
    """
    Locate the directory containing SR outputs for the given dataset/model.
    We try a few naming variants to accommodate inputs such as --input bug_report.
    """
    candidates = []
    model_variants = [model]
    if '_' in model:
        # e.g. model=Llama3_br -> fallback to Llama3
        model_variants.append(model.split('_')[0])

    suffixes = ["_SR", "_SR_br", "_SR_tt"]
    for mv in model_variants:
        for suff in suffixes:
            candidates.append(res_root / f"{mv}_{dataset}{suff}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Cannot locate SR results directory. Looked for:\n"
        f"{searched}\n"
        "Ensure the SR stage has been executed and specify --model/--rank consistently."
    )


def _ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Defects4J')
    parser.add_argument('--model', default='Llama3')
    parser.add_argument('--rank', default='All')
    parser.add_argument('--bug-list', default=None,
                        help='Optional path to a custom bug list (default: data/bug_list/<dataset>/bug_list.txt)')
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    rank = args.rank

    SRC_ROOT = Path(__file__).resolve().parent
    PROJ_ROOT = SRC_ROOT.parent
    data_root = PROJ_ROOT / "data"
    res_root = PROJ_ROOT / "res"
    output_dir = data_root / "input" / "suspicious_methods" / dataset / f"{model}_{rank}"

    sr_dir = _find_sr_dir(res_root, dataset, model)

    _ensure_clean_dir(output_dir)

    if args.bug_list:
        bug_list_path = Path(args.bug_list)
    else:
        bug_list_path = data_root / "bug_list" / dataset / "bug_list.txt"
    with bug_list_path.open("r", encoding="utf-8") as f:
        bugs = [e.strip() for e in f.readlines() if e.strip()]

    for bug in bugs:
        suspicious_methods = []
        try:
            fl = 'SBIR'
            with (data_root / "FL_results" / fl / dataset / f"{bug}_method-susps.csv").open("r", encoding="utf-8") as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:5]
            fl = 'Ochiai'
            with (data_root / "FL_results" / fl / dataset / f"{bug}_method-susps.csv").open("r", encoding="utf-8") as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:5 * 2]
            fl = 'BoostN'
            with (data_root / "FL_results" / fl / dataset / f"{bug}_method-susps.csv").open("r", encoding="utf-8") as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:5 * 3]
        except Exception:
            if dataset == 'Defects4J':
                fl = 'Ochiai'
            with (data_root / "FL_results" / fl / dataset / f"{bug}_method-susps.csv").open("r", encoding="utf-8") as f:
                reader = DictReader(f)
                for row in reader:
                    method_name = row['File'] + '.' + row['Signature']
                    suspicious_methods.append(method_name)
            suspicious_methods = suspicious_methods[:5 * 3]

        with (sr_dir / f"{bug}.json").open("r", encoding="utf-8") as f:
            res = json.load(f)
            res = res[-1]['content']
            for line in res.split('\n'):
                for i in range(1, 6):
                    if f'Top_{i} : ' in line:
                        method = line.split(f'Top_{i} : ')[1].strip()
                        method = method.replace(', ', ',')
                        method = method.replace(' ,', ',')
                        code = get_code_snippet(bug, method, dataset)
                        if code.startswith('Do you'):
                            method = code.split('`')[1]
                        elif code.startswith('You') and '\n' in code:
                            method = code.split('\n')[1]
                        suspicious_methods.append(method)

        with (output_dir / f"{bug}.txt").open("w", encoding="utf-8") as f:
            f.write('\n'.join(suspicious_methods))