# FlexFL/src/function_call.py
from __future__ import annotations
from pathlib import Path
import json

# Optional: keep Levenshtein if installed; else fallback later if you want
import Levenshtein

# ---------- Path helpers (no circular import from pipeline) ----------
PKG_ROOT = Path(__file__).resolve().parents[1]               # .../FlexFL
PROJECT_ROOT = PKG_ROOT.parent                               # .../FlexFL_adapted
DATA_ROOT = PKG_ROOT / "data"
PREP_ROOT = PROJECT_ROOT / "prepare" / "buggy_program"
BUGGY_INPUT_ROOT = DATA_ROOT / "input" / "buggy_program"

def _buggy_base(dataset: str, bug: str = None) -> Path:
    """
    Find the base directory containing buggy program files for the given dataset (and optionally bug).
    Checks multiple locations and verifies files exist if bug is provided.
    Prefer data/input/buggy_program/<dataset>, else fallback to prepare/buggy_program/<dataset>
    or prepare/buggy_program/methods_buggy_<dataset>/.
    """
    # List of potential base directories to check (in priority order)
    candidates = [
        BUGGY_INPUT_ROOT / dataset,
        PREP_ROOT / dataset,
        PREP_ROOT / f"methods_buggy_{dataset}",
    ]
    
    # Add case variant for Defects4J
    if dataset == "Defects4J":
        candidates.append(PREP_ROOT / "methods_buggy_Defects4j")
    
    # If bug is provided, check if required files exist in each location
    if bug:
        required_file = f"{bug}.corpusMappingWithPackageSeparatorMethodLevelGranularity"
        for candidate in candidates:
            if candidate.exists():
                if (candidate / required_file).exists():
                    return candidate
    else:
        # If no bug specified, return first existing directory
        for candidate in candidates:
            if candidate.exists():
                return candidate
    
    # If bug was specified but not found, try again without bug check for directory structure
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Return the primary location even if it doesn't exist (callers will handle the error)
    return BUGGY_INPUT_ROOT / dataset

def _require_exists(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {what}: {p}\n"
            "Run the prepare step or copy/symlink the buggy_program artifacts."
        )

def _read_lines(p: Path):
    _require_exists(p, "file")
    with p.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f.readlines()]

# ---------- String matching utilities ----------
def split4search(query: str):
    if '(' not in query:
        querys = query.split('.')
    else:
        signature = query[query.find('(')+1:query.find(')')]
        method = query[:query.find('(')]
        # keep only simple type names for signature
        querys = method.split('.') + [e.strip().split('.')[-1] for e in signature.split(',') if e.strip()]
    return querys

def fuzzy_search(query: str, choices: list[str]) -> list[str]:
    query = query.replace('#','.')
    query = query.replace('$','.')
    match_res = []
    querys = split4search(query)

    # exact-component match pass
    for choice in choices:
        match_choice = split4search(choice)
        if all((q in match_choice) for q in querys):
            match_res.append(choice)

    if len(match_res) == 0:
        # normalize signature types to simple names
        if '(' in query:
            signature = query[query.find('(')+1:query.find(')')]
            query = query.split('(')[0]+'('+','.join([e.strip().split('.')[-1] for e in signature.split(',') if e.strip()])+')'
        distances = [(choice, Levenshtein.distance(query, choice)) for choice in choices]
        distances.sort(key=lambda x: x[1])
        for choice, dist in distances:
            if dist <= 5:
                match_res.append(choice)
            else:
                break
        if len(match_res) == 0:
            match_res = [choice for choice, _ in distances[:5]]
    return match_res

# ---------- File-driven helpers ----------
# Filenames used by the prepared artifacts
def _mapping_path(base: Path, bug: str) -> Path:
    return base / f"{bug}.corpusMappingWithPackageSeparatorMethodLevelGranularity"

def _raw_path(base: Path, bug: str) -> Path:
    return base / f"{bug}.corpusRawMethodLevelGranularity"

def get_code_snippet(bug: str, function: str, dataset: str) -> str:
    function = function.replace(', ',',').replace(' ,',',')
    base = _buggy_base(dataset, bug)

    mapping = _mapping_path(base, bug)
    raw     = _raw_path(base, bug)

    methods = _read_lines(mapping)          # lines like "path$pkg.Class.method(sig...)"
    codes   = _read_lines(raw)              # one raw method per line

    # normalize once
    norm_methods = [m.replace('$','.', 1).strip() for m in methods]

    # direct match
    for method, code in zip(norm_methods, codes):
        if method == function:
            return code

    # fuzzy fallback
    results = fuzzy_search(function, norm_methods)
    if len(results) == 1:
        method = results[0]
        # find its code
        for m, code in zip(norm_methods, codes):
            if m == method:
                return f"Do you mean `{method}`? Its code snippet is as follows.\n{code}"
        return f"Do you mean `{method}`? (code not found)"
    elif len(results) == 0:
        return "You provide a wrong method name. You can call `get_methods_of_class` first to get a right method name."
    else:
        return "You provide a wrong method name. Please try the following method names.\n" + '\n'.join(results)

def get_paths(bug: str, dataset: str) -> str:
    base = _buggy_base(dataset, bug)
    mapping = _mapping_path(base, bug)
    lines = _read_lines(mapping)
    # prefix before first '$' is a path
    paths = sorted(set(line.split('$', 1)[0] for line in lines if '$' in line))
    return '\n'.join(paths)

def get_classes(bug: str, path_name: str, dataset: str) -> str:
    base = _buggy_base(dataset, bug)
    mapping = _mapping_path(base, bug)
    lines = _read_lines(mapping)

    # classes from entries whose path matches path_name
    classes = []
    for e in lines:
        if '$' not in e:
            continue
        pfx, rest = e.split('$', 1)
        if pfx != path_name:
            continue
        # rest like pkg.Class.method(args)
        method_part = rest.strip().split('(')[0]
        cls = '.'.join(method_part.split('.')[:-1])
        if cls:
            classes.append(cls)

    classes = sorted(set(classes))
    if classes:
        return '\n'.join(classes)

    # fuzzy fallback on paths
    all_paths = sorted(set(line.split('$', 1)[0] for line in lines if '$' in line))
    results = fuzzy_search(path_name, all_paths)
    if len(results) == 1:
        return f"Do you mean `{results[0]}`? Its classes are as follows.\n{get_classes(bug, results[0], dataset)}"
    elif len(results) != 0:
        return "You provide a wrong path name. Please try the following path names.\n" + '\n'.join(results)
    else:
        return "You provide a wrong path name. You can call `get_paths` first to get a right path name."

def get_methods(bug: str, class_name: str, dataset: str) -> str:
    base = _buggy_base(dataset, bug)
    mapping = _mapping_path(base, bug)
    lines = _read_lines(mapping)

    methods = []
    # normalize mapping lines into fully qualified "pkg.Class.method(args)"
    for e in lines:
        if '$' not in e:
            continue
        _, rest = e.split('$', 1)
        rest = rest.replace('$','.', 1).strip()
        pos = rest.find('(')
        if pos == -1:
            continue
        cls = '.'.join(rest[:pos].split('.')[:-1])
        if cls == class_name:
            methods.append(rest[len(cls)+1:])  # only "method(args)"

    methods = sorted(set(methods))
    if methods:
        return '\n'.join(methods)

    # fuzzy fallback on classes
    all_classes = sorted(set('.'.join(e.strip().replace('$','.',1).split('(')[0].split('.')[:-1])
                             for e in lines if '$' in e and '(' in e))
    results = fuzzy_search(class_name, all_classes)
    if len(results) == 1:
        return f"Do you mean `{results[0]}`? Its methods are as follows.\n{get_methods(bug, results[0], dataset)}"
    elif len(results) != 0:
        return "You provide a wrong class name. Please try the following class names.\n" + '\n'.join(results)
    else:
        return "You provide a wrong class name. You can call `get_classes_of_path` first to get a right class name."

def find_class(bug: str, class_name: str, dataset: str) -> str:
    base = _buggy_base(dataset, bug)
    mapping = _mapping_path(base, bug)
    lines = _read_lines(mapping)

    classes = sorted(set('.'.join(e.strip().replace('$','.',1).split('(')[0].split('.')[:-1])
                         for e in lines if '$' in e and '(' in e))

    if '.' in class_name:
        find = fuzzy_search(class_name, classes)
    else:
        # try simple name exact matches first
        find = [c for c in classes if c.split('.')[-1] == class_name]
        if not find:
            simple_pool = sorted(set(c.split('.')[-1] for c in classes))
            fuzzy = fuzzy_search(class_name, simple_pool)
            if len(fuzzy) == 1:
                return f"Do you mean `{fuzzy[0]}`? Its result of fuzzy search is as follows.\n{find_class(bug, fuzzy[0], dataset)}"
            else:
                return f"Do not find `{class_name}` again because it is an invalid name. You can try the following names.\n" + '\n'.join(fuzzy)
    return '\n'.join(sorted(find))

def find_method(bug: str, method_name: str, dataset: str) -> str:
    base = _buggy_base(dataset, bug)
    mapping = _mapping_path(base, bug)
    lines = _read_lines(mapping)

    methods = sorted(set(e.strip().replace('$','.',1) for e in lines if '$' in e))
    results = fuzzy_search(method_name, methods)
    return '\n'.join(results)
