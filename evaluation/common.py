"""Command-line plumbing shared by the evaluation scripts.

Every script takes `--data` (the data root, laid out as in the original project),
`--out` (a JSON file) and an optional `--id-map` (JSON from real video id to an
anonymous label). The map is applied to every key and string value in the output
that equals a real id, so published tables can use V01, V02, ….
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / 'data'


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA, help='data root (default: data/ inside the repository root)')
    ap.add_argument('--out', type=Path, required=True, help='output JSON path')
    ap.add_argument('--id-map', type=Path, default=None, help='JSON object from real video id to anonymous label')
    return ap


def load_id_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with open(path) as fh:
        return dict(json.load(fh))


def anonymize(obj: Any, id_map: Mapping[str, str]) -> Any:
    """Replace every dict key and string equal to a real id with its label."""
    if not id_map:
        return obj
    if isinstance(obj, dict):
        return {id_map.get(k, k) if isinstance(k, str) else k: anonymize(v, id_map) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [anonymize(v, id_map) for v in obj]
    if isinstance(obj, str):
        return id_map.get(obj, obj)
    return obj


def write_json(obj: Any, path: Path, id_map: Optional[Mapping[str, str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(anonymize(obj, id_map or {}), fh, indent=1)
        fh.write('\n')


def manifest_groups(data: Path) -> Dict[str, str]:
    """Video id to lighting group, from the second column of footage/manifest.tsv."""
    with open(Path(data) / 'footage' / 'manifest.tsv') as fh:
        return {row[0]: row[1] for row in csv.reader(fh, delimiter='\t') if len(row) >= 2}


def load_groups(data: Path, path: Optional[Path] = None) -> Dict[str, str]:
    """Video id to group: a JSON object or two-column TSV if given, else the manifest."""
    if path is None:
        return manifest_groups(data)
    path = Path(path)
    if path.suffix == '.json':
        with open(path) as fh:
            return dict(json.load(fh))
    with open(path) as fh:
        return {row[0]: row[1] for row in csv.reader(fh, delimiter='\t') if len(row) >= 2}
