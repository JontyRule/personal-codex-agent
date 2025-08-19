from __future__ import annotations

import os
import glob
import yaml
from typing import Dict, List, Tuple

from .persona import Profile


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_profile(profile_path: str | None = None) -> Profile:
    path = profile_path or os.path.join(DATA_DIR, "profile.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Profile(**data)


def list_markdown_files() -> List[str]:
    pattern = os.path.join(DATA_DIR, "*.md")
    return sorted(glob.glob(pattern))


def load_markdown_files() -> List[Tuple[str, str]]:
    files = list_markdown_files()
    contents: List[Tuple[str, str]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                contents.append((fp, f.read()))
        except Exception as e:
            print(f"Failed to read {fp}: {e}")
    return contents
