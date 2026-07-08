#!/usr/bin/env python3
"""Apply GEPA-optimized prompt slices into constraint_validate system.minijinja."""
import re
import pathlib
import sys

ROOT = pathlib.Path("eval/tensorzero/config/constraint_validate/system.minijinja")
OUT = pathlib.Path("gepa-constraint-out")
PAIRS = [
    ("GEPA:RESEARCH", OUT / "constraint_research.prompt.txt"),
    ("GEPA:MATERIALIZE", OUT / "constraint_materialize.prompt.txt"),
    ("GEPA:COMPILE_FIX", OUT / "constraint_compile_fix.prompt.txt"),
]


def main() -> int:
    text = ROOT.read_text(encoding="utf-8")
    for tag, prompt_path in PAIRS:
        body = prompt_path.read_text(encoding="utf-8").strip()
        pattern = rf"\{{# {tag}_START #\}}.*?\{{# {tag}_END #\}}"
        repl = f"{{# {tag}_START #}}\n{body}\n{{# {tag}_END #}}"
        text, n = re.subn(pattern, repl, text, count=1, flags=re.S)
        if n != 1:
            print(f"anchor {tag} not found or ambiguous", file=sys.stderr)
            return 1
    ROOT.write_text(text, encoding="utf-8")
    print("applied 3 GEPA slices to", ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
