"""Put this checkout's src/ first on sys.path, so the tests always exercise the sources next to
them (the mutation check relies on this when it runs the suite on a mutated copy)."""

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
for name in [m for m in sys.modules if m == "mre" or m.startswith("mre.")]:
    del sys.modules[name]
