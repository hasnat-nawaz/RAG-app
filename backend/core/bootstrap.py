import os
import sys
from pathlib import Path

from dotenv import load_dotenv

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

_CORE = Path(__file__).resolve().parent
_BACKEND = _CORE.parent
_ROOT = _BACKEND.parent

for path in (_BACKEND, _CORE):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

load_dotenv(dotenv_path=_ROOT / ".env")
