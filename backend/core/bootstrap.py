import os
import sys
from pathlib import Path

from dotenv import load_dotenv

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent

if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

load_dotenv(dotenv_path=_ROOT / ".env")
