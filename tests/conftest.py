import sys
from pathlib import Path

# Add the polyga package directory to sys.path so imports like "from polyga.xxx"
# work during tests.
_polyga_path = Path(__file__).resolve().parent.parent / "src" / "models"
if str(_polyga_path) not in sys.path:
    sys.path.insert(0, str(_polyga_path))
