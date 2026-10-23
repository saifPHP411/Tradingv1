"""
Phase-1.8 dependency-compatibility regression tests.

Two independent forward-compatibility hazards, both triggered by the version
range a fresh (Railway/NIXPACKS) install resolves:

  1  ``pandas`` 3.x deprecates the ``inplace`` keyword (PDEP-8). ``yfinance``
     still calls it in ~29 places and declares only ``pandas>=1.3.0``, so an
     unpinned ``pandas`` resolves to 3.x and floods startup/market-data logs
     with ``Pandas4Warning``. requirements.txt caps ``pandas<3``.

  2  ``pd.Timedelta(minutes=N)`` builds the delta from a numpy *generic*
     timedelta, which numpy >= 2 deprecates (and will eventually raise). The
     engine's cooldown path used it. ``pd.Timedelta(N, "m")`` is the unitful
     form with the identical value and no warning.

These tests assert the installed/declared state, not a mock.
"""
import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


# ============================================================================
# 1) pandas cap
# ============================================================================

def _pandas_spec() -> str:
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if re.match(r"^pandas\b", line):
            return line
    raise AssertionError("requirements.txt declares no pandas requirement")


def test_requirements_pins_pandas_below_3():
    spec = _pandas_spec()
    assert "<3" in spec.replace(" ", ""), (
        f"pandas must be capped below 3 to avoid the PDEP-8 inplace warning "
        f"(got {spec!r})")


def test_installed_pandas_satisfies_the_cap():
    major = int(pd.__version__.split(".")[0])
    assert major < 3, (
        f"installed pandas {pd.__version__} is >= 3; yfinance's inplace= usage "
        f"will emit Pandas4Warning on every market-data fetch")


def test_inplace_keyword_is_warning_free_on_installed_pandas():
    """If pandas >= 3 were installed, ``inplace=`` would warn; the cap prevents
    that. Assert the operation stays silent on the installed version."""
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df.rename(columns={"a": "A"}, inplace=True)
    offenders = [w for w in caught
                 if "inplace" in str(w.message).lower()
                 or "Pandas4" in type(w.message).__name__]
    assert not offenders, [str(w.message) for w in offenders]
    assert "A" in df.columns


# ============================================================================
# 2) unitful Timedelta in the engine cooldown path
# ============================================================================

def test_unitful_timedelta_has_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        delta = pd.Timedelta(120, "m")
    # 7200s is the same instant, expressed without the deprecated keyword form.
    assert delta == pd.Timedelta(7200, "s")
    assert delta.total_seconds() == 7200.0
    assert not [w for w in caught
                if issubclass(w.category, DeprecationWarning)], \
        [str(w.message) for w in caught]


def test_engine_does_not_use_generic_keyword_timedelta():
    """The deprecated ``pd.Timedelta(minutes=...)`` form must not reappear."""
    source = (Path(__file__).resolve().parents[1] / "backtest" / "engine.py"
              ).read_text(encoding="utf-8")
    offenders = re.findall(r"pd\.Timedelta\((?:minutes|seconds|hours|days)=",
                           source)
    assert not offenders, (
        "engine.py uses the numpy-generic pd.Timedelta(minutes=...) form, which "
        "numpy >= 2 deprecates; use pd.Timedelta(n, 'm') instead")


@pytest.mark.parametrize("ts", ["2024-01-02 10:00:00", pd.Timestamp("2024-01-02 10:00:00")])
def test_cooldown_computation_is_warning_free(ts):
    """The exact engine expression, in both string and Timestamp form."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = (pd.Timestamp(ts) + pd.Timedelta(120, "m")).isoformat()
    assert out == "2024-01-02T12:00:00"
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
