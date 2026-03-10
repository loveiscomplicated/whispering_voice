"""pytest configuration: make numerically-prefixed src modules importable.

Python's ``import`` statement does not allow identifiers that start with a
digit (e.g. ``import src.1_download_youtube`` is a SyntaxError).  We work
around this in tests by using :func:`importlib.import_module` to load each
stage module and then injecting it into ``sys.modules`` under an underscore-
prefixed alias.

After this conftest is loaded, test files can write::

    from src._1_download_youtube import YouTubeDownloader
    from src._2_quality_validation_1 import QualityValidator1
    # …
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src.*`` imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Mapping: alias (valid Python identifier) → actual module name
_STAGE_ALIASES: dict[str, str] = {
    "src._1_download_youtube":          "src.1_download_youtube",
    "src._2_quality_validation_1":      "src.2_quality_validation_1",
    "src._3_run_stt_and_vad":           "src.3_run_stt_and_vad",
    "src._4_synthesize_noise":          "src.4_synthesize_noise",
    "src._5_quality_validation_2":      "src.5_quality_validation_2",
    "src._6_generate_finetuning_dataset": "src.6_generate_finetuning_dataset",
}

for _alias, _real in _STAGE_ALIASES.items():
    if _alias not in sys.modules:
        try:
            mod = importlib.import_module(_real)
            sys.modules[_alias] = mod
        except ModuleNotFoundError:
            # Module not yet implemented — skip silently so unrelated tests
            # can still run.
            pass
