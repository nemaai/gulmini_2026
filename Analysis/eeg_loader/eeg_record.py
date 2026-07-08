from dataclasses import dataclass
from typing import Any


@dataclass
class EEGRecord:

    source_file: str

    format: str

    sfreq: float | None

    channels: list

    data: Any

    dataset: str = "UNKNOWN"

    metadata: dict | None = None