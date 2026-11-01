"""
Finding image files on any machine.

The index stores each figure's `image_path` as the absolute path from the
machine that built it (for example D:\\Projects\\... on Windows). That path
does not exist on a server, so the app looks the file up by name in the
project folders instead.
"""

from pathlib import Path
from typing import Optional

import config


def resolve_image_path(stored_path: str | None) -> Optional[str]:
    """
    Return a path that exists on this machine for a stored image path.

    Args:
        stored_path: The `image_path` from the index metadata. May be a path
            from another computer.

    Returns:
        The first existing path among:
            1. the stored path itself,
            2. data/extracted_images/<filename>,
            3. data/documents/<filename>,
        or None if none of them exist.
    """
    if not stored_path:
        return None

    filename = Path(stored_path.replace("\\", "/")).name
    candidates = [
        Path(stored_path),
        config.EXTRACTED_IMAGES_DIR / filename,
        config.DOCUMENTS_DIR / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
