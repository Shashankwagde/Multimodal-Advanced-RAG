"""
Per-visitor workspaces for hosted demos.

Each visitor gets a folder under `workspaces/<id>/` with their own documents,
figures, index and image cache. A workspace that has had no activity for
WORKSPACE_IDLE_MINUTES is deleted, which is how uploads are cleared when a
visitor leaves (a browser cannot tell the server that a tab was closed, so
idle time is used instead).

Workspaces are used only when PER_SESSION_WORKSPACES=true. On your own
computer, leave it unset and the project's `data/` and `vectorstore/` folders
are used as before.
"""

import logging
import re
import shutil
import time
import uuid
from pathlib import Path

import config

logger = logging.getLogger(__name__)

WORKSPACES_DIR = config.PROJECT_ROOT / "workspaces"
LAST_SEEN_FILE = "last_seen.txt"

# Workspace ids are generated as 32 hex characters. Anything else is rejected,
# so a crafted id cannot point outside the workspaces folder.
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def create_workspace() -> str:
    """
    Create a new, empty workspace for a visitor.

    Returns:
        The new workspace id.
    """
    workspace_id = uuid.uuid4().hex
    root = WORKSPACES_DIR / workspace_id
    (root / "data" / "documents").mkdir(parents=True, exist_ok=True)
    (root / "data" / "extracted_images").mkdir(parents=True, exist_ok=True)
    (root / "vectorstore").mkdir(parents=True, exist_ok=True)
    touch_workspace(workspace_id)
    return workspace_id


def workspace_root(workspace_id: str) -> Path:
    """
    Return the folder for a workspace id.

    Args:
        workspace_id: An id returned by `create_workspace`.

    Returns:
        The workspace folder.

    Raises:
        ValueError: If the id is not a valid workspace id.
    """
    if not _ID_PATTERN.match(workspace_id or ""):
        raise ValueError("Invalid workspace id.")
    return WORKSPACES_DIR / workspace_id


def touch_workspace(workspace_id: str) -> None:
    """
    Record that a visitor is active now, so their workspace is not cleaned up.

    Args:
        workspace_id: The workspace to mark as used.
    """
    root = workspace_root(workspace_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / LAST_SEEN_FILE).write_text(str(time.time()), encoding="utf-8")


def activate_workspace(workspace_id: str) -> None:
    """
    Make this workspace the one used by the current run, and mark it active.

    Args:
        workspace_id: The visitor's workspace id.
    """
    config.set_workspace(workspace_root(workspace_id))
    touch_workspace(workspace_id)


def cleanup_idle_workspaces() -> int:
    """
    Delete every workspace that has been idle for too long.

    Returns:
        The number of workspaces deleted.
    """
    if not WORKSPACES_DIR.exists():
        return 0

    limit_seconds = config.WORKSPACE_IDLE_MINUTES * 60
    now = time.time()
    deleted = 0

    for folder in WORKSPACES_DIR.iterdir():
        if not folder.is_dir() or not _ID_PATTERN.match(folder.name):
            continue

        last_seen_path = folder / LAST_SEEN_FILE
        try:
            last_seen = float(last_seen_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            last_seen = folder.stat().st_mtime

        if now - last_seen > limit_seconds:
            shutil.rmtree(folder, ignore_errors=True)
            deleted += 1

    if deleted:
        logger.info("Removed %d idle workspace(s).", deleted)
    return deleted
