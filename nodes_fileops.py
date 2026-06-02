"""FoleyTune input-file management — rename / delete videos in the ComfyUI input dir.

Provides a small companion node (``FoleyTuneVideoFileManager``) that wires into the
optional ``file_manager`` input of ``FoleyTuneVideoLoaderUpload``. The rename / delete
buttons live on the JS side (``web/js/FoleyTuneVideo.js``) and call the two API routes
registered here, acting on the video currently selected on the connected loader.

The routes operate strictly inside the ComfyUI *input* directory: every target path is
resolved and checked against the input dir with ``os.path.commonpath`` to block path
traversal. Deletes are recoverable — files are sent to the system trash (``send2trash``
if installed) or moved into an ``input/.foleytune_trash`` folder, never hard-removed.
"""

import os
import logging

logger = logging.getLogger(__name__)

import folder_paths

_TRASH_DIRNAME = ".foleytune_trash"


def _resolve_input_file(name):
    """Resolve a combo filename to (input_dir, abs_path) inside the input dir.

    Returns (input_dir, None) if the name escapes the input directory.
    """
    if not name:
        return None, None
    # Strip any ComfyUI annotation like "clip.mp4 [input]".
    fname = name.split(" [", 1)[0] if name.endswith("]") and " [" in name else name
    input_dir = os.path.abspath(folder_paths.get_input_directory())
    abs_path = os.path.abspath(os.path.join(input_dir, fname))
    if os.path.commonpath((input_dir, abs_path)) != input_dir:
        return input_dir, None
    return input_dir, abs_path


def _free_path(path):
    """Return ``path`` if free, else append ' (n)' before the extension until free."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base} ({i}){ext}"):
        i += 1
    return f"{base} ({i}){ext}"


def _rename_input_file(old_name, new_name):
    """Rename a file in the input dir, keeping its original extension.

    Returns (final_name, None) on success or (None, error_message) on failure.
    """
    input_dir, old_path = _resolve_input_file(old_name)
    if old_path is None:
        return None, f"unsafe path: {old_name}"
    if not os.path.isfile(old_path):
        return None, f"file not found: {old_name}"

    ext = os.path.splitext(old_path)[1]
    # The new name is a base name; ignore any directory and any extension the user typed.
    safe_base = os.path.splitext(os.path.basename(new_name.strip()))[0]
    if not safe_base:
        return None, "invalid new name"

    new_full = safe_base + ext
    _, new_path = _resolve_input_file(new_full)
    if new_path is None:
        return None, f"unsafe target name: {new_full}"
    if os.path.abspath(new_path) == os.path.abspath(old_path):
        return new_full, None  # no-op rename
    if os.path.exists(new_path):
        return None, f"already exists: {new_full}"

    os.rename(old_path, new_path)
    logger.info(f"FoleyTune: renamed input '{old_name}' -> '{new_full}'")
    return new_full, None


def _delete_input_file(name):
    """Send an input-dir file to trash (recoverable).

    Returns (location, None) on success or (None, error_message) on failure, where
    ``location`` describes where the file went ("system trash" or the trash folder path).
    """
    input_dir, path = _resolve_input_file(name)
    if path is None:
        return None, f"unsafe path: {name}"
    if not os.path.isfile(path):
        return None, f"file not found: {name}"

    # Prefer the OS trash if send2trash is available.
    try:
        from send2trash import send2trash
        send2trash(path)
        logger.info(f"FoleyTune: sent input '{name}' to system trash")
        return "system trash", None
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"FoleyTune: send2trash failed for '{name}' ({e}); using trash folder")

    trash_dir = os.path.join(input_dir, _TRASH_DIRNAME)
    os.makedirs(trash_dir, exist_ok=True)
    dest = _free_path(os.path.join(trash_dir, os.path.basename(path)))
    os.rename(path, dest)
    logger.info(f"FoleyTune: moved input '{name}' to {dest}")
    return os.path.join(_TRASH_DIRNAME, os.path.basename(dest)), None


# ---------------------------------------------------------------------------------------
# API routes — registered only inside a running ComfyUI server (no-op under tests).
# ---------------------------------------------------------------------------------------
try:
    from server import PromptServer
    from aiohttp import web

    _server = PromptServer.instance
except Exception:
    _server = None

if _server is not None:

    @_server.routes.post("/foleytune/rename_input")
    async def _foleytune_rename_input(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        old = data.get("name")
        new = data.get("new_name")
        if not old or not (new or "").strip():
            return web.json_response({"error": "missing 'name' or 'new_name'"}, status=400)
        final_name, err = _rename_input_file(old, new)
        if err is not None:
            status = 404 if err.startswith("file not found") else (
                409 if err.startswith("already exists") else 400)
            return web.json_response({"error": err}, status=status)
        return web.json_response({"name": final_name})

    @_server.routes.post("/foleytune/delete_input")
    async def _foleytune_delete_input(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        name = data.get("name")
        if not name:
            return web.json_response({"error": "missing 'name'"}, status=400)
        location, err = _delete_input_file(name)
        if err is not None:
            status = 404 if err.startswith("file not found") else (
                400 if err.startswith("unsafe") else 500)
            return web.json_response({"error": err}, status=status)
        return web.json_response({"name": name, "location": location})


# ---------------------------------------------------------------------------------------
# NODE: FoleyTune Video File Manager — companion for the upload loader
# ---------------------------------------------------------------------------------------

class FoleyTuneVideoFileManager:
    """Rename / delete the video selected on a connected FoleyTune Video Loader (Upload).

    Wire this node's ``file_manager`` output into the loader's optional ``file_manager``
    input. The rename field (an autocomplete input suggesting existing file names) + button
    and the delete button are added in JS (``web/js/FoleyTuneVideo.js``) and act on the
    loader's currently selected ``video``, calling the server routes in this module.
    The node itself is a no-op at graph-execution time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # The rename field and buttons are JS-only DOM widgets; no server-side inputs.
        return {"required": {}}

    RETURN_TYPES = ("FOLEYTUNE_FILEMGR",)
    RETURN_NAMES = ("file_manager",)
    FUNCTION = "noop"
    CATEGORY = "FoleyTune"

    def noop(self):
        return (None,)


NODE_CLASS_MAPPINGS = {
    "FoleyTuneVideoFileManager": FoleyTuneVideoFileManager,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyTuneVideoFileManager": "FoleyTune Video File Manager",
}
