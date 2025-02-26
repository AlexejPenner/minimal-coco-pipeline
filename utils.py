from zenml.io import fileio
import os
from zenml.types import HTMLString

def visualize_folder_structure(path: str, max_files: int = 3, indent: str = "", level: int = 0) -> HTMLString:
    """
    Creates a visual representation of a folder structure using fileio functions.
    Shows up to max_files number of files per file extension at each level.

    Args:
        path: Remote file path to visualize
        max_files: Maximum number of files to show per extension (default: 3)
        indent: Current indentation string (used for recursion)
        level: Current directory level (used for recursion)

    Returns:
        String representation of the folder structure
    """
    if not fileio.exists(path):
        return f"{indent}Path does not exist: {path}"

    result = []
    if level == 0:
        result.append(f"📁 {os.path.basename(path) or path}")

    # Group files by extension
    files_by_ext = {}
    dirs = []

    try:
        for item in fileio.listdir(path):
            full_path = os.path.join(path, item)
            if fileio.isdir(full_path):
                dirs.append(item)
            else:
                ext = os.path.splitext(item)[1] or 'no_extension'
                if ext not in files_by_ext:
                    files_by_ext[ext] = []
                files_by_ext[ext].append(item)

        # Process directories
        for d in sorted(dirs):
            result.append(f"{indent}├── 📁 {d}")
            subdir_content = visualize_folder_structure(
                os.path.join(path, d),
                max_files,
                indent + "│   ",
                level + 1
            )
            result.append(subdir_content)

        # Process files by extension
        for ext, files in sorted(files_by_ext.items()):
            files = sorted(files)
            shown_files = files[:max_files]
            remaining = len(files) - len(shown_files)

            for f in shown_files:
                result.append(f"{indent}├── 📄 {f}")
            if remaining > 0:
                result.append(f"{indent}├── ... and {remaining} more {ext} files")

    except Exception as e:
        result.append(f"{indent}Error reading directory: {str(e)}")

    html_content = "<html><body>"
    html_content += "<p>" + "<br>".join(result) + "</p>"
    html_content += "</body></html>"
    return HTMLString(html_content)


def copy_recursive(src: str, dst: str) -> None:
    """
    Recursively copy a directory tree using fileio.copy.

    Args:
        src: Source directory path
        dst: Destination directory path
    """
    if not fileio.exists(dst):
        fileio.makedirs(dst)
    
    for item in fileio.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)
        
        if fileio.isdir(src_path):
            copy_recursive(src_path, dst_path)
        else:
            fileio.copy(src_path, dst_path)