import os
import shutil
import glob
from typing import List, Dict, Any

class FileDirectoryManager:
    @staticmethod
    def list_directory(path: str = '.') -> List[str]:
        """List contents of a directory."""
        return os.listdir(path)

    @staticmethod
    def create_directory(path: str) -> bool:
        """Create a new directory."""
        os.makedirs(path, exist_ok=True)
        return os.path.exists(path)

    @staticmethod
    def delete_item(path: str) -> bool:
        """Delete a file or directory."""
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        return not os.path.exists(path)

    @staticmethod
    def move_item(source: str, destination: str) -> bool:
        """Move a file or directory."""
        shutil.move(source, destination)
        return os.path.exists(destination)

    @staticmethod
    def copy_item(source: str, destination: str) -> bool:
        """Copy a file or directory."""
        if os.path.isfile(source):
            shutil.copy2(source, destination)
        elif os.path.isdir(source):
            shutil.copytree(source, destination)
        return os.path.exists(destination)

    @staticmethod
    def read_file(path: str, max_size: int = 1024 * 1024) -> str:
        """Read contents of a file (with size limit)."""
        if os.path.getsize(path) > max_size:
            return f"File is too large. Max size is {max_size} bytes."
        with open(path, 'r') as file:
            return file.read()

    @staticmethod
    def write_file(path: str, content: str) -> bool:
        """Write content to a file."""
        with open(path, 'w') as file:
            file.write(content)
        return os.path.exists(path)

    @staticmethod
    def search_files(pattern: str) -> List[str]:
        """Search for files matching a pattern."""
        return glob.glob(pattern, recursive=True)

    @staticmethod
    def get_file_info(path: str) -> Dict[str, Any]:
        """Get information about a file or directory."""
        stat = os.stat(path)
        return {
            "name": os.path.basename(path),
            "path": os.path.abspath(path),
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "is_directory": os.path.isdir(path)
        }

def file_directory_manager(action: str, **kwargs) -> Dict[str, Any]:
    """Main function to handle file and directory operations."""
    manager = FileDirectoryManager()
    try:
        if action == "list_directory":
            result = manager.list_directory(kwargs.get("path", "."))
        elif action == "create_directory":
            result = manager.create_directory(kwargs["path"])
        elif action == "delete_item":
            result = manager.delete_item(kwargs["path"])
        elif action == "move_item":
            result = manager.move_item(kwargs["source"], kwargs["destination"])
        elif action == "copy_item":
            result = manager.copy_item(kwargs["source"], kwargs["destination"])
        elif action == "read_file":
            result = manager.read_file(kwargs["path"], kwargs.get("max_size", 1024 * 1024))
        elif action == "write_file":
            result = manager.write_file(kwargs["path"], kwargs["content"])
        elif action == "search_files":
            result = manager.search_files(kwargs["pattern"])
        elif action == "get_file_info":
            result = manager.get_file_info(kwargs["path"])
        else:
            raise ValueError(f"Unknown action: {action}")
        
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}