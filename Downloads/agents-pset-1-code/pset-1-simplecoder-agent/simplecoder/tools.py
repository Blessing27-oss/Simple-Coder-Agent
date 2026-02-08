# Author: Blessing Ndeh
# Date: 03/02/2026
#  Class: AI Agents: COSC 89.34

"""For tools functions and schemas. At minimum, I need
tools to list, read, search, write, and edit source files"""

import os
import re
import glob
from dataclasses import dataclass
from typing import Callable, Dict, Any, List


@dataclass
class Tool:
    """
    Represents a tool that the agent can use.

    A tool is a function that the LLM can invoke by generating
    structured text. Each tool needs:
    - name: unique identifier
    - description: what it does (LLM reads this)
    - parameters: JSON Schema defining inputs
    - function: actual Python code to execute
    - requires_permission: whether to ask user before executing
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    requires_permission: bool = True


class ToolRegistry:
    """
    Manages all available tools for the agent.

    Responsibilities:
    1. Register tools (add them to the registry)
    2. Execute tools safely with error handling
    3. Provide tool schemas to LLM (so it knows what's available)
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Add a tool to the registry"""
        self.tools[tool.name] = tool

    def execute(self, tool_name: str, **kwargs) -> str:
        """
        Execute a tool with given parameters.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            Result as a string (or error message if something went wrong)
        """
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"

        try:
            tool = self.tools[tool_name]
            result = tool.function(**kwargs)
            return str(result)
        except TypeError as e:
            return f"Error: Invalid parameters for {tool_name}. {str(e)}"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tool schemas for LLM prompt.

        Returns a list of dictionaries describing each tool.
        The LLM reads these to understand what tools are available.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "requires_permission": tool.requires_permission
            }
            for tool in self.tools.values()
        ]

    def get_tool(self, tool_name: str) -> Tool | None:
        """Get a specific tool by name"""
        return self.tools.get(tool_name)


# ============================================================================
# TOOL FUNCTIONS - The actual operations the agent can perform
# ============================================================================

def list_files(directory: str = ".", pattern: str = "*") -> str:
    """
    List files in a directory matching a pattern.

    Args:
        directory: Directory to search in (default: current directory)
        pattern: Glob pattern to match (e.g., "*.py", "**/*.js")

    Returns:
        Formatted string listing all matching files, or error message

    Example:
        list_files(".", "*.py") -> "main.py\nagent.py\ntools.py"
    """
    try:
        # Construct the full pattern (directory + pattern)
        if pattern.startswith("**"):
            # Recursive pattern
            full_pattern = os.path.join(directory, pattern)
        else:
            full_pattern = os.path.join(directory, pattern)

        # Use glob to find matching files
        matches = glob.glob(full_pattern, recursive=True)

        if not matches:
            return f"No files found matching '{pattern}' in '{directory}'"

        # Filter out directories, keep only files
        files = [f for f in matches if os.path.isfile(f)]

        if not files:
            return f"No files found matching '{pattern}' in '{directory}'"

        # Sort and format nicely
        files.sort()
        return "\n".join(files)



    except Exception as e:
        return f"Error listing files: {str(e)}"


def read_file(file_path: str, start_line: int = 0, end_line: int = None) -> str:
    """
    Read contents of a file, optionally specifying line range.

    Args:
        file_path: Path to the file to read
        start_line: Line number to start reading from (0-indexed)
        end_line: Line number to stop reading at (exclusive). None = read all

    Returns:
        File contents as string, or error message

    Example:
        read_file("main.py") -> [full file contents]
        read_file("main.py", 0, 10) -> [first 10 lines]
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found. Use list_files to see available files."

        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file (it's a directory)."

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Handle line range
        if end_line is None:
            end_line = len(lines)

        # Validate range
        if start_line < 0 or start_line >= len(lines):
            return f"Error: start_line {start_line} out of range. File has {len(lines)} lines."

        if end_line > len(lines):
            end_line = len(lines)

        # Get the requested lines
        selected_lines = lines[start_line:end_line]

        # Format with line numbers (helpful for editing later)
        result = []
        for i, line in enumerate(selected_lines, start=start_line + 1):
            result.append(f"{i}: {line.rstrip()}")

        return "\n".join(result)

    except UnicodeDecodeError:
        return f"Error: '{file_path}' is not a text file (binary file detected)."
    except PermissionError:
        return f"Error: Permission denied reading '{file_path}'."
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file (creates new file or overwrites existing).

    Args:
        file_path: Path to the file to write
        content: Content to write to the file

    Returns:
        Success message or error message

    Example:
        write_file("hello.py", "print('Hello')") -> "File created successfully"
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Check if file already exists
        file_exists = os.path.exists(file_path)

        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        if file_exists:
            return f"File '{file_path}' overwritten successfully."
        else:
            return f"File '{file_path}' created successfully."

    except PermissionError:
        return f"Error: Permission denied writing to '{file_path}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"


def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    """
    Edit a file by replacing old_text with new_text.

    Args:
        file_path: Path to the file to edit
        old_text: Text to search for (must match exactly)
        new_text: Text to replace it with

    Returns:
        Success message or error message

    Example:
        edit_file("main.py", "def old_name():", "def new_name():")
        -> "File edited successfully. 1 occurrence replaced."
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if old_text exists
        if old_text not in content:
            return f"Error: Text not found in file. The exact text '{old_text[:50]}...' does not exist."

        # Count occurrences
        count = content.count(old_text)

        # Replace
        new_content = content.replace(old_text, new_text)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return f"File '{file_path}' edited successfully. {count} occurrence(s) replaced."

    except PermissionError:
        return f"Error: Permission denied editing '{file_path}'."
    except Exception as e:
        return f"Error editing file: {str(e)}"


def search_code(query: str, pattern: str = "**/*.py", directory: str = ".") -> str:
    """
    Search for a pattern in code files.

    Args:
        query: Text or regex pattern to search for
        pattern: File pattern to search in (default: all Python files)
        directory: Directory to search in (default: current directory)

    Returns:
        Formatted string showing matches with file paths and line numbers

    Example:
        search_code("def main", "**/*.py")
        -> "main.py:10: def main():
            utils.py:5: def main():"
    """
    try:
        # Get list of files matching pattern
        if pattern.startswith("**"):
            full_pattern = os.path.join(directory, pattern)
        else:
            full_pattern = os.path.join(directory, pattern)

        files = glob.glob(full_pattern, recursive=True)
        files = [f for f in files if os.path.isfile(f)]

        if not files:
            return f"No files found matching pattern '{pattern}' in '{directory}'"

        # Search in each file
        results = []
        total_matches = 0

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Search each line
                for line_num, line in enumerate(lines, start=1):
                    if re.search(query, line):
                        results.append(f"{file_path}:{line_num}: {line.rstrip()}")
                        total_matches += 1

            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read
                continue

        if not results:
            return f"No matches found for '{query}' in {len(files)} file(s)."

        # Limit results to avoid overwhelming output
        if len(results) > 50:
            results = results[:50]
            results.append(f"\n... ({total_matches - 50} more matches)")

        return f"Found {total_matches} match(es) in {len(files)} file(s):\n" + "\n".join(results)

    except re.error as e:
        return f"Error: Invalid regex pattern '{query}': {str(e)}"
    except Exception as e:
        return f"Error searching code: {str(e)}"


# ============================================================================
# TOOL REGISTRATION - Creating the default tool set
# ============================================================================

def create_default_tools() -> ToolRegistry:
    """
    Create and register all default tools.

    Returns:
        ToolRegistry with all tools registered
    """
    registry = ToolRegistry()

    # Tool 1: list_files
    registry.register(Tool(
        name="list_files",
        description="List files in a directory matching a glob pattern. Use this to discover what files exist.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to search in",
                    "default": "."
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g., '*.py', '**/*.js' for recursive)",
                    "default": "*"
                }
            },
            "required": []
        },
        function=list_files,
        requires_permission=False  # Safe operation
    ))

    # Tool 2: read_file
    registry.register(Tool(
        name="read_file",
        description="Read the contents of a file. Can optionally specify line range for large files.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-indexed)",
                    "default": 0
                },
                "end_line": {
                    "type": "integer",
                    "description": "Line number to stop reading at (exclusive). Omit to read entire file",
                    "default": None
                }
            },
            "required": ["file_path"]
        },
        function=read_file,
        requires_permission=False  # Safe operation
    ))

    # Tool 3: write_file
    registry.register(Tool(
        name="write_file",
        description="Create a new file or overwrite an existing file with content.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        },
        function=write_file,
        requires_permission=True  # Potentially dangerous
    ))

    # Tool 4: edit_file
    registry.register(Tool(
        name="edit_file",
        description="Edit a file by replacing exact text with new text. The old_text must match exactly.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to search for and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace it with"
                }
            },
            "required": ["file_path", "old_text", "new_text"]
        },
        function=edit_file,
        requires_permission=True  # Potentially dangerous
    ))

    # Tool 5: search_code
    registry.register(Tool(
        name="search_code",
        description="Search for a text pattern or regex in code files. Returns file paths and line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text or regex pattern to search for"
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern to search in (e.g., '**/*.py')",
                    "default": "**/*.py"
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in",
                    "default": "."
                }
            },
            "required": ["query"]
        },
        function=search_code,
        requires_permission=False  # Safe operation
    ))

    return registry