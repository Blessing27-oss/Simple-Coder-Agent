# Author: Blessing Ndeh
# Date: 03/02/2026
#  Class: AI Agents: COSC 89.34

"""For managing task and session level permissions from the user for reading, writing files, etc."""

import os
import json
from typing import Dict, Optional


class PermissionManager:
    """
    Manages user permissions for dangerous operations.

    Features:
    1. Ask user before dangerous operations (write, edit files)
    2. Remember choices for the session
    3. Optional persistence to disk (across sessions)

    Permission Levels:
    - "allow": Execute without asking
    - "deny": Always deny
    - "ask": Ask each time (default)
    """

    def __init__(self, persist: bool = False, config_file: Optional[str] = None):
        """
        Initialize permission manager.

        Args:
            persist: Whether to save permissions to disk
            config_file: Path to config file (default: ~/.simplecoder_permissions.json)
        """
        self.persist = persist
        self.config_file = config_file or os.path.expanduser("~/.simplecoder_permissions.json")

        # In-memory permission cache
        # Format: {tool_name: "allow" | "deny" | "ask"}
        self.permissions: Dict[str, str] = {}

        # Load persisted permissions if enabled
        if self.persist:
            self._load_permissions()

    def check_permission(
        self,
        tool_name: str,
        action_description: str,
        auto_approve: bool = False
    ) -> bool:
        """
        Check if user permits this operation.

        Args:
            tool_name: Name of the tool being used
            action_description: Human-readable description of what will happen
            auto_approve: If True, automatically approve (for testing/verbose mode)

        Returns:
            True if permitted, False if denied
        """
        # Check if we have a cached permission
        if tool_name in self.permissions:
            cached = self.permissions[tool_name]
            if cached == "allow":
                return True
            elif cached == "deny":
                return False
            # If "ask", fall through to prompt user

        # Auto-approve if requested (useful for --verbose testing)
        if auto_approve:
            return True

        # Ask user for permission
        return self._prompt_user(tool_name, action_description)

    def _prompt_user(self, tool_name: str, action_description: str) -> bool:
        """
        Prompt user for permission.

        Args:
            tool_name: Tool name
            action_description: What will happen

        Returns:
            True if user approves, False otherwise
        """
        print(f"\n⚠️  Permission Required ⚠️")
        print(f"Tool: {tool_name}")
        print(f"Action: {action_description}")
        print(f"\nOptions:")
        print(f"  [y] Yes, allow this operation")
        print(f"  [n] No, deny this operation")
        print(f"  [a] Always allow '{tool_name}' (this session)")
        print(f"  [d] Always deny '{tool_name}' (this session)")

        if self.persist:
            print(f"  [A] Always allow '{tool_name}' (permanently)")
            print(f"  [D] Always deny '{tool_name}' (permanently)")

        while True:
            response = input("\nYour choice: ").strip().lower()

            if response == 'y':
                return True
            elif response == 'n':
                return False
            elif response == 'a':
                # Allow for this session
                self.permissions[tool_name] = "allow"
                return True
            elif response == 'd':
                # Deny for this session
                self.permissions[tool_name] = "deny"
                return False
            elif response == 'A' and self.persist:
                # Allow permanently
                self.permissions[tool_name] = "allow"
                self._save_permissions()
                return True
            elif response == 'D' and self.persist:
                # Deny permanently
                self.permissions[tool_name] = "deny"
                self._save_permissions()
                return False
            else:
                print("Invalid choice. Please enter y, n, a, d" +
                      (", A, or D" if self.persist else ""))

    def set_permission(self, tool_name: str, level: str):
        """
        Manually set permission level for a tool.

        Args:
            tool_name: Tool name
            level: "allow", "deny", or "ask"
        """
        if level not in ["allow", "deny", "ask"]:
            raise ValueError(f"Invalid permission level: {level}")

        self.permissions[tool_name] = level

        if self.persist:
            self._save_permissions()

    def reset_permissions(self):
        """Clear all permissions (reset to asking each time)"""
        self.permissions.clear()

        if self.persist:
            self._save_permissions()

    def _load_permissions(self):
        """Load permissions from disk"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.permissions = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load permissions: {e}")
            self.permissions = {}

    def _save_permissions(self):
        """Save permissions to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            with open(self.config_file, 'w') as f:
                json.dump(self.permissions, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save permissions: {e}")

    def get_all_permissions(self) -> Dict[str, str]:
        """Get all current permissions"""
        return self.permissions.copy()