# Author: Blessing Ndeh
# Date: 03/02/2026
#  Class: AI Agents: COSC 89.34

"""Task planning and decomposition - breaks complex tasks into manageable subtasks"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from litellm import completion


class TaskStatus(Enum):
    """Status of a subtask"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """
    Represents a single subtask in a plan.

    A subtask is one atomic step toward completing the main task.
    It has dependencies (other subtasks that must complete first).
    """
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[int] = field(default_factory=list)  # IDs of tasks this depends on
    result: Optional[str] = None  # Result after completion

    def is_ready(self, completed_ids: List[int]) -> bool:
        """
        Check if this task is ready to execute.

        A task is ready if all its dependencies are completed.

        Args:
            completed_ids: List of completed task IDs

        Returns:
            True if ready to execute
        """
        return all(dep_id in completed_ids for dep_id in self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "result": self.result
        }


@dataclass
class Plan:
    """
    Represents a complete plan with multiple subtasks.

    A plan is an ordered list of subtasks that together
    accomplish the main goal.
    """
    main_task: str
    subtasks: List[SubTask] = field(default_factory=list)

    def add_subtask(self, subtask: SubTask):
        """Add a subtask to the plan"""
        self.subtasks.append(subtask)

    def get_next_task(self) -> Optional[SubTask]:
        """
        Get the next task that's ready to execute.

        Returns:
            Next pending task that has all dependencies met, or None
        """
        completed_ids = [
            task.id for task in self.subtasks
            if task.status == TaskStatus.COMPLETED
        ]

        for task in self.subtasks:
            if task.status == TaskStatus.PENDING and task.is_ready(completed_ids):
                return task

        return None

    def is_complete(self) -> bool:
        """Check if all tasks are completed"""
        return all(task.status == TaskStatus.COMPLETED for task in self.subtasks)

    def get_progress(self) -> Dict[str, int]:
        """
        Get progress statistics.

        Returns:
            Dict with counts of pending, in_progress, completed, failed tasks
        """
        stats = {
            "total": len(self.subtasks),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0
        }

        for task in self.subtasks:
            stats[task.status.value] += 1

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "main_task": self.main_task,
            "subtasks": [task.to_dict() for task in self.subtasks],
            "progress": self.get_progress()
        }


class Planner:
    """
    Plans and executes complex tasks by decomposition.

    Uses LLM to break down complex tasks into subtasks,
    then executes them in dependency order.
    """

    def __init__(self, model: str = "openai/openai.gpt-4.1-2025-04-14", verbose: bool = False):
        """
        Initialize planner.

        Args:
            model: LLM model to use for planning
            verbose: Whether to print debug info
        """
        self.model = model
        self.verbose = verbose

    def create_plan(self, task: str) -> Plan:
        """
        Create a plan by decomposing a task into subtasks.

        Uses LLM to generate subtasks with dependencies.

        Args:
            task: The main task to decompose

        Returns:
            Plan object with subtasks
        """
        if self.verbose:
            print(f"🗓️  Creating plan for: {task}")

        # Build prompt for planning
        prompt = self._build_planning_prompt(task)

        # Call LLM to generate plan
        response = self._call_llm(prompt)

        # Parse LLM response into Plan
        plan = self._parse_plan(task, response)

        if self.verbose:
            self._print_plan(plan)

        return plan

    def _build_planning_prompt(self, task: str) -> List[Dict[str, str]]:
        """
        Build prompt for LLM to generate plan.

        Args:
            task: Main task

        Returns:
            Messages list for LLM
        """
        system_prompt = """You are an expert task planner for software engineering tasks.

Your job: Break down complex tasks into clear, sequential subtasks.

## Guidelines

1. **Atomic subtasks**: Each subtask should be a single, clear action
2. **Dependencies**: Identify which tasks must complete before others
3. **Order**: List tasks in logical execution order
4. **Specificity**: Be concrete, not vague

## Output Format

You MUST respond with ONLY valid JSON in this exact format:

{
  "subtasks": [
    {
      "id": 1,
      "description": "Clear description of what to do",
      "dependencies": []
    },
    {
      "id": 2,
      "description": "Another task",
      "dependencies": [1]
    }
  ]
}

**Important:**
- Use sequential IDs starting from 1
- dependencies is a list of task IDs that must complete first
- Empty dependencies [] means no prerequisites
- Only output valid JSON, nothing else

## Example

Task: "Create a Flask web app with home and about pages"

Response:
{
  "subtasks": [
    {
      "id": 1,
      "description": "Create main.py file",
      "dependencies": []
    },
    {
      "id": 2,
      "description": "Import Flask and create app instance",
      "dependencies": [1]
    },
    {
      "id": 3,
      "description": "Define home route that returns 'Welcome'",
      "dependencies": [2]
    },
    {
      "id": 4,
      "description": "Define about route that returns 'About page'",
      "dependencies": [2]
    },
    {
      "id": 5,
      "description": "Add if __name__ == '__main__': app.run() block",
      "dependencies": [3, 4]
    }
  ]
}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\n\nGenerate a plan:"}
        ]

        return messages

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call LLM to generate plan.

        Args:
            messages: Messages list

        Returns:
            LLM response
        """
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Slight creativity but mostly deterministic
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            if self.verbose:
                print(f"❌ Error calling LLM: {e}")
            raise

    def _parse_plan(self, main_task: str, response: str) -> Plan:
        """
        Parse LLM response into a Plan object.

        Args:
            main_task: Original task
            response: LLM's JSON response

        Returns:
            Plan object
        """
        try:
            # Extract JSON from response (in case LLM adds extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

            # Parse JSON
            data = json.loads(json_str)

            # Create plan
            plan = Plan(main_task=main_task)

            # Add subtasks
            for task_data in data.get("subtasks", []):
                subtask = SubTask(
                    id=task_data["id"],
                    description=task_data["description"],
                    dependencies=task_data.get("dependencies", [])
                )
                plan.add_subtask(subtask)

            return plan

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ Failed to parse plan JSON: {e}")
                print(f"Response was: {response}")

            # Fallback: create single-task plan
            plan = Plan(main_task=main_task)
            plan.add_subtask(SubTask(
                id=1,
                description=main_task,
                dependencies=[]
            ))
            return plan

    def _print_plan(self, plan: Plan):
        """
        Pretty print a plan.

        Args:
            plan: Plan to print
        """
        print(f"\n{'='*60}")
        print(f"📋 Plan for: {plan.main_task}")
        print(f"{'='*60}\n")

        for task in plan.subtasks:
            deps_str = f" (depends on: {task.dependencies})" if task.dependencies else ""
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌"
            }[task.status]

            print(f"{status_icon} Task {task.id}: {task.description}{deps_str}")

        print(f"\n{'='*60}\n")

    def format_plan_for_display(self, plan: Plan) -> str:
        """
        Format plan as a string for display.

        Args:
            plan: Plan to format

        Returns:
            Formatted string
        """
        lines = [
            f"## Plan: {plan.main_task}\n",
            f"**Progress:** {plan.get_progress()['completed']}/{plan.get_progress()['total']} tasks completed\n"
        ]

        for task in plan.subtasks:
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌"
            }[task.status]

            deps_str = f" (after: {task.dependencies})" if task.dependencies else ""
            lines.append(f"{status_icon} **Task {task.id}**: {task.description}{deps_str}")

        return "\n".join(lines)
