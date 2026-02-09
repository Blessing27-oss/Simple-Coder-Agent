# Author: Blessing Ndeh
# Date: 03/02/2026
#  Class: AI Agents: COSC 89.34

"""For main agent logic - implements ReAct loop"""

import re
import json
from typing import List, Dict, Any, Optional
from litellm import completion

from simplecoder.tools import create_default_tools, ToolRegistry
from simplecoder.context import ContextManager
from simplecoder.permissions import PermissionManager
from simplecoder.rag import CodeIndexer, CodeRetriever
from simplecoder.planner import Planner, TaskStatus


class Agent:
    """
    SimpleCoder AI Agent - Uses ReAct pattern to reason and act.

    The agent works by:
    1. REASON: Think about what to do next
    2. ACT: Execute a tool
    3. OBSERVE: See the result
    4. Repeat until task complete

    This loop gives the LLM agency - ability to interact with the world.
    """

    def __init__(
        self,
        model: str = "gemini/gemini-1.5-flash",
        max_iterations: int = 10,
        verbose: bool = False,
        use_planning: bool = False,
        use_rag: bool = False,
        rag_embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
        rag_index_pattern: str = "**/*.py"
    ):
        """
        Initialize the agent.

        Args:
            model: LLM model to use (e.g., "gpt-4", "gemini/gemini-pro")
            max_iterations: Max ReAct loop iterations (safety limit)
            verbose: Whether to print debug information
            use_planning: Enable task planning (Phase 4)
            use_rag: Enable semantic code search (Phase 3)
            rag_embedder: Embedding model for RAG
            rag_index_pattern: File pattern to index for RAG
        """
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.use_planning = use_planning
        self.use_rag = use_rag
        self.rag_embedder = rag_embedder
        self.rag_index_pattern = rag_index_pattern

        # Initialize tools
        self.tools: ToolRegistry = create_default_tools()

        # Phase 2: Context management with automatic compacting
        self.context = ContextManager(
            max_tokens=6000,  # Conservative limit
            compact_threshold=0.8
        )

        # Phase 2: Permission management
        self.permissions = PermissionManager(
            persist=False  # Can be made configurable
        )

        # Phase 3: Initialize RAG if enabled
        self.rag = None
        self.rag_retriever = None
        if use_rag:
            if self.verbose:
                print(f"🔍 Initializing RAG with {rag_embedder}...")

            # Create indexer and index the codebase
            indexer = CodeIndexer(embedder_name=rag_embedder)
            num_chunks = indexer.index_codebase(
                pattern=rag_index_pattern,
                directory=".",
                max_chunk_lines=50
            )

            if self.verbose:
                print(f"✅ Indexed {num_chunks} code chunks")

            # Create retriever
            self.rag_retriever = CodeRetriever(indexer)
            self.rag = indexer  # Store for reference

        # Phase 4: Initialize planner if enabled
        self.planner = None
        if use_planning:
            if self.verbose:
                print(f"📋 Initializing task planner...")
            self.planner = Planner(model=model, verbose=verbose)

    def run(self, task: str) -> str:
        """
        Main entry point - execute a task.

        Args:
            task: Natural language description of what to do

        Returns:
            Final answer from the agent

        Example:
            agent = Agent()
            result = agent.run("Create a hello world Python file")
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}\n")

        # Phase 4: If planning enabled, create plan first
        if self.use_planning and self.planner:
            return self._execute_with_planning(task)

        # Phase 3: If RAG enabled, retrieve relevant context
        if self.use_rag and self.rag_retriever:
            if self.verbose:
                print(f"🔍 Searching codebase for relevant context...")

            # Retrieve relevant code snippets
            context_str = self.rag_retriever.get_context_string(task, top_k=3)

            if self.verbose:
                print(f"📚 Found relevant code:\n{context_str[:200]}...\n")

            # Add retrieved context to conversation
            self.context.add_message(
                "system",
                f"Relevant code from the codebase:\n\n{context_str}"
            )

        # Execute the main ReAct loop
        return self._react_loop(task)

    def _execute_with_planning(self, task: str) -> str:
        """
        Execute a task using planning.

        Process:
        1. Generate plan (decompose into subtasks)
        2. Execute each subtask in dependency order
        3. Track progress
        4. Return summary

        Args:
            task: Main task

        Returns:
            Summary of execution
        """
        # Step 1: Generate plan
        plan = self.planner.create_plan(task)

        if not plan.subtasks:
            # No plan generated, fall back to regular execution
            if self.verbose:
                print("⚠️  No plan generated, executing task directly")
            return self._react_loop(task)

        # Step 2: Execute subtasks in order
        results = []

        while not plan.is_complete():
            # Get next ready task
            next_task = plan.get_next_task()

            if next_task is None:
                # No ready tasks but plan not complete
                # This shouldn't happen if dependencies are valid
                remaining = [t for t in plan.subtasks if t.status == TaskStatus.PENDING]
                error_msg = f"⚠️  Planning error: No ready tasks, but {len(remaining)} pending"
                if self.verbose:
                    print(error_msg)
                break

            # Mark as in progress
            next_task.status = TaskStatus.IN_PROGRESS

            if self.verbose:
                progress = plan.get_progress()
                print(f"\n{'─'*60}")
                print(f"🔄 Executing Task {next_task.id}/{len(plan.subtasks)}")
                print(f"   {next_task.description}")
                print(f"   Progress: {progress['completed']}/{progress['total']} complete")
                print(f"{'─'*60}\n")

            # Execute this subtask using ReAct loop
            try:
                # Clear context for fresh start on each subtask
                # But keep system messages and RAG context
                subtask_result = self._react_loop(next_task.description)

                # Mark as completed
                next_task.status = TaskStatus.COMPLETED
                next_task.result = subtask_result
                results.append(f"✅ Task {next_task.id}: {next_task.description}\n   Result: {subtask_result[:100]}...")

                if self.verbose:
                    print(f"\n✅ Task {next_task.id} completed!\n")

            except Exception as e:
                # Mark as failed
                next_task.status = TaskStatus.FAILED
                next_task.result = f"Error: {str(e)}"
                results.append(f"❌ Task {next_task.id}: {next_task.description}\n   Error: {str(e)}")

                if self.verbose:
                    print(f"\n❌ Task {next_task.id} failed: {e}\n")

                # Stop execution on failure
                break

        # Step 3: Generate summary
        progress = plan.get_progress()
        summary_lines = [
            f"# Task Planning Execution Complete\n",
            f"**Main Task:** {plan.main_task}\n",
            f"**Progress:** {progress['completed']}/{progress['total']} subtasks completed\n",
            f"\n## Execution Summary:\n"
        ]

        for i, result in enumerate(results, 1):
            summary_lines.append(f"{i}. {result}\n")

        if plan.is_complete():
            summary_lines.append(f"\n✅ **All tasks completed successfully!**")
        else:
            failed = progress['failed']
            pending = progress['pending']
            if failed > 0:
                summary_lines.append(f"\n⚠️ **{failed} task(s) failed**")
            if pending > 0:
                summary_lines.append(f"\n⚠️ **{pending} task(s) not executed**")

        return "\n".join(summary_lines)

    def _react_loop(self, task: str) -> str:
        """
        Core ReAct loop: Reason + Act + Observe (repeat).

        Process:
        1. Add user task to history
        2. Loop:
           a. Build prompt with task, history, available tools
           b. Call LLM to get next action OR final answer
           c. If action: execute tool, observe result, continue
           d. If final answer: return to user
           e. If max iterations: give up (safety)

        Args:
            task: The task to complete

        Returns:
            Final answer string
        """
        # Add initial user task to context
        self.context.add_message("user", task)

        # ReAct loop
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'─'*60}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'─'*60}")
            elif iteration > 0:
                # Show simple progress dot for multi-step tasks
                print(f"🤔 Thinking... (step {iteration + 1})")

            # REASON: Build prompt and get LLM response
            prompt = self._build_prompt()
            response = self._call_llm(prompt)

            if self.verbose:
                print(f"\n🤖 LLM Response:\n{response}\n")

            # Check if LLM decided it's done
            if self._is_final_answer(response):
                final_answer = self._extract_final_answer(response)
                if self.verbose:
                    print(f"\n✅ Task Complete!\n")
                return final_answer

            # ACT: Parse tool call from LLM response
            tool_call = self._parse_tool_call(response)

            if tool_call is None:
                # LLM didn't generate valid tool call
                # Add error to context so LLM can try again
                self.context.add_message("assistant", response)
                self.context.add_message(
                    "system",
                    "Error: No valid tool call found. Please use the format:\nThought: [your reasoning]\nAction: tool_name\nAction Input: {\"param\": \"value\"}"
                )
                continue

            # Check permissions before executing dangerous tools
            tool = self.tools.get_tool(tool_call['name'])
            if tool and tool.requires_permission:
                # Build description for permission prompt
                action_desc = f"{tool_call['name']}({tool_call['parameters']})"

                # Check permission (auto-approve in verbose mode for convenience)
                if not self.permissions.check_permission(
                    tool_call['name'],
                    action_desc,
                    auto_approve=False  # Set to self.verbose to auto-approve in verbose mode
                ):
                    # User denied permission
                    result = f"Permission denied by user for {tool_call['name']}"
                    if self.verbose:
                        print(f"🚫 {result}")

                    # Add denial to context
                    self.context.add_message("assistant", f"Action: {tool_call['name']}\nAction Input: {json.dumps(tool_call['parameters'])}")
                    self.context.add_message("user", f"Observation: {result}")
                    continue

            # Execute the tool
            if self.verbose:
                print(f"🔧 Executing: {tool_call['name']}({tool_call['parameters']})")
            else:
                # Show brief progress indicator even in non-verbose mode
                tool_desc = {
                    'write_file': f"✍️  Creating {tool_call['parameters'].get('file_path', 'file')}...",
                    'edit_file': f"✏️  Editing {tool_call['parameters'].get('file_path', 'file')}...",
                    'read_file': f"📖 Reading {tool_call['parameters'].get('file_path', 'file')}...",
                    'list_files': f"📁 Listing files...",
                    'search_code': f"🔍 Searching code..."
                }.get(tool_call['name'], f"⚙️  Running {tool_call['name']}...")
                print(tool_desc)

            result = self.tools.execute(tool_call['name'], **tool_call['parameters'])

            if self.verbose:
                # Show first 200 chars of result
                result_preview = result[:200] + "..." if len(result) > 200 else result
                print(f"📊 Result: {result_preview}\n")
            else:
                # Brief success indicator
                if "Error" not in result and "error" not in result.lower():
                    print(f"   ✅ Done")
                else:
                    print(f"   ⚠️  {result[:80]}...")

            # OBSERVE: Add interaction to context
            self.context.add_message(
                "assistant",
                f"Thought: I will use {tool_call['name']}\nAction: {tool_call['name']}\nAction Input: {json.dumps(tool_call['parameters'])}"
            )
            self.context.add_message(
                "user",
                f"Observation: {result}"
            )

        # Safety: Reached max iterations without completing
        last_messages = self.context.get_last_n_messages(2)
        return f"⚠️ Reached maximum iterations ({self.max_iterations}) without completing the task.\n\nLast observations:\n{last_messages}"

    def _build_prompt(self) -> List[Dict[str, str]]:
        """
        Construct the full prompt for the LLM.

        The prompt includes:
        1. System message (instructions on how to use tools)
        2. Conversation history (what's happened so far)

        Returns:
            List of messages in OpenAI chat format:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        # Build system message with tool descriptions
        system_message = self._build_system_message()

        # Combine system message with conversation history from context manager
        messages = [{"role": "system", "content": system_message}]
        messages.extend(self.context.get_messages())

        return messages

    def _build_system_message(self) -> str:
        """
        Create system instructions explaining how to use tools.

        This is the "operating manual" for the agent.
        It tells the LLM:
        - What tools are available
        - How to format tool calls
        - When to give final answer
        - Rules to follow
        """
        # Get tool schemas
        tool_schemas = self.tools.get_schemas()

        # Format tools in a readable way for the LLM
        tools_text = "\n\n".join([
            f"### {tool['name']}\n"
            f"**Description:** {tool['description']}\n"
            f"**Parameters:** {json.dumps(tool['parameters'], indent=2)}"
            for tool in tool_schemas
        ])

        system_prompt = f"""You are SimpleCoder, a helpful AI coding assistant that can use tools to help with software engineering tasks.

## Available Tools

{tools_text}

## How to Use Tools

To use a tool, you MUST output in this EXACT format:

Thought: [Explain your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}

You will then receive an Observation with the tool's result.

## When You're Done

When you have completed the task, output:

Thought: [Explain why the task is complete]
Final Answer: [Your response to the user]

## Important Rules

1. **Always think before acting** - Include a "Thought:" explaining your reasoning
2. **Use tools to gather information** before making changes
3. **Only use available tools** - Don't make up new ones
4. **Action Input must be valid JSON** - Use double quotes, proper escaping
5. **Take one action at a time** - Wait for observation before next action
6. **If a tool returns an error**, adjust your approach and try something else
7. **Read files before editing** - Understand what you're changing
8. **Verify your work** - After creating/editing files, read them to confirm
9. **CRITICAL: Always end with "Final Answer:"** - Even for greetings or simple questions, you MUST use the "Final Answer:" format. Never respond without it.
10. **NEVER write "Observation:" yourself** - The system provides observations after tool execution. You MUST output ONLY ONE action, then STOP. Do NOT continue with more actions or write fake observations.

## For Conversational Messages

If the user greets you or asks how you're doing (not a coding task), respond like this:

Thought: The user is greeting me, not requesting a coding task
Final Answer: Hello! I'm SimpleCoder, ready to help with your coding tasks. What would you like to work on?

## Example Interactions

**Example 1: Coding Task**

User: "Create a file called test.py with hello world"

YOUR FIRST RESPONSE (output ONLY this, then STOP):
Thought: I need to create a new file with Python code that prints hello world
Action: write_file
Action Input: {{"file_path": "test.py", "content": "print('Hello, world!')"}}

[SYSTEM PROVIDES]: Observation: File 'test.py' created successfully.

YOUR NEXT RESPONSE (after seeing the observation):
Thought: The file was created successfully. Task is complete.
Final Answer: I've created test.py with a hello world print statement.

**Example 2: Greeting/Conversation**

User: "Hello"

Thought: The user is greeting me. This is not a coding task, so I should respond politely and ask how I can help.
Final Answer: Hello! I'm SimpleCoder, your AI coding assistant. I can help you create files, search code, edit files, and more. What would you like to work on?
"""

        return system_prompt

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM API and get response.

        Uses LiteLLM which provides unified API across providers:
        - OpenAI (gpt-4, gpt-3.5-turbo)
        - Google (gemini/gemini-pro, gemini/gemini-flash)
        - Anthropic (claude-3-opus, claude-3-sonnet)
        - And many more!

        Args:
            messages: List of chat messages

        Returns:
            LLM response as string
        """
        try:
            # GPT-5 models require temperature=1, others work with 0
            temp = 1.0 if "gpt-5" in self.model.lower() else 0.0

            response = completion(
                model=self.model,
                messages=messages,
                temperature=temp,  # Deterministic for non-GPT-5 models
                max_tokens=2000   # Limit response length
            )

            content = response.choices[0].message.content

            # Handle case where API returns None content
            if content is None:
                return "Error: LLM returned empty response. Please try again."

            return content

        except Exception as e:
            error_msg = f"Error calling LLM: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return error_msg

    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call from LLM response.

        Expected format:
            Action: tool_name
            Action Input: {"param": "value"}

        Args:
            response: LLM's text response

        Returns:
            Dictionary with 'name' and 'parameters', or None if no valid tool call
            Example: {"name": "read_file", "parameters": {"file_path": "main.py"}}
        """
        # Regex to find "Action: tool_name"
        action_pattern = r"Action:\s*(\w+)"
        action_match = re.search(action_pattern, response, re.IGNORECASE)

        if not action_match:
            return None

        tool_name = action_match.group(1)

        # Regex to find "Action Input: {...}"
        # Use DOTALL to match across multiple lines
        input_pattern = r"Action Input:\s*(\{.*?\})"
        input_match = re.search(input_pattern, response, re.DOTALL | re.IGNORECASE)

        if not input_match:
            # Tool call without parameters (some tools don't need params)
            return {"name": tool_name, "parameters": {}}

        # Parse JSON parameters
        try:
            params_str = input_match.group(1)
            # Clean up the JSON string (remove extra whitespace)
            params_str = params_str.strip()
            parameters = json.loads(params_str)
            return {"name": tool_name, "parameters": parameters}
        except json.JSONDecodeError as e:
            # LLM generated invalid JSON
            if self.verbose:
                print(f"⚠️ JSON parse error: {e}")
                print(f"   Attempted to parse: {params_str}")
            return None

    def _is_final_answer(self, response: str) -> bool:
        """
        Check if LLM indicated task completion.

        Args:
            response: LLM's text response

        Returns:
            True if response contains "Final Answer:"
        """
        if response is None:
            return False
        return "Final Answer:" in response

    def _extract_final_answer(self, response: str) -> str:
        """
        Extract the final answer text from LLM response.

        Args:
            response: LLM's text response

        Returns:
            The text after "Final Answer:", or full response as fallback
        """
        if response is None:
            return "Error: Received empty response from LLM"

        match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip()  # Fallback: return full response
