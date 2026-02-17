# SimpleCoder: AI Coding Agent

**Author:** Blessing Ndeh
**Course:** AI Agents: COSC 89.34
**Problem Set:** 1, Part III

## Overview

**SimpleCoder** is a ReAct-style AI coding agent that helps with software engineering tasks through:

- **Tool Use**: File manipulation, code search, and navigation
- **ReAct Pattern**: Explicit reasoning + acting + observing loop
- **RAG**: Semantic code search using vector embeddings
- **Task Planning**: Automatic decomposition of complex tasks
- **Permission Management**: Safe file operations with user approval
- **Context Management**: Automatic compacting to stay within token limits

The agent uses LiteLLM to connect to multiple LLM providers (OpenAI, Anthropic, Google, etc.) and can assist with tasks like creating files, refactoring code, understanding codebases, and building complete applications.

**Configured to use:** GPT-4.1 via Dartmouth Chat API as the default model, with optional GPT-5.2 alias for faster performance.

## Demo

📹 **[Watch SimpleCoder in Action](https://drive.google.com/file/d/1Sa1DuANdMK8jgXFp8VDhOQdqF6ll4apG/view?usp=sharing)** - See the agent interact, plan tasks, and build applications step-by-step.

## Implementation Details

📄 **For detailed design decisions and justifications**, see **[IMPLEMENTATION.md](IMPLEMENTATION.md)**

This document covers:

- Architecture choices (why ReAct pattern?)
- Tool system design
- Context management strategy
- Permission system trade-offs
- RAG chunking and retrieval approach
- Task planning implementation

## Features

### Core Capabilities

1. **File Operations**

   - List files with glob patterns (`*.py`, `**/*.js`)
   - Read files with optional line ranges
   - Create and edit files with user permission
   - Search code using regex patterns
2. **ReAct Loop**

   - Explicit reasoning at each step
   - Tool execution with observation
   - Error recovery and adjustment
   - Maximum iteration safety limit
3. **Context Management**

   - Automatic token counting
   - Smart compacting at 80% threshold
   - Preserves system prompt and recent messages
   - Sliding window strategy
4. **Permission System**

   - Interactive approval for dangerous operations
   - Session-level and permanent settings
   - Granular per-tool control
   - Safe-by-default approach
5. **RAG (Semantic Search)**

   - Vector embeddings for code chunks
   - Smart chunking (functions/classes) or fixed-size
   - Cosine similarity ranking
   - Support for sentence-transformers or Gemini embeddings
6. **Task Planning**

   - LLM-based task decomposition
   - Dependency resolution
   - Progress tracking
   - Subtask execution with fresh contexts

## Installation

### Quick Start (For TAs)

Get up and running in 4 steps:

```bash
# 1. Navigate to the project directory
cd pset-1-simplecoder-agent

# 2. Install the package
pip install -e .

# 3. Set your Dartmouth API credentials (get key from https://chat.dartmouth.edu/)
export OPENAI_API_KEY="your-dartmouth-api-key"
export OPENAI_API_BASE="https://chat.dartmouth.edu/api"

# 4. Run the agent! (Uses Claude Sonnet 4.5 by default)
simplecoder "create a hello.py file that prints Hello World"
```

**That's it!** The agent will create the file for you.

---

### Prerequisites

- Python 3.10+
- pip
- Dartmouth Chat API key (get from https://chat.dartmouth.edu/)

### Setup

1. **Navigate to the project directory**:

   ```bash
   cd pset-1-simplecoder-agent
   ```
2. **Install dependencies**:

   ```bash
   pip install -e .
   ```
3. **Set up Dartmouth Chat API**:

   ```bash
   # Get your API key from: https://chat.dartmouth.edu/
   export OPENAI_API_KEY="your-dartmouth-api-key"
   export OPENAI_API_BASE="https://chat.dartmouth.edu/api"

   # Make it permanent:
   echo 'export OPENAI_API_KEY="your-dartmouth-api-key"' >> ~/.zshrc
   echo 'export OPENAI_API_BASE="https://chat.dartmouth.edu/api"' >> ~/.zshrc
   source ~/.zshrc
   ```
4. **Verify installation**:

   ```bash
   simplecoder --help
   ```
5. **(Optional) Alias - Not needed since default model is already set**:

   The default model is already Claude Sonnet 4.5, so you can just type `simplecoder` without specifying a model!
6. **(Optional) Suppress LiteLLM info messages**:

   ```bash
   echo 'export LITELLM_LOG=ERROR' >> ~/.zshrc
   source ~/.zshrc
   ```

## Usage

### Basic Commands

**Interactive mode** (recommended):

```bash
# Uses Claude Sonnet 4.5 by default
simplecoder

# Or specify a different Dartmouth model
simplecoder --model "openai/openai.gpt-4.1-2025-04-14"
```

**Simple task**:

```bash
simplecoder "create a hello.py file"
```

**With RAG** (semantic code search):

```bash
simplecoder --use-rag "what does the Agent class do?"
```

**With planning** (complex multi-step tasks):

```bash
simplecoder --use-planning "create a web server with routes for home and about"
```

**Available Dartmouth models** (use with `openai/` prefix):

- `openai/openai_responses.gpt-5.2-chat-latest` (GPT-5.2 - **recommended**, fast via alias)
- `openai/openai.gpt-4.1-2025-04-14` (GPT-4.1 - **default in code**, stable)
- `openai/anthropic.claude-sonnet-4-5-20250929` (Claude Sonnet 4.5 - best reasoning, slower)
- `openai/vertex_ai.gemini-2.5-pro` (Gemini 2.5 Pro - good alternative)
- `openai/mistral.mistral-large-2512` (Mistral Large)

### Example Session

```bash
$ simplecoder --use-planning "build a simple Python-based text-adventure game"

📋 Creating plan for: build a simple Python-based text-adventure game

═══════════════════════════════════════
Plan for: build a simple Python-based text-adventure game
═══════════════════════════════════════

⏳ Task 1: Create game.py file
⏳ Task 2: Define start_game() function with welcome message
⏳ Task 3: Implement first choice (left/right paths)
⏳ Task 4: Create cave_path() function
⏳ Task 5: Create clearing_path() function
⏳ Task 6: Add main block to run game

═══════════════════════════════════════

🔄 Executing Task 1/6
   Create game.py file
   Progress: 0/6 complete

Thought: I need to create an empty file first
Action: write_file
Action Input: {"file_path": "game.py", "content": ""}

⚠️  Permission Required ⚠️
Tool: write_file
Action: write_file({'file_path': 'game.py', 'content': ''})

Options:
  [y] Yes, allow this operation
  [a] Always allow 'write_file' (this session)

Your choice: a

✅ Task 1 completed!
...
```

### Command-Line Options

```bash
Options:
  --model TEXT                    LLM model to use (default: gemini/gemini-pro)
  --max-iterations INTEGER        Maximum ReAct iterations (default: 10)
  --verbose                       Enable debug output
  --interactive / --no-interactive Run in interactive mode (default: True)
  --use-planning                  Enable task decomposition
  --use-rag                       Enable semantic code search
  --rag-embedder TEXT             Embedding model (default: gemini/gemini-embedding-001)
  --rag-index-pattern TEXT        File pattern for RAG (default: **/*.py)
  --help                          Show this message and exit
```

## Project Structure

```
pset-1-simplecoder-agent/
├── simplecoder/
│   ├── __init__.py
│   ├── main.py          # CLI entry point
│   ├── agent.py         # ReAct loop implementation
│   ├── tools.py         # Tool functions and registry
│   ├── context.py       # Context management with compacting
│   ├── permissions.py   # Permission system
│   ├── rag.py           # RAG indexer and retriever
│   └── planner.py       # Task planning and decomposition
├── pyproject.toml       # Package configuration
├── README.md            # This file
└── IMPLEMENTATION.md    # Detailed design documentation
```

## Examples

### Example 1: File Creation

```bash
$ simplecoder "create a hello.py file that prints hello world"

✅ Task Complete!
Created hello.py with hello world print statement.
```

### Example 2: Code Understanding (with RAG)

```bash
$ simplecoder --use-rag "how does the ReAct loop work?"

🔍 Searching codebase for relevant context...
📚 Found relevant code in agent.py...

The ReAct loop works by:
1. Building a prompt with conversation history
2. Calling the LLM to get a response
3. Parsing tool calls from the response
4. Executing tools and observing results
5. Repeating until a final answer is provided
...
```

### Example 3: Complex Task (with Planning)

```bash
$ simplecoder --use-planning "create a Flask API with CRUD endpoints for users"

📋 Plan for: create a Flask API with CRUD endpoints for users

Task 1: Create app.py file
Task 2: Import Flask and initialize app
Task 3: Define User model/schema
Task 4: Implement GET /users endpoint
Task 5: Implement POST /users endpoint
Task 6: Implement PUT /users/<id> endpoint
Task 7: Implement DELETE /users/<id> endpoint
Task 8: Add error handling
Task 9: Add main block

Executing...
✅ All tasks completed successfully!
```

## Troubleshooting

### API Key Issues

**Problem**: `API key not valid` or `Authentication Error`

**Solution**:

```bash
# Check if keys are set
echo $OPENAI_API_KEY
echo $OPENAI_API_BASE

# If empty, set them:
export OPENAI_API_KEY="your-dartmouth-api-key"
export OPENAI_API_BASE="https://chat.dartmouth.edu/api"

# Make it permanent:
echo 'export OPENAI_API_KEY="your-dartmouth-api-key"' >> ~/.zshrc
echo 'export OPENAI_API_BASE="https://chat.dartmouth.edu/api"' >> ~/.zshrc
source ~/.zshrc
```

**Get API key**: Visit https://chat.dartmouth.edu/ and generate your Dartmouth Chat API key

### Permission Denied

**Problem**: Agent can't write files

**Solution**: Approve the operation when prompted, or use `[a]` to always allow for the session

### Token Limit Exceeded

**Problem**: Context too large

**Solution**: The context manager should handle this automatically. If issues persist, try:

- Breaking tasks into smaller subtasks
- Using `--max-iterations` to limit loop length
- Restarting the agent to clear context

### LiteLLM Info Messages

**Problem**: Seeing "Provider List" messages repeatedly

**Solution**: These are just informational messages, not errors. To suppress them:

```bash
export LITELLM_LOG=ERROR
simplecoder
```

Or redirect stderr:

```bash
simplecoder 2>/dev/null
```

## Design Decisions

Key architectural choices (see [IMPLEMENTATION.md](IMPLEMENTATION.md) for details):

1. **ReAct over Chain-of-Thought**: Enables action and observation
2. **String-based tool outputs**: Simple for LLM to parse
3. **Sliding window context**: Balances memory and relevance
4. **Permission prompts**: Safety without sandboxing complexity
5. **Smart code chunking**: Preserves semantic units (functions)
6. **LLM-based planning**: Flexible, adapts to any task

## Challenges & Solutions

### Challenge 1: Model Availability Issues (404 Errors)

**Problem**: Initial implementation used direct Gemini API (`gemini/gemini-1.5-flash`), but the model returned 404 "not found" errors through LiteLLM's API version v1beta.

**Root Cause**:
- Gemini model naming conventions changed
- LiteLLM was using an incompatible API version
- Direct Gemini API access had reliability issues

**Solution**:
- Migrated to Dartmouth Chat API (OpenAI-compatible gateway)
- Changed API credentials from `GEMINI_API_KEY` to `OPENAI_API_KEY` + `OPENAI_API_BASE`
- Updated default model from `gemini/gemini-1.5-flash` to `openai/openai.gpt-4.1-2025-04-14`
- Set up alias for GPT-5.2: `alias simplecoder="simplecoder --model openai/openai_responses.gpt-5.2-chat-latest"`

### Challenge 2: ReAct Loop Logic Bug

**Problem**: Agent would claim to complete tasks (e.g., "I created the file") but wouldn't actually execute the action.

**Root Cause**:
The agent checked for "Final Answer:" in the LLM response **before** parsing and executing tool calls. When the LLM included both an action AND a final answer in the same response, the agent would return early without executing the tool.

**Code Issue** (agent.py lines 290-294):
```python
# WRONG ORDER:
if self._is_final_answer(response):  # Checked first!
    return final_answer

tool_call = self._parse_tool_call(response)  # Never executed!
```

**Solution**:
Reversed the logic to parse tool calls first, and only check for final answer if no tool call exists:

```python
# CORRECT ORDER:
tool_call = self._parse_tool_call(response)  # Parse first!

if tool_call is None and self._is_final_answer(response):  # Only if no action
    return final_answer
```

This ensures actions are always executed before returning, fixing the reliability issue.

### Challenge 3: Performance & Speed

**Problem**: After migrating from Gemini to Dartmouth API, response times increased significantly (30-60+ seconds per task).

**Root Cause**:
- Dartmouth API adds gateway overhead
- Claude Sonnet 4.5 (initially tried) is slower than Gemini
- No model specified via alias, using slower default

**Solution**:
- Set up shell alias to use GPT-5.2 by default: `simplecoder --model openai/openai_responses.gpt-5.2-chat-latest`
- This provides fast responses (5-15 seconds) while maintaining reliability
- Alternative: GPT-4.1 for stable, moderate-speed performance

**Performance Comparison**:
- Gemini (direct, when working): ~3-5 seconds ⚡
- GPT-5.2 (Dartmouth): ~5-15 seconds ⚡✅
- GPT-4.1 (Dartmouth): ~10-20 seconds ✅
- Claude Sonnet 4.5 (Dartmouth): ~30-60 seconds 🐢

## Recent Improvements

- **Critical bug fix**: Corrected ReAct loop order to execute actions before checking final answer
- **API migration**: Moved from direct Gemini API to Dartmouth's multi-model gateway for better reliability
- **Model flexibility**: Supports GPT-4.1, GPT-5.2, Claude, and Gemini through Dartmouth API
- **Performance optimization**: Shell alias setup for fast model selection
- **Robust error handling**: Added defensive checks for None responses and API errors
- **Better prompt engineering**: Enhanced system prompt to ensure proper ReAct format

## Limitations & Future Work

### Current Limitations

- No syntax validation (can create invalid Python)
- Single-threaded execution (no parallelism)
- Limited error recovery on repeated failures
- No test execution capability

## Demo: Adventure Game

A text-adventure game created with SimpleCoder is included as a demonstration.

### Running the Game

```bash
python adventure_game.py
```

### How to Play

**Commands:**

- `north`, `south`, `east`, `west` (or `n`, `s`, `e`, `w`) - Move between rooms
- `take <item>` - Pick up an item
- `inventory` or `i` - View your inventory
- `look` - Examine the current room
- `help` - Show available commands
- `quit` - Exit the game

**Objective:** Explore the rooms, collect items, and find your way to the treasure!

---

## References

- **ReAct Pattern**: Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models"
- **LiteLLM**: Unified interface to 100+ LLMs - https://docs.litellm.ai/
- **Sentence Transformers**: Semantic embeddings - https://www.sbert.net/
- **Dartmouth Chat API**: Multi-model gateway - https://chat.dartmouth.edu/
