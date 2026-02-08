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

**Configured to use:** Dartmouth's multi-model API gateway with GPT-5.2 or GPT-4.1 as the primary model.

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

### Prerequisites

- Python 3.10+
- pip

### Setup

1. **Clone the repository**:

   ```bash
   cd pset-1-simplecoder-agent
   ```
2. **Install dependencies**:

   pip install -e .
3. **Set up Dartmouth API**:

   ```bash
   # Get your API key from: https://chat.dartmouth.edu/
   export OPENAI_API_KEY="your-dartmouth-api-key"
   export OPENAI_API_BASE="https://chat.dartmouth.edu/api"

   # Make it permanent:
   echo 'export OPENAI_API_KEY="your-dartmouth-api-key"' >> ~/.zshrc
   echo 'export OPENAI_API_BASE="https://chat.dartmouth.edu/api"' >> ~/.zshrc
   source ~/.zshrc
   ```

   **Note**: Dartmouth's API provides access to 40+ models including Claude, GPT, Gemini, and Mistral.
4. **Verify installation**:

   ```bash
   simplecoder --help
   ```
5. **(Optional) Create alias for easier use**:

   ```bash
   # Add to ~/.zshrc for convenience (choose one)

   # Option 1: GPT-5.2 (latest, requires temperature=1)
   echo 'alias simplecoder="simplecoder --model openai/openai_responses.gpt-5.2-chat-latest"' >> ~/.zshrc

   # Option 2: GPT-4.1 (stable, recommended)
   echo 'alias simplecoder="simplecoder --model openai/openai.gpt-4.1-2025-04-14"' >> ~/.zshrc

   source ~/.zshrc

   # Now you can just type: simplecoder
   ```
6. **(Optional) Suppress LiteLLM info messages**:

   ```bash
   echo 'export LITELLM_LOG=ERROR' >> ~/.zshrc
   source ~/.zshrc
   ```

## Usage

### Basic Commands

**Interactive mode** (recommended):

```bash
# With GPT-5.2 (latest)
simplecoder --model "openai/openai_responses.gpt-5.2-chat-latest"

# Or with GPT-4.1 (stable)
simplecoder --model "openai/openai.gpt-4.1-2025-04-14"
```

**Simple task**:

```bash
simplecoder --model "openai/openai_responses.gpt-5.2-chat-latest" "create a hello.py file"
```

**With RAG** (semantic code search):

```bash
simplecoder --model "openai/openai.gpt-4.1-2025-04-14" --use-rag "what does the Agent class do?"
```

**With planning** (complex multi-step tasks):

```bash
simplecoder --model "openai/openai.gpt-4.1-2025-04-14" --use-planning "create a web server with routes for home and about"
```

**Available Dartmouth models** (use with `openai/` prefix):

- `openai/openai_responses.gpt-5.2-chat-latest` (latest OpenAI)
- `openai/openai.gpt-4.1-2025-04-14` (stable, recommended)
- `openai/anthropic.claude-sonnet-4-5-20250929` (for Claude)
- `openai/vertex_ai.gemini-2.5-pro` (for Gemini)
- `openai/mistral.mistral-large-2512` (for Mistral)
- [See full list](https://chat.dartmouth.edu/)

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
  --model TEXT                    LLM model to use (default: gemini/gemini-1.5-flash)
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
```

**Get API key**: Visit https://chat.dartmouth.edu/ and generate your API key

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

## Recent Improvements

- ✅ **Robust error handling**: Added defensive checks for None responses from LLM
- ✅ **Better prompt engineering**: Enhanced system prompt to ensure proper ReAct format
- ✅ **Multi-provider support**: Configured for Dartmouth's multi-model API gateway
- ✅ **GPT-5 compatibility**: Auto-adjusts temperature for GPT-5 models (temperature=1)
- ✅ **Model flexibility**: Supports GPT-4.1, GPT-5.2, Claude, Gemini, and Mistral

## Limitations & Future Work

### Current Limitations

- No syntax validation (can create invalid Python)
- Single-threaded execution (no parallelism)
- Limited error recovery on repeated failures
- No test execution capability

### Future Improvements

- [ ] Add Python AST validation before writing files
- [ ] Tool for running tests (pytest/unittest)
- [ ] Git integration (commit, branch, push)
- [ ] Multi-agent architecture (planner/coder/tester agents)
- [ ] Streaming output for real-time progress
- [ ] Cost and token usage tracking
- [ ] Support for more languages (JavaScript, Go, etc.)

## References

- **ReAct Pattern**: Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models"
- **LiteLLM**: Unified interface to 100+ LLMs - https://docs.litellm.ai/
- **Sentence Transformers**: Semantic embeddings - https://www.sbert.net/
- **Dartmouth Chat API**: Multi-model gateway - https://chat.dartmouth.edu/
- **Claude Sonnet 4.5**: Anthropic's latest model - https://www.anthropic.com/
