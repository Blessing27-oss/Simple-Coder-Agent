# SimpleCoder Implementation Report

**Author:** Blessing Ndeh
**Course:** AI Agents: COSC 89.34
**Date:** February 3, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture: ReAct Pattern](#architecture-react-pattern)
3. [Tool System](#tool-system)
4. [Context Management](#context-management)
5. [Permission Management](#permission-management)
6. [RAG: Semantic Code Search](#rag-semantic-code-search)
7. [Task Planning & Decomposition](#task-planning--decomposition)
8. [Design Trade-offs](#design-trade-offs)
9. [Evaluation & Testing](#evaluation--testing)

---

## Overview

SimpleCoder is a ReAct-style coding agent that assists with software engineering tasks through tool use, semantic search, context management, and task planning. The agent follows the **Reason-Act-Observe** loop, where it:

1. **Reasons** about what to do next
2. **Acts** by executing a tool
3. **Observes** the result
4. Repeats until the task is complete

This implementation leverages Google's Gemini API (via LiteLLM) for LLM capabilities and supports multiple optional features that can be enabled via CLI flags.

---

## Architecture: ReAct Pattern

### Design Decision

I chose the **ReAct (Reasoning and Acting) pattern** as the core architecture because:

1. **Transparency**: Each step shows explicit reasoning, making the agent's behavior interpretable
2. **Error Recovery**: The agent can observe failures and adjust its approach
3. **Modularity**: Tools can be added/removed without changing the core loop
4. **Proven Pattern**: ReAct is well-established in agent research (Yao et al., 2023)

### Implementation Details

The core loop (`_react_loop` in `agent.py`) follows this structure:

```python
for iteration in range(max_iterations):
    # 1. REASON: Build prompt and get LLM response
    prompt = self._build_prompt()
    response = self._call_llm(prompt)

    # 2. Check if done
    if self._is_final_answer(response):
        return self._extract_final_answer(response)

    # 3. ACT: Parse and execute tool
    tool_call = self._parse_tool_call(response)
    result = self.tools.execute(tool_call['name'], **tool_call['parameters'])

    # 4. OBSERVE: Add to context
    self.context.add_message("assistant", thought_and_action)
    self.context.add_message("user", f"Observation: {result}")
```

### Justification

**Why ReAct over other patterns?**

- **vs. Chain-of-Thought**: ReAct adds action capabilities, making it suitable for coding tasks that require file manipulation
- **vs. Direct Prompting**: The iterative loop allows multi-step reasoning and error correction
- **vs. Function Calling**: ReAct's explicit format makes debugging easier and gives the LLM more control

**Safety Limits**: The `max_iterations` parameter prevents infinite loops, with a default of 10 iterations as a reasonable balance between task completion and safety.

---

## Tool System

### Design Decision

I implemented a **tool registry pattern** with five core tools:

1. **list_files**: Discover files using glob patterns
2. **read_file**: Read file contents with optional line ranges
3. **write_file**: Create or overwrite files
4. **edit_file**: Precise text replacement
5. **search_code**: Regex search across files

### Architecture

```
ToolRegistry
├── Tool (dataclass)
│   ├── name: str
│   ├── description: str
│   ├── parameters: JSON Schema
│   ├── function: Callable
│   └── requires_permission: bool
└── Methods
    ├── register(tool)
    ├── execute(name, **kwargs)
    └── get_schemas()
```

### Key Design Choices

#### 1. **JSON Schema for Parameters**

**Decision**: Use JSON Schema to define tool parameters

**Justification**:
- Industry standard for API specifications
- Self-documenting: the LLM can read parameter types and descriptions
- Validation-ready: could add automatic parameter validation in future

**Example**:
```python
"parameters": {
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
        }
    },
    "required": ["file_path"]
}
```

#### 2. **Separate Read and Write Operations**

**Decision**: Separate `read_file`, `write_file`, and `edit_file` rather than a single unified tool

**Justification**:
- **Safety**: Read is non-destructive and doesn't require permission; write operations are flagged
- **Clarity**: Single-purpose tools are easier for the LLM to understand
- **Granular Control**: Can set different permission levels per operation

#### 3. **Line-Numbered Output**

**Decision**: `read_file` returns content with line numbers

**Example Output**:
```
1: def hello():
2:     print("Hello, world!")
3:
4: if __name__ == "__main__":
5:     hello()
```

**Justification**:
- Makes it easier for the LLM to reference specific lines
- Helps with edit operations (can say "change line 2")
- Improves debugging when the agent makes mistakes

#### 4. **Error Handling Strategy**

**Decision**: Return error messages as strings rather than raising exceptions

**Justification**:
- Errors become observations that the LLM can read and react to
- Allows the agent to recover from mistakes (e.g., "file not found → use list_files first")
- More robust than crashing the entire agent

---

## Context Management

### The Problem

LLMs have **token limits** (e.g., Gemini 1.5 Flash: ~1M tokens input, but costs increase with context size). Without management:
- Long conversations exceed limits
- Costs spiral
- Latency increases

### Design Decision

I implemented **automatic context compacting** with a sliding window strategy:

```python
class ContextManager:
    def __init__(
        self,
        max_tokens: int = 6000,
        compact_threshold: float = 0.8
    ):
        self.messages: List[Message] = []
```

### Compaction Strategy

When token usage exceeds 80% of the limit (4,800 tokens), the system:

1. **Always keep**: First message (system prompt with tool descriptions)
2. **Always keep**: Last 5 messages (recent context)
3. **Remove**: Middle messages until back to 60% capacity
4. **Add marker**: Insert a system message noting what was compacted

**Visual representation**:
```
Before: [System | Msg1 | Msg2 | Msg3 | ... | Msg50 | Msg51 | Msg52 | Msg53 | Msg54]
After:  [System | [Compaction Note] | Msg50 | Msg51 | Msg52 | Msg53 | Msg54]
```

### Justification

**Why sliding window over other strategies?**

| Strategy | Pros | Cons | Why Not? |
|----------|------|------|----------|
| **Summarization** | Compresses info | Lossy; requires extra LLM call | Too expensive for frequent use |
| **Fixed window** | Simple | May lose system prompt | System prompt is critical |
| **Retrieval-based** | Selective retention | Complex implementation | Overkill for small contexts |
| **Sliding window** ✓ | Simple, preserves recency | Loses middle history | Best balance for our use case |

**Why 80% threshold?**
- Provides buffer before hitting hard limit
- Reduces frequency of compaction (better performance)
- Leaves room for LLM response

**Why keep last 5 messages?**
- Typical ReAct cycle: Thought → Action → Observation = 2-3 messages
- Last 5 covers the most recent 1-2 cycles
- Maintains short-term "working memory"

### Token Estimation

```python
def estimate_tokens(self) -> int:
    """1 token ≈ 4 characters (rule of thumb for English)"""
    return len(self.content) // 4
```

**Justification**: This approximation is:
- Fast (no API call needed)
- Reasonably accurate for English text (~75% accurate per OpenAI)
- Conservative (overestimates slightly, which is safer)

---

## Permission Management

### The Problem

An AI agent with file system access is **dangerous**. Without safeguards, it could:
- Overwrite important files
- Delete code
- Create malicious files

### Design Decision

I implemented a **user approval system** with session-level persistence:

```python
class PermissionManager:
    def check_permission(self, tool_name, action_description) -> bool:
        # Prompt user with options:
        # [y] Yes (once)
        # [n] No (once)
        # [a] Always allow (this session)
        # [d] Always deny (this session)
        # [A] Always allow (permanently)
        # [D] Always deny (permanently)
```

### Key Features

#### 1. **Granular Permission Levels**

**Decision**: Per-tool permissions, not blanket approval

**Justification**:
- `list_files` and `read_file` are safe → No permission needed
- `write_file` and `edit_file` are dangerous → Require approval
- User can allow reads but deny writes

#### 2. **Session vs. Persistent Permissions**

**Decision**: Support both temporary (session) and permanent (disk-persisted) permissions

**Implementation**:
```python
# Session-only (default)
pm = PermissionManager(persist=False)

# Persistent (saves to ~/.simplecoder_permissions.json)
pm = PermissionManager(persist=True)
```

**Justification**:
- **Session-only**: Good for experimentation (permissions reset each run)
- **Persistent**: Good for trusted environments (don't ask every time)
- Default to session-only for safety

#### 3. **Descriptive Prompts**

**Decision**: Show both tool name and action description

**Example**:
```
⚠️  Permission Required ⚠️
Tool: write_file
Action: write_file({'file_path': 'game.py', 'content': '...'})
```

**Justification**:
- User sees exactly what will happen
- Can make informed decision
- Transparency builds trust

### Alternative Considered: Sandboxing

**Why not run in a Docker container?**

| Approach | Pros | Cons |
|----------|------|------|
| **Docker sandbox** | Isolated; can't harm host | Complex setup; limits tool capabilities |
| **Permission system** ✓ | Simple; user maintains control | Relies on user vigilance |

I chose permissions because:
1. Simpler for educational/development use
2. User maintains full control
3. Easier to debug (no container complexity)

---

## RAG: Semantic Code Search

### The Problem

For large codebases, the LLM needs to:
- Find relevant code without reading everything
- Understand semantic relationships (not just keyword matching)
- Stay within context limits

### Design Decision

I implemented **semantic search using vector embeddings**:

```
Indexing Phase:
Code Files → Chunks → Embeddings → Vector Database

Query Phase:
User Query → Embedding → Cosine Similarity → Top-K Results
```

### Architecture

#### 1. **CodeIndexer: Building the Index**

```python
class CodeIndexer:
    def index_codebase(self, pattern="**/*.py"):
        # 1. Find files
        files = glob.glob(pattern, recursive=True)

        # 2. Chunk files
        for file in files:
            chunks = self._chunk_file(file)

        # 3. Generate embeddings
        embeddings = self.embedder.encode(chunk_contents)

        # 4. Store
        self.chunks = chunks_with_embeddings
```

#### 2. **Chunking Strategy: Smart vs. Fixed**

**Decision**: Try smart chunking first, fall back to fixed-size

**Smart Chunking** (preferred):
```python
# Detect function/class boundaries
if re.match(r'^(def |class )', line):
    # Start new chunk
```

**Fixed Chunking** (fallback):
```python
# Split every 50 lines
chunks = [lines[i:i+50] for i in range(0, len(lines), 50)]
```

**Justification**:

| Strategy | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Smart** ✓ | Semantic units (complete functions) | May fail on unusual code | Default for Python |
| **Fixed** | Always works | May split mid-function | Fallback for non-Python |

Smart chunking preserves **semantic coherence**: a function's body stays together, making it more useful as context.

#### 3. **Embedding Model Choice**

**Default**: `sentence-transformers/all-MiniLM-L6-v2`

**Justification**:
- **Small**: 80MB model, fast inference
- **Good quality**: 384-dim embeddings, competitive on semantic similarity
- **Offline**: No API calls, works without internet
- **Free**: No cost per use

**Alternative**: `gemini/gemini-embedding-001` (Google's API)

**Comparison**:

| Model | Dims | Speed | Cost | Quality |
|-------|------|-------|------|---------|
| **MiniLM** ✓ | 384 | Fast (local) | Free | Good |
| **Gemini** | 768 | Slower (API) | $0.00025/1K | Better |

I chose MiniLM as default because:
1. Good enough for most code search tasks
2. No API costs
3. Faster (local inference)
4. Works offline

#### 4. **Cosine Similarity for Search**

**Decision**: Use cosine similarity to rank chunks

```python
similarity = dot(query_vec, chunk_vec) / (||query_vec|| * ||chunk_vec||)
```

**Justification**:
- Standard for semantic search
- Range [0, 1] is intuitive (1 = most similar)
- Efficient to compute with NumPy

**Why not other metrics?**
- **Euclidean distance**: Sensitive to magnitude, not just direction
- **Dot product**: Doesn't normalize, biases toward longer vectors

#### 5. **Top-K Retrieval**

**Decision**: Default to `top_k=3` results

**Justification**:
- **3 chunks ≈ 150 lines**: Fits comfortably in context without overwhelming
- Multiple results provide context from different files
- Balances breadth (see multiple places) vs. depth (detailed code)

### Integration with ReAct Loop

```python
if self.use_rag:
    # Retrieve relevant context
    context_str = self.rag_retriever.get_context_string(task, top_k=3)

    # Add to context as system message
    self.context.add_message("system", f"Relevant code:\n{context_str}")
```

**Justification**: Adding as system message ensures the LLM sees it before reasoning, providing grounding for its responses.

---

## Task Planning & Decomposition

### The Problem

Complex tasks (e.g., "build a web server") are too big for a single ReAct loop. The agent needs to:
- Break down tasks into steps
- Execute steps in order
- Handle dependencies between steps

### Design Decision

I implemented **hierarchical planning** with a Planner + Agent architecture:

```
User Task → Planner → [Subtask 1, Subtask 2, ...] → Agent executes each
```

### Architecture

#### 1. **Plan Representation**

```python
@dataclass
class SubTask:
    id: int
    description: str
    status: TaskStatus  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    dependencies: List[int]  # IDs of tasks that must complete first
    result: Optional[str]
```

**Example Plan**:
```json
{
  "subtasks": [
    {"id": 1, "description": "Create main.py file", "dependencies": []},
    {"id": 2, "description": "Import Flask", "dependencies": [1]},
    {"id": 3, "description": "Define home route", "dependencies": [2]},
    {"id": 4, "description": "Define about route", "dependencies": [2]},
    {"id": 5, "description": "Add main block", "dependencies": [3, 4]}
  ]
}
```

#### 2. **Dependency Resolution**

**Decision**: Use a simple dependency graph with greedy execution

```python
def get_next_task(self) -> Optional[SubTask]:
    completed_ids = [t.id for t in subtasks if t.status == COMPLETED]

    for task in subtasks:
        if task.status == PENDING and all(dep in completed_ids for dep in task.dependencies):
            return task  # This task is ready
```

**Justification**:
- **Correct**: Ensures dependencies are met before execution
- **Simple**: No complex graph algorithms needed
- **Greedy**: Takes first ready task (usually correct for linear plans)

**Limitation**: Doesn't optimize for parallelism (could run tasks 3 and 4 simultaneously), but parallelism isn't supported in the current ReAct implementation anyway.

#### 3. **LLM-Based Planning**

**Decision**: Use the LLM to generate plans (not hard-coded)

**Prompt Strategy**:
```python
system_prompt = """You are an expert task planner.
Break down complex tasks into clear, sequential subtasks.

Output Format: JSON with subtasks and dependencies
Example: [shows example]
"""
```

**Justification**:
- **Flexible**: Works for any task type, not just predefined templates
- **Intelligent**: LLM understands task semantics and can identify dependencies
- **Adaptable**: As LLMs improve, planning quality improves automatically

**Fallback**: If LLM fails to generate valid JSON, create a single-task plan (execute original task directly).

#### 4. **Execution Model**

```python
for subtask in plan:
    next_task = plan.get_next_task()  # Gets next ready task
    result = self._react_loop(next_task.description)  # Execute using ReAct
    next_task.status = COMPLETED
```

**Key Decision**: Execute each subtask with a **fresh ReAct loop**

**Justification**:
- **Clean state**: Each subtask starts with clear context
- **Error isolation**: Failure in subtask N doesn't corrupt subtask N+1
- **Simplicity**: Reuses existing ReAct implementation

**Trade-off**: Loses context between subtasks (can't reference previous results). Could be improved by passing results as context to next subtask.

---

## Design Trade-offs

### 1. **String-Based Tool Results vs. Structured Objects**

**Decision**: Tools return strings, not structured data

**Pros**:
- Simple for LLM to parse (no JSON schema needed)
- Human-readable in conversation history
- Flexible format

**Cons**:
- LLM might misparse complex output
- No automatic validation

**Why chosen**: For a coding agent, string output (file contents, error messages) is natural and easy for LLMs to work with.

---

### 2. **Single-Model Architecture**

**Decision**: Use same model for planning, reasoning, and execution

**Alternative**: Specialist models (e.g., GPT-4 for planning, GPT-3.5 for execution)

**Pros of single-model**:
- Simpler configuration
- Consistent behavior
- Lower cost (if using cheaper model)

**Cons**:
- Can't optimize each component independently
- May be overkill for simple tasks

**Why chosen**: Simplicity for educational implementation. Production systems might use specialist models.

---

### 3. **Synchronous Execution**

**Decision**: Execute one tool at a time, blocking on results

**Alternative**: Async execution with parallel tools

**Pros of synchronous**:
- Simpler to implement and debug
- Matches ReAct pattern (observe before next action)
- Easier to manage state

**Cons**:
- Slower for independent operations
- Can't parallelize (e.g., read multiple files simultaneously)

**Why chosen**: Educational clarity over performance. Async could be a future enhancement.

---

## Evaluation & Testing

### Manual Testing Approach

I tested the system with increasingly complex tasks:

1. **Basic tool use**: "List Python files"
   - Tests: File discovery, tool execution
   - Success: ✓

2. **File reading**: "What does agent.py do?"
   - Tests: Read + comprehension
   - Success: ✓

3. **File creation**: "Create hello.py"
   - Tests: Write tool, permission system
   - Success: ✓

4. **Multi-step task**: "Build a text adventure game"
   - Tests: Planning, multi-file operations
   - Success: ✓

5. **RAG**: "Find all functions that use completion()"
   - Tests: Semantic search, code understanding
   - Success: ✓

### Known Limitations

1. **No syntax validation**: Agent can create invalid Python (no AST checking)
2. **Limited error recovery**: If a tool fails repeatedly, agent may loop unproductively
3. **No test execution**: Can't verify created code works
4. **Single-file context**: Hard to reason about multi-file projects

### Future Improvements

1. **Syntax checking**: Run Python AST parser before writing files
2. **Test execution**: Add tool to run pytest/unittest
3. **Git integration**: Track changes, create commits
4. **Multi-agent**: Separate planning agent, coding agent, testing agent
5. **Streaming output**: Show progress in real-time
6. **Cost tracking**: Monitor token usage and API costs

---

## Conclusion

This implementation demonstrates a working coding agent using modern AI agent techniques:

- **ReAct**: For transparent, iterative reasoning
- **Tool use**: For grounded, verifiable actions
- **RAG**: For semantic code understanding
- **Planning**: For complex task decomposition
- **Safety**: Through permission management

The design prioritizes **simplicity**, **transparency**, and **safety** over maximum performance, making it suitable for educational purposes and as a foundation for more advanced agent systems.

Total implementation: ~1,500 lines of Python across 6 modules, with comprehensive docstrings and error handling.

---

## References

- Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models"
- LiteLLM Documentation: https://docs.litellm.ai/
- Sentence Transformers: https://www.sbert.net/
- OpenAI Function Calling Guide

---

**End of Implementation Report**
