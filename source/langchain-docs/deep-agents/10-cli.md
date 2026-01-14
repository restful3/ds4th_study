# Deep Agents CLI

Interactive command-line interface for building with Deep Agents

A terminal interface for interacting with Deep Agents, providing a direct way to build and test agents with full access to their capabilities:

- **File operations** - read, write, and edit files in your project with tools that enable agents to manage and modify code and documentation.
- **Shell command execution** - execute shell commands to run tests, build projects, manage dependencies, and interact with version control systems.
- **Web search** - search the web for up-to-date information and documentation (requires Tavily API key).
- **HTTP requests** - make HTTP requests to APIs and external services for data fetching and integration tasks.
- **Task planning and tracking** - break down complex tasks into discrete steps and track progress through the built-in todo system.
- **Memory storage and retrieval** - store and retrieve information across sessions, enabling agents to remember project conventions and learned patterns.
- **Human-in-the-loop** - require human approval for sensitive tool operations.

[Watch the demo video](https://youtu.be/IrnacLa9PJc?si=3yUnPbxnm2yaqVQb)

## Quick start

### 1. Set your API key

Export as an environment variable:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or create a `.env` file in your project root:

```text
ANTHROPIC_API_KEY=your-api-key
```

### 2. Run the CLI

```bash
uvx deepagents-cli
```

### 3. Give the agent a task

```text
> Create a Python script that prints "Hello, World!"
```

The agent proposes changes with diffs for your approval before modifying files.

<details>
<summary>Configure tracing (optional)</summary>

To trace your agents in LangSmith, set these environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
```

You can optionally set a custom project name for agent traces:

```bash
export DEEPAGENTS_LANGSMITH_PROJECT="my-agent-project"
```

For user code executed by the agent, you can set a separate project:

```bash
export LANGSMITH_PROJECT="my-user-code-project"
```

</details>

<details>
<summary>Additional installation and configuration options</summary>

**Install via pip**

If you prefer not to use `uvx`, you can install the package directly:

```bash
pip install deepagents-cli
```

**Configure OpenAI (alternative to Anthropic)**

```bash
export OPENAI_API_KEY="your-key"
```

**Configure web search**

```bash
export TAVILY_API_KEY="your-key"
```

You can also put these in a `.env` file.

</details>

## Configuration

### Command-line options

| Option | Description |
| :--- | :--- |
| `--agent NAME` | Name of the agent profile to use |
| `--auto-approve` | Skip approval prompts (or toggle with `Ctrl+T`) |
| `--sandbox TYPE` | Use a remote sandbox (`modal`, `daytona`, or `runloop`) |
| `--sandbox-id ID` | ID of an existing sandbox to connect to |
| `--sandbox-setup PATH` | Path to a setup script for the sandbox |

### CLI commands

| Command | Description |
| :--- | :--- |
| `deepagents list` | List available agent profiles |
| `deepagents help` | Show help information |
| `deepagents reset --agent NAME` | Reset an agent's state |
| `deepagents reset --agent NAME --target SOURCE` | Reset agent state to a specific source |

## Interactive mode

### Slash commands

- `/tokens` - Display token usage
- `/clear` - Clear conversation history
- `/exit` or `/quit` - Exit the CLI

### Bash commands

Prefix commands with `!` to run them directly in your shell:

```bash
!git status
!npm test
!ls -la
```

### Keyboard shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Enter` | Send message (single line) |
| `Option+Enter` / `Alt+Enter` | Insert newline |
| `Ctrl+E` | Open external editor |
| `Ctrl+T` | Toggle auto-approve mode |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit the CLI |

## Set project conventions with memories

The CLI stores project-specific memories in `~/.deepagents/AGENT_NAME/memories/`. The agent uses these memories in three ways:

1. **Research**: Searches memory for relevant context before starting tasks
2. **Response**: Checks memory when uncertain during execution
3. **Learning**: Automatically saves new information for future sessions

Example memory structure:

```text
~/.deepagents/backend-dev/memories/
├── api-conventions.md
├── database-schema.md
└── deployment-process.md
```

To set conventions, simply tell the agent:

```bash
uvx deepagents-cli --agent backend-dev
> Our API uses snake_case and includes created_at/updated_at timestamps
```

Future tasks will follow these conventions automatically:

```text
> Create a /users endpoint
# Applies conventions without prompting
```

## Use remote sandboxes

Remote sandboxes provide isolated environments for code execution with several benefits:

- **Safety**: Protect your local machine from potentially harmful code execution
- **Clean environments**: Use specific dependencies or OS configurations without local setup
- **Parallel execution**: Run multiple agents simultaneously in isolated environments
- **Long-running tasks**: Execute time-intensive operations without blocking your machine
- **Reproducibility**: Ensure consistent execution environments across teams

### To use a remote sandbox, follow these steps:

1. **Configure your sandbox provider** ([Runloop](https://www.runloop.ai/), [Daytona](https://www.daytona.io/), or [Modal](https://modal.com/)):

```bash
# Runloop
export RUNLOOP_API_KEY="your-key"

# Daytona
export DAYTONA_API_KEY="your-key"

# Modal
modal setup
```

2. **Run the CLI with a sandbox:**

```bash
uvx deepagents-cli --sandbox runloop --sandbox-setup ./setup.sh
```

The agent runs locally but executes all code operations in the remote sandbox. Optional setup scripts can configure environment variables, clone repositories, and prepare dependencies.

3. **(Optional) Create a `setup.sh` file to configure your sandbox environment:**

```bash
#!/bin/bash
set -e

# Clone repository using GitHub token
git clone https://x-access-token:${GITHUB_TOKEN}@github.com/username/repo.git $HOME/workspace
cd $HOME/workspace

# Make environment variables persistent
cat >> ~/.bashrc <<'EOF'
export GITHUB_TOKEN="${GITHUB_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd $HOME/workspace
EOF

source ~/.bashrc
```

Store secrets in a local `.env` file for the setup script to access.

---

<p align="center">
  <a href="09-middleware.md">← Previous: Middleware</a> • <a href="README.md">Table of Contents</a>
</p>
