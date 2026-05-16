import dotenv from "dotenv";
import Anthropic from "@anthropic-ai/sdk";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs/promises";
import path from "node:path";

dotenv.config({
  path: ".env",
  override: true,
});

const execFileAsync = promisify(execFile);

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const MODEL = process.env.CLAUDE_MODEL ?? "claude-sonnet-4-6";

const WORKDIR = process.cwd();

const SYSTEM = `
You are DDAgent, a minimal coding agent.

You can use the bash tool to inspect the local project.
When you use a tool, wait for the tool result before continuing.
Do not claim you ran a command unless the tool result confirms it.
Keep answers concise and practical.
`.trim();

const TOOLS: Anthropic.Messages.ToolUnion[] = [
  {
    name: "bash",
    description:
      "Run a shell command in the current DDAgent project directory and return stdout/stderr. On Windows, this uses PowerShell. On macOS/Linux, this uses bash. Use this for commands like pwd, ls, npm run build, or checking project status.",
    input_schema: {
      type: "object",
      properties: {
        command: {
          type: "string",
          description: "The shell command to run.",
        },
      },
      required: ["command"],
    },
  },
  {
    name: "read_file",
    description:
      "Read a UTF-8 text file from the current DDAgent project directory. Use this instead of shell commands like cat/type when you need to inspect file contents.",
    input_schema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description:
            "Relative path to the file, for example package.json or src/index.ts.",
        },
        limit: {
          type: "number",
          description:
            "Optional maximum number of lines to return. Use this for large files.",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description:
      "Create or overwrite a UTF-8 text file inside the current DDAgent project directory. Use this when the user asks to create a new file or replace a file completely.",
    input_schema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description:
            "Relative path to the file to write, for example README.md or src/test.ts.",
        },
        content: {
          type: "string",
          description: "The complete file content to write.",
        },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "edit_file",
    description:
      "Edit a UTF-8 text file by replacing an exact old_text string with new_text. Use this for small targeted edits. The old_text must match exactly.",
    input_schema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description:
            "Relative path to the file to edit, for example src/index.ts.",
        },
        old_text: {
          type: "string",
          description: "The exact text currently in the file.",
        },
        new_text: {
          type: "string",
          description: "The replacement text.",
        },
      },
      required: ["path", "old_text", "new_text"],
    },
  },
];

type Message = Anthropic.Messages.MessageParam;

type LoopState = {
  messages: Message[];
  turnCount: number;
};

type TodoStatus = "pending" | "in_progress" | "completed";

type TodoItem = {
  content: string;
  status: TodoStatus;
  activeForm?: string;
};

class TodoManager {
  private items: TodoItem[] = [];
  public roundsSinceUpdate = 0;

  update(items: TodoItem[]) {
    const validated: TodoItem[] = [];
    let inProgressCount = 0;

    for (const item of items) {
      if (!item.content || typeof item.content !== "string") {
        throw new Error("Each todo item must have a content string.");
      }

      const status = item.status ?? "pending";

      if (!["pending", "in_progress", "completed"].includes(status)) {
        throw new Error(`Invalid todo status: ${status}`);
      }

      if (status === "in_progress") {
        inProgressCount += 1;
      }

      validated.push({
        content: item.content,
        status,
        activeForm: item.activeForm ?? "",
      });
    }

    if (inProgressCount > 1) {
      throw new Error("Only one todo item can be in_progress.");
    }

    this.items = validated;
    this.roundsSinceUpdate = 0;

    return this.render();
  }

  tick() {
    this.roundsSinceUpdate += 1;
  }

  shouldRemind() {
    return this.roundsSinceUpdate >= 3;
  }

  render() {
    if (this.items.length === 0) {
      return "[todo list is empty]";
    }

    return this.items
      .map((item) => {
        const marker =
          item.status === "completed"
            ? "[x]"
            : item.status === "in_progress"
              ? "[>]"
              : "[ ]";

        return `${marker} ${item.content}`;
      })
      .join("\n");
  }
}

const TODO = new TodoManager();

function truncateOutput(text: string, maxLength = 8000) {
  if (text.length <= maxLength) {
    return text;
  }

  return text.slice(0, maxLength) + "\n\n[Output truncated]";
}

async function runBash(command: string): Promise<string> {
  try {
    const { stdout, stderr } = await execFileAsync("bash", ["-lc", command], {
      cwd: process.cwd(),
      timeout: 10_000,
      maxBuffer: 1024 * 1024,
    });

    const output = [stdout, stderr].filter(Boolean).join("\n");

    return truncateOutput(output || "[Command completed with no output]");
  } catch (error: any) {
    const output = [
      `Command failed: ${error.message}`,
      error.stdout,
      error.stderr,
    ]
      .filter(Boolean)
      .join("\n");

    return truncateOutput(output);
  }
}

function safePath(inputPath: string) {
  if (!inputPath || typeof inputPath !== "string") {
    throw new Error("path must be a non-empty string");
  }

  const fullPath = path.resolve(WORKDIR, inputPath);
  const relative = path.relative(WORKDIR, fullPath);

  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Path escapes workspace: ${inputPath}`);
  }

  return fullPath;
}

async function runReadFile(input: { path?: string; limit?: number }) {
  if (!input.path) {
    throw new Error("missing path");
  }

  const filePath = safePath(input.path);
  const text = await fs.readFile(filePath, "utf-8");

  const lines = text.split(/\r?\n/);

  if (input.limit && input.limit > 0 && lines.length > input.limit) {
    return truncateOutput(
      lines.slice(0, input.limit).join("\n") +
        `\n\n[Output truncated to ${input.limit} lines]`,
      50000
    );
  }

  return truncateOutput(text, 50000);
}

async function runWriteFile(input: { path?: string; content?: string }) {
  if (!input.path) {
    throw new Error("missing path");
  }

  if (typeof input.content !== "string") {
    throw new Error("missing content");
  }

  const filePath = safePath(input.path);

  await fs.mkdir(path.dirname(filePath), {
    recursive: true,
  });

  await fs.writeFile(filePath, input.content, "utf-8");

  return `File written: ${input.path}`;
}

async function runEditFile(input: {
  path?: string;
  old_text?: string;
  new_text?: string;
}) {
  if (!input.path) {
    throw new Error("missing path");
  }

  if (typeof input.old_text !== "string") {
    throw new Error("missing old_text");
  }

  if (typeof input.new_text !== "string") {
    throw new Error("missing new_text");
  }

  const filePath = safePath(input.path);
  const text = await fs.readFile(filePath, "utf-8");

  if (!text.includes(input.old_text)) {
    throw new Error("old_text not found in file");
  }

  const nextText = text.replace(input.old_text, input.new_text);

  await fs.writeFile(filePath, nextText, "utf-8");

  return `File edited: ${input.path}`;
}

const TOOL_HANDLERS: Record<string, (input: any) => Promise<string>> = {
  bash: async (input: { command?: string }) => {
    if (!input.command) {
      throw new Error("missing command");
    }

    console.log(`\n[tool:bash] ${input.command}`);

    return await runBash(input.command);
  },

  read_file: async (input: { path?: string; limit?: number }) => {
    console.log(`\n[tool:read_file] ${input.path}`);

    return await runReadFile(input);
  },

  write_file: async (input: { path?: string; content?: string }) => {
    console.log(`\n[tool:write_file] ${input.path}`);

    return await runWriteFile(input);
  },

  edit_file: async (input: {
    path?: string;
    old_text?: string;
    new_text?: string;
  }) => {
    console.log(`\n[tool:edit_file] ${input.path}`);

    return await runEditFile(input);
  },
};

async function runTool(block: Anthropic.Messages.ToolUseBlock): Promise<string> {
  try {
    const handler = TOOL_HANDLERS[block.name];

    if (!handler) {
      return `Tool error: unknown tool "${block.name}"`;
    }

    return await handler(block.input);
  } catch (error) {
    return `Tool error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function getFinalText(content: Anthropic.Messages.ContentBlock[]) {
  return content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}

async function agentLoop(query: string) {
  const state: LoopState = {
    messages: [
      {
        role: "user",
        content: query,
      },
    ],
    turnCount: 1,
  };

  while (true) {
    console.log(`\n--- Turn ${state.turnCount} ---`);

    const response = await client.messages.create({
      model: MODEL,
      system: SYSTEM,
      messages: state.messages,
      tools: TOOLS,
      max_tokens: 4000,
    });

    state.messages.push({
      role: "assistant",
      content: response.content,
    });

    if (response.stop_reason !== "tool_use") {
      const finalText = getFinalText(response.content);

      console.log("\nAssistant:\n");
      console.log(finalText || "[No text response]");
      return;
    }

    const toolResults: Anthropic.Messages.ToolResultBlockParam[] = [];

    for (const block of response.content) {
      if (block.type === "tool_use") {
        const output = await runTool(block);

        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: output,
        });
      }
    }

    state.messages.push({
      role: "user",
      content: toolResults,
    });

    state.turnCount += 1;
  }
}

const query = process.argv.slice(2).join(" ");

if (!query) {
  console.error('Usage: npm run dev -- "请运行 pwd，然后告诉我当前目录"');
  process.exit(1);
}

agentLoop(query).catch((error) => {
  console.error(error);
  process.exit(1);
});