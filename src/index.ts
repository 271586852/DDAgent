import "dotenv/config";
import Anthropic from "@anthropic-ai/sdk";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const MODEL = process.env.CLAUDE_MODEL ?? "claude-sonnet-4-6";

const SYSTEM = `
You are DDAgent, a minimal coding agent.

You can use the bash tool to inspect the local project.
When you use a tool, wait for the tool result before continuing.
Do not claim you ran a command unless the tool result confirms it.
Keep answers concise and practical.
`.trim();

const TOOLS = [
  {
    name: "bash",
    description:
      "Run a bash command in the current DDAgent project directory and return stdout/stderr.",
    input_schema: {
      type: "object",
      properties: {
        command: {
          type: "string",
          description: "The bash command to run.",
        },
      },
      required: ["command"],
    },
  },
] as const;

type Message = Anthropic.Messages.MessageParam;

type LoopState = {
  messages: Message[];
  turnCount: number;
};

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

async function runTool(block: Anthropic.Messages.ToolUseBlock): Promise<string> {
  if (block.name === "bash") {
    const input = block.input as { command?: string };

    if (!input.command) {
      return "Tool error: missing command";
    }

    console.log(`\n[tool:bash] ${input.command}`);

    return await runBash(input.command);
  }

  return `Tool error: unknown tool "${block.name}"`;
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