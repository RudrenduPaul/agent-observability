#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");

const args = process.argv.slice(2);

const INSTALL_HELP = `
agent-trace-cli is an npm wrapper: the actual CLI ships as a Python package.

Install it, then re-run this command:

  pip install agent-observability-trace-cli
  # or
  uv add agent-observability-trace-cli
  # or, for an isolated global install
  pipx install agent-observability-trace-cli

Docs: https://github.com/RudrenduPaul/agent-observability
`;

function run(command, commandArgs, extraEnv) {
  return spawnSync(command, commandArgs, {
    stdio: "inherit",
    env: { ...process.env, ...extraEnv },
  });
}

// Guard against self-recursion: npx/npm exec prepend this package's own
// node_modules/.bin (or the npx cache dir) to PATH before running this
// script. That means a plain PATH lookup for "agent-trace" can resolve
// back to THIS wrapper instead of the real pip/uv/pipx-installed console
// script -- and since this script's own exit/error handling only treats
// an ENOENT as "not found," a self-match doesn't error, it just re-execs
// itself, unboundedly, spawning child processes until the machine's
// process table is exhausted. A previously-published version of this
// wrapper did exactly that (confirmed via a real recursive self-exec
// reproduction in a fresh npx invocation). The fix: mark the environment
// once before the first attempt, and if this process sees that marker
// already set, skip straight to the Python fallback below instead of
// trying "agent-trace" on PATH again.
const RECURSION_GUARD = "AGENT_TRACE_JS_WRAPPER_ACTIVE";
if (!process.env[RECURSION_GUARD]) {
  // Preferred path: the real "agent-trace" console script is on PATH
  // (installed via pip/uv/pipx per project.scripts in pyproject.toml).
  let result = run("agent-trace", args, { [RECURSION_GUARD]: "1" });
  if (!(result.error && result.error.code === "ENOENT")) {
    if (result.error) {
      throw result.error;
    }
    process.exit(result.status === null ? 1 : result.status);
  }
}

// Fallback: agent-trace's console script isn't on PATH (e.g. pip
// installed into a venv not exported to PATH). Try invoking the
// module's CLI entry point directly through Python. Output is
// captured (not streamed) here so a missing module can be turned into
// the friendlier INSTALL_HELP message below instead of a raw traceback.
const pyScript =
  "import sys; from agent_trace._cli import main; sys.argv = ['agent-trace'] + sys.argv[1:]; main()";
let pythonFound = false;
for (const python of ["python3", "python"]) {
  const pyResult = spawnSync(python, ["-c", pyScript, ...args], {
    stdio: ["inherit", "inherit", "pipe"],
    encoding: "utf8",
  });
  if (pyResult.error && pyResult.error.code === "ENOENT") {
    continue;
  }
  pythonFound = true;
  if (pyResult.stderr && pyResult.stderr.includes("ModuleNotFoundError: No module named 'agent_trace'")) {
    break;
  }
  if (pyResult.stderr) {
    process.stderr.write(pyResult.stderr);
  }
  process.exit(pyResult.status === null ? 1 : pyResult.status);
}

process.stderr.write(INSTALL_HELP);
process.exit(1);
