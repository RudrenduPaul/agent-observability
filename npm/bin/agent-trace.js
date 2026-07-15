#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");

const args = process.argv.slice(2);

const INSTALL_HELP = `
agent-trace-cli is an npm wrapper: the actual CLI ships as a Python package.

Install it, then re-run this command:

  pip install agent-observability-trace
  # or
  uv add agent-observability-trace
  # or, for an isolated global install
  pipx install agent-observability-trace

Docs: https://github.com/RudrenduPaul/agent-observability
`;

function run(command, commandArgs) {
  return spawnSync(command, commandArgs, { stdio: "inherit" });
}

// Preferred path: the real "agent-trace" console script is on PATH
// (installed via pip/uv/pipx per project.scripts in pyproject.toml).
let result = run("agent-trace", args);
if (!(result.error && result.error.code === "ENOENT")) {
  if (result.error) {
    throw result.error;
  }
  process.exit(result.status === null ? 1 : result.status);
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
