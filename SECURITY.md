# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Email: agent.obs.oss.security@gmail.com
Subject line: `[agent-observability SECURITY] <brief description>`

We will acknowledge within 48 hours and provide a fix timeline within 7 days.
Public disclosure follows 90 days after the report, or after the patch ships — whichever comes first.

## Scope

- **In scope:** Core library (`src/`), CLI, fixture storage, transport interceptors
- **Best-effort:** Third-party integrations (`src/agent_trace/integrations/`) — please identify the specific integration
- **Out of scope:** Issues in dependencies (report directly to the dependency maintainers)

## Security Considerations

**Fixture files contain full HTTP request/response bodies**, including any API keys or sensitive data passed in headers or request bodies. Do not commit fixture files to version control. Add `*.db` to your `.gitignore`.

The HTTP transport interceptor operates at the application layer and does not bypass TLS. It records cleartext bodies after decryption — treat fixture files with the same sensitivity as your API keys.
