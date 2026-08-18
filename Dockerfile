FROM python:3.12-slim

WORKDIR /app

# Build from this repo tip (not PyPI) so Glama Deploy evaluates current tools.
# Pin mcp<2: mcp 2.0 removed Server.list_tools / call_tool decorator API used by these servers.
COPY . .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir '.[mcp]' 'mcp>=1.0,<2'

# MCP stdio server — Glama sends JSON-RPC over stdin, reads responses from stdout
CMD ["agentdelta-mcp"]
