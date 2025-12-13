# MCP Server & Client

This directory contains both the MCP server implementation and a robust client library.

## Files

- `mcp_server.py` - MCP server with 79+ tools for AI performance engineering
- `mcp_client.py` - Robust client implementation with proper message ID tracking
- `tool_generator.py` - Tool generation utilities

## Quick Start

### Server

```bash
# Start server
python -m mcp.mcp_server --serve

# List available tools
python -m mcp.mcp_server --list
```

### Client

```python
from mcp.mcp_client import create_client

# Simple usage
with create_client(debug=True) as client:
    # List tools
    tools = client.list_tools()
    print(f"Found {len(tools)} tools")
    
    # Call a tool
    result = client.call_tool("aisp_status", {})
    print(result)
```

## Robustness Features

### Server (`mcp_server.py`)
- ✅ Request deduplication
- ✅ Automatic stale request cleanup
- ✅ Graceful error handling
- ✅ Debug logging support

### Client (`mcp_client.py`)
- ✅ Thread-safe message ID generation
- ✅ Request/response correlation
- ✅ Automatic timeout handling
- ✅ Duplicate response detection
- ✅ Graceful error recovery
- ✅ Context manager support

## Examples

See `examples/mcp_client_example.py` for usage examples.

## Documentation

- `docs/mcp_client_guide.md` - Comprehensive robustness guide
- `docs/mcp_tools.md` - Tool reference documentation

## Testing

```bash
# Test server
pytest tests/test_mcp_tools.py

# Test client
pytest tests/test_mcp_client.py
```

## Configuration

### Environment Variables

- `AISP_MCP_DEBUG=1` - Enable debug logging (server)
- `AISP_MCP_REQUEST_TIMEOUT=300` - Request timeout in seconds (server)

### Client Options

```python
client = RobustMCPClient(
    command=["python", "-m", "mcp.mcp_server", "--serve"],
    cwd="/path/to/workspace",
    timeout=300.0,  # Request timeout
    enable_debug=True  # Debug logging
)
```

## Architecture

```
┌─────────────────┐
│  MCP Client     │  ← Robust client with message ID tracking
│  (mcp_client)   │
└────────┬────────┘
         │ JSON-RPC over stdio
         ↓
┌─────────────────┐
│  MCP Server     │  ← Server with deduplication & cleanup
│  (mcp_server)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ PerformanceEngine│  ← Core functionality
│  (79+ tools)    │
└─────────────────┘
```

## Troubleshooting

### "Unknown message ID" errors

The robust client handles this automatically. If using a custom client:
1. Track all pending requests by message ID
2. Handle unknown IDs gracefully (log but don't crash)
3. Clean up stale requests periodically
4. Reset on reconnection

See `docs/mcp_client_guide.md` for details.

