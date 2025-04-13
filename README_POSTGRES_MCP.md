# PostgreSQL MCP Server for ME2AI

This document describes the PostgreSQL Model Context Protocol (MCP) server implementation for ME2AI, which allows AI assistants in Windsurf to directly query and interact with your PostgreSQL database.

## Features

- Direct database access from Windsurf AI assistants
- Schema-specific queries with proper security measures
- Special tools for common database operations
- PLZ (postal code) lookup functionality

## Available Tools

### 1. execute_query

Executes a SQL query on the PostgreSQL database.

**Parameters:**
- `query` (string): SQL query to execute
- `schema` (string): Database schema to use (e.g., 'poco' or 'poco-test')

**Example usage in Windsurf:**
```
I need to find all entries in the customer table.
[Uses execute_query tool with query "SELECT * FROM customers LIMIT 10" and schema "poco"]
```

### 2. list_tables

Lists all tables in a specific schema.

**Parameters:**
- `schema` (string): Database schema to list tables from (e.g., 'poco' or 'poco-test')

**Example usage in Windsurf:**
```
What tables are available in the poco-test schema?
[Uses list_tables tool with schema "poco-test"]
```

### 3. get_plz_details

Gets details for a specific postal code (PLZ).

**Parameters:**
- `plz` (string): Postal code to look up (e.g., '72358')
- `schema` (string): Database schema to use (e.g., 'poco' or 'poco-test')

**Example usage in Windsurf:**
```
I need information about postal code 72358.
[Uses get_plz_details tool with plz "72358" and schema "poco"]
```

## Configuration

The MCP server is configured in Windsurf's MCP configuration file:
```
~/.codeium/windsurf/mcp_config.json
```

Current configuration:
```json
{
  "mcpServers": {
    "postgres-db": {
      "command": "python",
      "args": [
        "postgres_mcp.py"
      ],
      "cwd": "c:\\Users\\achim\\github\\me2ai",
      "env": {
        "POSTGRES_HOST": "c9ijs3l3qhrn1.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "uem4h7dfn2ghbi",
        "POSTGRES_PASSWORD": "******",
        "POSTGRES_DATABASE": "d2tqio36qhkn50"
      }
    }
  }
}
```

## Important Notes

1. Always specify the schema ('poco' or 'poco-test') when querying the database
2. For PLZ function calls, use the string format: `SELECT * FROM "poco".fn_plz_details('72358')`
3. Remember that PLZ values are stored as INTEGER in the database but should be passed as VARCHAR in parameters
4. All database operations are logged in the console output
5. Query results are limited to 50 rows maximum to prevent overwhelming responses

## Starting the Server

To start the MCP server manually:

```bash
python postgres_mcp.py
```

When Windsurf is opened, it should automatically start the server based on the MCP configuration.

## Troubleshooting

If the MCP server isn't working properly:

1. Check if the server is running:
   ```powershell
   Get-Process -Name python | Where-Object {$_.CommandLine -like "*postgres_mcp*"}
   ```

2. Verify database connectivity:
   ```bash
   python -c "import psycopg2; conn = psycopg2.connect('postgres://uem4h7dfn2ghbi:pc0f9dae381d06c5c45e0fd030b76f6fbe48d3d95d8cabb93551dc81fa5539f7e@c9ijs3l3qhrn1.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com:5432/d2tqio36qhkn50'); print('Connection successful')"
   ```

3. Check the MCP server logs in the console for detailed error messages

## Future Enhancements

- Add support for more database operations
- Implement query caching for frequently accessed data
- Add visualization capabilities for query results
- Create specialized tools for common business operations
