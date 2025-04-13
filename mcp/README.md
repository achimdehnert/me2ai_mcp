# ME2AI Model Context Protocol (MCP) Server

This module implements a Model Context Protocol (MCP) server for ME2AI, providing AI assistants like Cascade with the ability to interact directly with the ME2AI system and database resources.

## Postgres Database Agent

The Postgres Database Agent provides direct access to your Postgres database through MCP tools. This allows AI assistants to query and analyze data during conversations without requiring manual SQL execution.

### Available Tools

1. **execute_query**
   - Execute SQL queries on the Postgres database
   - Parameters:
     - `query` (string): SQL query to execute
     - `schema` (string): Database schema to use (e.g., 'poco' or 'poco-test')

2. **list_tables**
   - List all tables in a specific schema
   - Parameters:
     - `schema` (string): Database schema to list tables from

3. **describe_table**
   - Describe the structure of a table (columns and constraints)
   - Parameters:
     - `schema` (string): Database schema
     - `table` (string): Name of the table to describe

4. **list_functions**
   - List all functions in a specific schema
   - Parameters:
     - `schema` (string): Database schema to list functions from

### Usage Example

1. Start the MCP server:

   ```bash
   python -m mcp.run_server
   ```

2. In Windsurf, the AI assistant can now use commands like:

   ```text
   I'll query the database to get information about PLZ details using the poco schema.
   [Uses execute_query tool with query "SELECT * FROM fn_plz_details('72358')" and schema "poco"]
   ```

## Security Considerations

- The MCP server has full access to your database with the credentials provided in environment variables
- Always ensure your `.env` file is properly secured and not committed to version control
- Consider using a read-only database user for the MCP server to prevent accidental data modification

## Implementation Notes

- Proper error handling is implemented to prevent sensitive error details from being exposed
- Query results are truncated to a maximum of 100 rows to prevent overwhelming responses
- All database operations properly use parameterized queries to prevent SQL injection
