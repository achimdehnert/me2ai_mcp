# ME2AI MCP Database Integration Guide

This guide provides detailed information on using the database integration features introduced in ME2AI MCP version 0.1.1.

## Overview

Version 0.1.1 introduces comprehensive database integration components with support for:

- PostgreSQL database connections and operations
- MySQL database connections and operations
- Flexible credential management
- LangChain-compatible database tools

## Installation

### Installing with Database Support

```bash
# Install with all database support
pip install me2ai_mcp[db]

# Install with PostgreSQL support only
pip install me2ai_mcp[postgres]

# Install with MySQL support only
pip install me2ai_mcp[mysql]

# Install with LangChain compatibility
pip install me2ai_mcp[langchain]

# Install with all features
pip install me2ai_mcp[all]
```

## Credential Management

The ME2AI MCP package provides a flexible credential management system that supports multiple credential sources:

### Environment Variables

```python
# PostgreSQL environment variables
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USERNAME=user
POSTGRES_PASSWORD=password

# MySQL environment variables
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USERNAME=user
MYSQL_PASSWORD=password
```

### JSON Credential Files

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "username": "user",
    "password": "password"
  },
  "connections": {
    "production": {
      "host": "prod-server",
      "port": 5432,
      "username": "prod-user",
      "password": "prod-password"
    },
    "development": {
      "host": "dev-server",
      "port": 5432,
      "username": "dev-user",
      "password": "dev-password"
    }
  }
}
```

## PostgreSQL Integration

### Basic Usage

```python
from me2ai_mcp.db.postgres import PostgreSQLConnection

# Create connection
conn = PostgreSQLConnection(
    env_prefix="POSTGRES",
    default_schema="public",
    allowed_schemas=["public", "schema1", "schema2"]
)

# Execute query
result = conn.execute_query(
    query="SELECT * FROM users WHERE id = %s",
    params=[1],
    schema="public"
)

# List tables in a schema
tables = conn.list_tables(schema="public")

# Get table columns
columns = conn.get_table_columns(
    table="users",
    schema="public"
)
```

### Using the PostgreSQL MCP Server

```bash
# Start the PostgreSQL MCP server
python postgres_mcp_server.py --default-schema public
```

## MySQL Integration

### Basic Usage

```python
from me2ai_mcp.db.mysql import MySQLConnection

# Create connection
conn = MySQLConnection(
    env_prefix="MYSQL",
    default_schema="my_database",
    allowed_schemas=["my_database", "another_db"],
    pool_size=5
)

# Execute query
result = conn.execute_query(
    query="SELECT * FROM users WHERE id = %s",
    params=[1],
    schema="my_database"
)

# List tables in a schema
tables = conn.list_tables(schema="my_database")

# Get table columns
columns = conn.get_table_columns(
    table="users",
    schema="my_database"
)

# Get database information
db_info = conn.get_database_info()
```

### Using the MySQL MCP Server

```bash
# Start the MySQL MCP server
python mysql_mcp_server.py --default-schema my_database
```

## LangChain Integration

The ME2AI MCP package provides LangChain-compatible tools for database operations:

```python
from me2ai_mcp.integrations.langchain import LangChainToolFactory
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

# Get PostgreSQL tools
postgres_tools = LangChainToolFactory.create_postgres_tools()

# Get MySQL tools
mysql_tools = LangChainToolFactory.create_mysql_tools()

# Get database tools based on type
db_tools = LangChainToolFactory.create_database_tools(db_type="mysql")

# Initialize LangChain agent with tools
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=db_tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Run the agent
agent.run("List all tables in the database")
```

## MCP Tools

### PostgreSQL Tools

The package provides the following PostgreSQL tools:

- `ExecuteQueryTool`: Execute SQL queries on PostgreSQL databases
- `ListTablesTool`: List tables in a PostgreSQL schema
- `GetTableColumnsTool`: Get column information for a PostgreSQL table
- `GetPLZDetailsTool`: Get postal code (PLZ) details

### MySQL Tools

The package provides the following MySQL tools:

- `ExecuteQueryTool`: Execute SQL queries on MySQL databases
- `ListTablesTool`: List tables in a MySQL schema
- `GetTableColumnsTool`: Get column information for a MySQL table
- `GetDatabaseInfoTool`: Get MySQL database server information

## Advanced Configuration

### Connection Pooling

Both PostgreSQL and MySQL connections support connection pooling:

```python
# PostgreSQL with connection pooling
postgres_conn = PostgreSQLConnection(
    min_connections=1,
    max_connections=10,
    connection_timeout=5,
    idle_timeout=60
)

# MySQL with connection pooling
mysql_conn = MySQLConnection(
    pool_size=5,
    pool_name="my_connection_pool",
    connect_timeout=10
)
```

### Multiple Named Connections

```python
# Using a named connection from a JSON credentials file
conn = PostgreSQLConnection(
    credential_file="credentials.json",
    connection_name="production"
)
```

## Security Best Practices

- Store credentials in environment variables or secure credential files
- Use connection-specific schemas with limited permissions
- Apply the principle of least privilege
- Sanitize query parameters
- Use connection pooling to manage resources effectively
- Set timeouts to prevent hanging connections

## Error Handling

The package provides custom exception classes for specific error scenarios:

- `PostgreSQLError` / `MySQLError`: Base class for database errors
- `ConnectionError`: Error establishing a database connection
- `QueryError`: Error executing a database query
- `SchemaError`: Error related to database schema operations

Example error handling:

```python
from me2ai_mcp.db.postgres import PostgreSQLConnection, QueryError, SchemaError

try:
    conn = PostgreSQLConnection()
    result = conn.execute_query("SELECT * FROM users")
except SchemaError as e:
    print(f"Schema error: {e}")
except QueryError as e:
    print(f"Query error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing

Run the comprehensive database test suite:

```bash
# Run all database tests
python run_tests.py --module db

# Run with coverage
python run_tests.py --module db --coverage

# Generate HTML report
python run_tests.py --module db --html-report
```
