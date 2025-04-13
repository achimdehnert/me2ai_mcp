"""Simple test script for PostgreSQL connection."""
import os
import sys
from dotenv import load_dotenv

# Show Python version and execution path
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Script location: {__file__}")
print("=" * 60)

try:
    # Try to import required packages
    import psycopg2
    print("✓ Successfully imported psycopg2")
    
    from mcp.server import MCPServer
    print("✓ Successfully imported mcp package")
    
    # Load environment variables
    load_dotenv()
    print("✓ Loaded environment variables")
    
    # Get database connection parameters
    db_host = os.getenv("POSTGRES_HOST")
    db_port = os.getenv("POSTGRES_PORT")
    db_user = os.getenv("POSTGRES_USER")
    db_pass = os.getenv("POSTGRES_PASSWORD")
    db_name = os.getenv("POSTGRES_DATABASE")
    
    # Display connection parameters (without password)
    print("\nDatabase connection parameters:")
    print(f"Host: {db_host}")
    print(f"Port: {db_port}")
    print(f"User: {db_user}")
    print(f"Database: {db_name}")
    print(f"Password provided: {'Yes' if db_pass else 'No'}")
    
    # Try to connect to the database
    print("\nAttempting database connection...")
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_pass,
        database=db_name
    )
    
    # Check if connection was successful
    cursor = conn.cursor()
    cursor.execute("SELECT current_database(), current_user, version()")
    result = cursor.fetchone()
    
    print(f"✓ Successfully connected to database: {result[0]}")
    print(f"✓ Connected as user: {result[1]}")
    print(f"✓ PostgreSQL version: {result[2][:30]}...")
    
    # Try to access schemas
    for schema in ['poco', 'poco-test']:
        try:
            cursor.execute(f'SET search_path TO "{schema}";')
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables " +
                f"WHERE table_schema = '{schema}'"
            )
            count = cursor.fetchone()[0]
            print(f"✓ Access to '{schema}' schema: {count} tables found")
        except Exception as e:
            print(f"✗ Error accessing '{schema}' schema: {str(e)}")
    
    # Clean up
    conn.close()
    print("✓ Database connection closed")
    
except ImportError as e:
    print(f"✗ Import error: {str(e)}")
    print("  Please run: pip install psycopg2-binary modelcontextprotocol")
    
except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    print(traceback.format_exc())

print("\nTest completed.")
input("Press Enter to exit...")
