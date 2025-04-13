"""Debug script for PostgreSQL connection testing.

This script tests the connection to the PostgreSQL database and prints
detailed diagnostic information, focusing on the correct credentials.
"""
import os
import sys
import traceback
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

def debug_postgres_connection():
    """Test PostgreSQL connection with detailed debug output."""
    # Print header
    print("=" * 60)
    print("PostgreSQL Connection Debug")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Script location: {os.path.abspath(__file__)}")
    print("=" * 60)
    
    # Load environment variables
    print("Loading environment variables...")
    load_dotenv()
    
    # Print connection parameters (hiding password)
    host = os.getenv("POSTGRES_HOST", "Not set")
    port = os.getenv("POSTGRES_PORT", "Not set")
    user = os.getenv("POSTGRES_USER", "Not set")
    password = "********" if os.getenv("POSTGRES_PASSWORD") else "Not set"
    database = os.getenv("POSTGRES_DATABASE", "Not set")
    uri = os.getenv("POSTGRES_URI", "Not set").replace(os.getenv("POSTGRES_PASSWORD", ""), "********")
    
    print(f"POSTGRES_HOST: {host}")
    print(f"POSTGRES_PORT: {port}")
    print(f"POSTGRES_USER: {user}")
    print(f"POSTGRES_PASSWORD: {password}")
    print(f"POSTGRES_DATABASE: {database}")
    print(f"POSTGRES_URI: {uri}")
    print("-" * 60)
    
    # Test connection using URI
    print("\nAttempting connection using URI...")
    try:
        conn_uri = os.getenv("POSTGRES_URI")
        if conn_uri:
            conn = psycopg2.connect(conn_uri)
            cursor = conn.cursor()
            cursor.execute("SELECT current_database(), current_user, version()")
            db, db_user, version = cursor.fetchone()
            print("✓ Connection successful using URI!")
            print(f"✓ Connected to database: {db}")
            print(f"✓ Connected as user: {db_user}")
            print(f"✓ PostgreSQL version: {version[:50]}...")
            
            # Test schemas
            print("\nTesting schema access:")
            for schema in ["poco", "poco-test"]:
                try:
                    cursor.execute(f'SET search_path TO "{schema}";')
                    cursor.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        f"WHERE table_schema = '{schema}'"
                    )
                    table_count = cursor.fetchone()[0]
                    print(f"✓ Access to '{schema}' schema: {table_count} tables found")
                    
                    if table_count > 0:
                        cursor.execute(
                            "SELECT table_name FROM information_schema.tables "
                            f"WHERE table_schema = '{schema}' LIMIT 5"
                        )
                        tables = cursor.fetchall()
                        print(f"  Sample tables: {', '.join([t[0] for t in tables])}")
                except Exception as e:
                    print(f"✗ Error accessing '{schema}' schema: {str(e)}")
            
            # Test PLZ function
            print("\nTesting PLZ function...")
            try:
                cursor.execute('SELECT * FROM "poco".fn_plz_details(%s) LIMIT 1', ('72358',))
                plz_results = cursor.fetchall()
                print(f"✓ PLZ function works! Found {len(plz_results)} results for '72358'")
                if plz_results:
                    print(f"  Sample column names: {[desc[0] for desc in cursor.description]}")
            except Exception as e:
                print(f"✗ Error with PLZ function: {str(e)}")
            
            conn.close()
            print("\n✓ Database test completed successfully")
            return True
        else:
            print("✗ POSTGRES_URI environment variable not set")
    except Exception as e:
        print(f"✗ URI connection error: {str(e)}")
        traceback.print_exc()
    
    # Test connection using individual parameters
    print("\nAttempting connection using individual parameters...")
    try:
        if all([host, port, user, os.getenv("POSTGRES_PASSWORD"), database]):
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=os.getenv("POSTGRES_PASSWORD"),
                database=database
            )
            print("✓ Connection successful using individual parameters!")
            conn.close()
            return True
        else:
            print("✗ One or more connection parameters are missing")
    except Exception as e:
        print(f"✗ Parameter connection error: {str(e)}")
        traceback.print_exc()
    
    print("\n✗ All connection attempts failed")
    return False

if __name__ == "__main__":
    success = debug_postgres_connection()
    print("\nDebug process completed with " + ("success" if success else "failure"))
    print("\nPress Enter to exit...")
    input()
