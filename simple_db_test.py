"""Simple PostgreSQL connection test for POCO database.

This script tests the connection to the POCO PostgreSQL database
using the connection parameters from the .env file.
"""
import os
import sys
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get connection parameters from environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "c9ijs3l3qhrn1.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "uem4h7dfn2ghbi")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pc0f9dae381d06c5c45e0fd030b76f6fbe48d3d95d8cabb93551dc81fa5539f7e")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "d2tqio36qhkn50")
POSTGRES_URL = os.getenv("POSTGRES_URI", f"postgres://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}")

def test_connection():
    """Test the connection to the PostgreSQL database."""
    print("=" * 60)
    print("POCO PostgreSQL Connection Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Database host: {POSTGRES_HOST}")
    print(f"Database name: {POSTGRES_DATABASE}")
    
    try:
        # Try connection using connection string
        print("\nAttempting connection using URL...")
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT current_database(), current_user, version()")
        db, user, version = cursor.fetchone()
        print("✓ Connection successful!")
        print(f"✓ Connected to database: {db}")
        print(f"✓ Connected as user: {user}")
        print(f"✓ PostgreSQL version: {version[:50]}...")
        
        # Test access to 'poco' schema
        print("\nTesting access to 'poco' schema...")
        try:
            cursor.execute('SET search_path TO "poco";')
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'poco'"
            )
            table_count = cursor.fetchone()[0]
            print(f"✓ Access to 'poco' schema: {table_count} tables found")
            
            # List some tables if any exist
            if table_count > 0:
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'poco' LIMIT 5"
                )
                tables = cursor.fetchall()
                print(f"✓ Sample tables: {', '.join([t[0] for t in tables])}")
                
            # Test PLZ function if it exists
            print("\nTesting PLZ function...")
            try:
                cursor.execute('SELECT * FROM "poco".fn_plz_details(%s)', ('72358',))
                plz_results = cursor.fetchall()
                print(f"✓ PLZ function works! Found {len(plz_results)} results for '72358'")
            except Exception as e:
                print(f"✗ Error with PLZ function: {str(e)}")
                
        except Exception as e:
            print(f"✗ Error accessing 'poco' schema: {str(e)}")
        
        # Test access to 'poco-test' schema
        print("\nTesting access to 'poco-test' schema...")
        try:
            cursor.execute('SET search_path TO "poco-test";')
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'poco-test'"
            )
            test_table_count = cursor.fetchone()[0]
            print(f"✓ Access to 'poco-test' schema: {test_table_count} tables found")
        except Exception as e:
            print(f"✗ Error accessing 'poco-test' schema: {str(e)}")
        
        # Close connection
        conn.close()
        print("\n✓ Database connection test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Connection error: {str(e)}")
        
        # Try alternate connection method
        print("\nAttempting connection using individual parameters...")
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                database=POSTGRES_DATABASE
            )
            print("✓ Connection successful using individual parameters!")
            conn.close()
            return True
        except Exception as alt_e:
            print(f"✗ Alternative connection failed: {str(alt_e)}")
            return False

if __name__ == "__main__":
    test_connection()
    print("\nPress Enter to exit...")
    input()
