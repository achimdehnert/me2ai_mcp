"""Debug PostgreSQL connection issues."""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print important debug information
print("=" * 50)
print("DATABASE CONNECTION DEBUG")
print("=" * 50)

# Database connection parameters
postgres_uri = os.getenv("POSTGRES_URI")
postgres_url = os.getenv("POSTGRES_URL")
postgres_host = os.getenv("POSTGRES_HOST")
postgres_port = os.getenv("POSTGRES_PORT")
postgres_user = os.getenv("POSTGRES_USER")
postgres_db = os.getenv("POSTGRES_DATABASE")
postgres_pwd = os.getenv("POSTGRES_PASSWORD")

print(f"URI available: {'Yes' if postgres_uri else 'No'}")
print(f"URL available: {'Yes' if postgres_url else 'No'}")
print(f"Host: {postgres_host}")
print(f"Port: {postgres_port}")
print(f"User: {postgres_user}")
print(f"Database: {postgres_db}")
print(f"Password provided: {'Yes' if postgres_pwd else 'No'}")

# Try connecting
try:
    import psycopg2
    print("\nAttempting connection...")
    
    if postgres_uri:
        conn = psycopg2.connect(postgres_uri)
        print("✓ Connected using URI!")
    else:
        conn = psycopg2.connect(
            host=postgres_host,
            port=postgres_port,
            user=postgres_user,
            password=postgres_pwd,
            database=postgres_db
        )
        print("✓ Connected using parameters!")
    
    cursor = conn.cursor()
    cursor.execute("SELECT current_database(), current_user")
    db, user = cursor.fetchone()
    print(f"Connected to database: {db} as user: {user}")
    
    # Try accessing poco schema
    try:
        cursor.execute('SET search_path TO "poco";')
        cursor.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'poco'")
        table_count = cursor.fetchone()[0]
        print(f"✓ Found {table_count} tables in 'poco' schema")
    except Exception as e:
        print(f"✗ Error accessing 'poco' schema: {str(e)}")
    
    conn.close()
    
except Exception as e:
    print(f"✗ Connection error: {str(e)}")

print("\n" + "=" * 50)
