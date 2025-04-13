"""Check database connection parameters."""
import os
from dotenv import load_dotenv

def main():
    """Print database connection parameters from environment variables."""
    load_dotenv()
    
    print("Database connection parameters:")
    print(f"Host: {os.getenv('POSTGRES_HOST')}")
    print(f"Database: {os.getenv('POSTGRES_DATABASE')}")
    print(f"Port: {os.getenv('POSTGRES_PORT')}")
    print(f"User: {os.getenv('POSTGRES_USER')}")
    print(f"Password: {'*' * 8 if os.getenv('POSTGRES_PASSWORD') else 'Not set'}")
    print(f"Schema: poco, poco-test")

if __name__ == "__main__":
    main()
