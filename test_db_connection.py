"""Test PostgreSQL database connection for ME2AI MCP server."""
import os
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("db-test")


def test_connection() -> bool:
    """Test connection to PostgreSQL database.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    # Load environment variables
    load_dotenv()
    
    # First try connection string
    connection_string = os.getenv("POSTGRES_URI") or os.getenv("POSTGRES_URL")
    if connection_string:
        try:
            logger.info("Attempting connection using connection string...")
            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()
            
            # Test query
            cursor.execute("SELECT current_database(), current_user, version()")
            db, user, version = cursor.fetchone()
            
            logger.info(f"Successfully connected to PostgreSQL database")
            logger.info(f"Database: {db}")
            logger.info(f"User: {user}")
            logger.info(f"Version: {version}")
            
            # Test schema access
            try:
                cursor.execute('SET search_path TO "poco";')
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables " 
                    "WHERE table_schema = 'poco' LIMIT 5"
                )
                tables = cursor.fetchall()
                if tables:
                    logger.info(f"Access to 'poco' schema confirmed")
                    logger.info(f"Sample tables: {', '.join([t[0] for t in tables])}")
                else:
                    logger.info("No tables found in 'poco' schema")
            except Exception as e:
                logger.warning(f"Could not access 'poco' schema: {str(e)}")
            
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Connection string method failed: {str(e)}")
    
    # Try individual parameters
    logger.info("Attempting connection using individual parameters...")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    user = os.getenv("POSTGRES_USER") 
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DATABASE")
    
    if not all([host, database, user, password]):
        logger.error("Missing required database connection parameters")
        return False
    
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT current_database(), current_user, version()")
        db, user, version = cursor.fetchone()
        
        logger.info(f"Successfully connected to PostgreSQL database")
        logger.info(f"Database: {db}")
        logger.info(f"User: {user}")
        logger.info(f"Version: {version}")
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Individual parameters method failed: {str(e)}")
        return False


if __name__ == "__main__":
    if test_connection():
        logger.info("Database connection test passed ✓")
    else:
        logger.error("Database connection test failed ✗")
