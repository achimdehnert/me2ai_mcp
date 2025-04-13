#!/usr/bin/env python
"""
POCO Dashboard: Datenbankverbindungen schließen
Dieses Skript beendet alle aktiven Datenbankverbindungen zur POCO Heroku-Datenbank.
"""
import os
import logging
import sys
import psycopg2
from dotenv import load_dotenv

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("connection_closer")

def close_all_connections():
    """Alle aktiven Datenbankverbindungen beenden."""
    # Umgebungsvariablen laden
    load_dotenv()
    
    # Verbindungsparameter aus Umgebungsvariablen
    connection_params = {
        'host': os.getenv('POSTGRES_HOST'),
        'port': os.getenv('POSTGRES_PORT'),
        'database': os.getenv('POSTGRES_DATABASE'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }
    
    try:
        # Verbindung zur Datenbank herstellen
        logger.info(f"Verbindung zur Datenbank {connection_params['host']}:{connection_params['port']}/{connection_params['database']} herstellen...")
        conn = psycopg2.connect(
            host=connection_params['host'],
            port=connection_params['port'],
            dbname=connection_params['database'],
            user=connection_params['user'],
            password=connection_params['password']
        )
        conn.autocommit = True
        
        # Aktive Verbindungen abfragen
        with conn.cursor() as cur:
            logger.info("Aktive Verbindungen abfragen...")
            cur.execute("""
                SELECT pid, usename, application_name, client_addr, backend_start, state, query
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
            """, (connection_params['database'],))
            active_connections = cur.fetchall()
            
            # Anzahl der aktiven Verbindungen ausgeben
            logger.info(f"{len(active_connections)} aktive Verbindungen gefunden.")
            
            if not active_connections:
                logger.info("Keine aktiven Verbindungen zum Beenden gefunden.")
                return
            
            # Alle aktiven Verbindungen beenden
            for pid, username, app_name, client_addr, backend_start, state, query in active_connections:
                logger.info(f"Beende Verbindung: PID={pid}, Benutzer={username}, App={app_name}, Status={state}")
                try:
                    # PG_TERMINATE_BACKEND beendet die Verbindung sofort
                    cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
                    
                except Exception as e:
                    logger.error(f"Fehler beim Beenden der Verbindung {pid}: {e}")
            
            # Überprüfung nach dem Beenden
            cur.execute("""
                SELECT count(*)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
            """, (connection_params['database'],))
            remaining = cur.fetchone()[0]
            logger.info(f"Nach dem Beenden verbleiben {remaining} aktive Verbindungen.")
    
    except Exception as e:
        logger.error(f"Fehler beim Verbinden zur Datenbank: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logger.info("Datenbankverbindung geschlossen.")
    
    return True

if __name__ == "__main__":
    logger.info("Starte Bereinigung von Datenbankverbindungen...")
    result = close_all_connections()
    if result:
        logger.info("Alle Verbindungen erfolgreich beendet.")
    else:
        logger.error("Fehler beim Bereinigen der Verbindungen.")
        sys.exit(1)
    sys.exit(0)
