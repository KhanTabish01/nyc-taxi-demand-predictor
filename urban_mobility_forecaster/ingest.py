import sqlite3
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path('data/processed/taxi.db')
RAW_DATA_DIR = Path('data/raw/2024')
COLUMNS_TO_KEEP = [
    'tpep_pickup_datetime',
    'PULocationID',
    'DOLocationID',
    'trip_distance',
    'passenger_count'
]
EXPECTED_ROWS = None  # TODO: Calculate based on downloaded files

def create_table_and_indexes(conn):
    """Create yellow_trips table and indexes if they don't exist."""
    cursor = conn.cursor()
    
    # TODO: Write CREATE TABLE statement
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS yellow_trips (
        id INTEGER PRIMARY KEY,
        tpep_pickup_datetime DATETIME NOT NULL,
        PULocationID TEXT NOT NULL,
        DOLocationID TEXT,
        trip_distance FLOAT,
        passenger_count INTEGER,
        loaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    cursor.execute(create_table_sql)
    
    # Create indexes on frequently queried columns
    index_sql_1 = "CREATE INDEX IF NOT EXISTS idx_pu_location ON yellow_trips (PULocationID)"
    index_sql_2 = "CREATE INDEX IF NOT EXISTS idx_pickup_datetime ON yellow_trips (tpep_pickup_datetime)"
    
    cursor.execute(index_sql_1)
    cursor.execute(index_sql_2)
    
    conn.commit()
    logger.info("Table and indexes created/verified.")

def ingest_data(year: int = 2024, validate: bool = True) -> dict:
    """
    Ingest NYC Yellow Taxi Parquet files into SQLite.
    
    Args:
        year: Year of data to ingest (default: 2024)
        validate: Whether to validate row counts (default: True)
    
    Returns:
        dict: Status report with keys: success (bool), message (str), row_count (int)
    """
    
    try:
        # Step 1: Connect to database
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        logger.info(f"Connected to database: {DB_PATH}")
        
        # Step 2: Create table if needed
        create_table_and_indexes(conn)
        
        # Step 3: Check if data already exists
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM yellow_trips")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            logger.info(f"Data already exists: {existing_count} rows. Skipping ingest.")
            return {
                'success': True,
                'message': 'Data already loaded',
                'row_count': existing_count
            }
        
        # Step 4: Find all Parquet files
        parquet_files = sorted(RAW_DATA_DIR.glob('yellow_tripdata_*.parquet'))
        
        if not parquet_files:
            return {
                'success': False,
                'message': f'No Parquet files found in {RAW_DATA_DIR}',
                'row_count': 0
            }
        
        logger.info(f"Found {len(parquet_files)} Parquet files")
        
        # Step 5: Load and ingest each file
        total_rows_loaded = 0
        for parquet_file in parquet_files:
            logger.info(f"Loading file: {parquet_file.name}")
            df = pd.read_parquet(parquet_file, columns=COLUMNS_TO_KEEP)
            
            # Ensure tpep_pickup_datetime is datetime type
            df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            
            # Append to SQLite (auto-increment id, loaded_at timestamp handled by DB)
            df.to_sql('yellow_trips', conn, if_exists='append', index=False)
            
            # Track total rows loaded
            total_rows_loaded += len(df)
            
            logger.info(f"✓ Loaded {len(df):,} rows from {parquet_file.name}")
        
        conn.commit()
        conn.close()
        
        # Step 6: Validate
        if validate:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM yellow_trips")
            final_count = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"Final row count: {final_count:,}")
            
            # Validate row count matches what we loaded
            if final_count != total_rows_loaded:
                logger.warning(f"Row count mismatch! Loaded: {total_rows_loaded:,}, DB has: {final_count:,}")
            else:
                logger.info("✓ Row count validation passed")
        
        logger.info("Ingest complete!")
        return {
            'success': True,
            'message': f'Successfully ingested {total_rows_loaded} rows',
            'row_count': total_rows_loaded
        }
    
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        return {
            'success': False,
            'message': str(e),
            'row_count': 0
        }

if __name__ == "__main__":
    result = ingest_data()
    logger.info(f"Result: {result}")
