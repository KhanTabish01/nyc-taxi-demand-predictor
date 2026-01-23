import requests
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
YEAR = 2024
MONTHS = range(1, 13)  # January through December
BASE_URL = 'https://d37ci6vzurychx.cloudfront.net'
OUTPUT_DIR = Path('data/raw') / str(YEAR)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Output directory: {OUTPUT_DIR}")

# Download each month
for month in MONTHS:
    month_str = f'{month:02d}'  # Format as 01, 02, ..., 12
    file_name = f'yellow_tripdata_{YEAR}-{month_str}.parquet'
    file_url = f'{BASE_URL}/trip-data/{file_name}'
    output_path = OUTPUT_DIR / file_name
    
    logger.info(f'Downloading {file_name}...')
    
    try:
        # Stream the download
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Raise error for bad status codes
        
        # Get file size from headers
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Download with progress bar
        with open(output_path, 'wb') as file, \
             tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=file_size, desc=file_name) as progress:
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)
                    progress.update(len(chunk))
        
        # Verify file was downloaded
        actual_size = output_path.stat().st_size
        if actual_size == 0:
            logger.error(f'✗ Downloaded file {file_name} is empty!')
        else:
            logger.info(f'✓ Successfully downloaded {file_name} ({actual_size:,} bytes)')
    
    except requests.exceptions.RequestException as e:
        logger.error(f'✗ Failed to download {file_name}: {e}')
    except Exception as e:
        logger.error(f'✗ Unexpected error downloading {file_name}: {e}')

logger.info('Download complete.')