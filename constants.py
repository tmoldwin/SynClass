# Constants for SynClass project
import os
import logging
from datetime import datetime

# Data paths
DATA_DIR = 'Data/synpase_raw_em/'
CSV_PATH = 'Data/synpase_raw_em/synapse_data.csv'
# Fallback: CSV at Data root when using 7z (e.g. Data/synapse_data.csv)
CSV_PATH_FALLBACK = 'Data/synapse_data.csv'
# Archive for loading without extraction (zip or 7z)
DATA_ARCHIVE = 'Data/synpase_raw_em.7z'

# Model save paths
MODEL_SAVE_PATHS = {
    'vgg2d': 'saved_models/best_synapse_model_vgg2d.pth',
    'resnet': 'saved_models/best_synapse_model_resnet.pth',
    'masked': 'saved_models/best_synapse_model_masked.pth',
    'fast': 'saved_models/best_synapse_model_fast.pth',
    '2dcnn': 'saved_models/best_synapse_model_2dcnn.pth',
    'default': 'saved_models/best_synapse_model.pth'
}

# Ensure directories exist
os.makedirs('saved_models', exist_ok=True)
os.makedirs('result_logs', exist_ok=True)

def setup_logging(model_name):
    """Setup logging to both file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"result_logs/{model_name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will output to console
        ]
    )
    
    return logging.getLogger() 