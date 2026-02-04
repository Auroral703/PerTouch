import os

class Config:
    # Path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MEMORY_DIR = os.path.join(BASE_DIR, 'memory_data')
    MEMORY_FILE = os.path.join(MEMORY_DIR, 'users_memory.json')
    SAM_CKPT = "../model/sam3/sam3.pt"
    
    # Image Parameter
    IMG_SIZE = (512, 512)
    CHANNELS = 4  # Colorfulness, Contrast, Temperature, Brightness
    
    # Parameter Mapping Channel Index
    IDX_COLORFULNESS = 0
    IDX_CONTRAST = 1
    IDX_TEMPERATURE = 2
    IDX_BRIGHTNESS = 3
    
    # VLM Configuration
    MAX_RETHINK_RETRIES = 3  # Maximum number of retries for strong instructions
    API_KEY = ""
    API_URL = ""
    VLM_MODEL_NAME = ""
