# config.py

# ========================================================
# 1. FILE SYSTEM PATHS
# ========================================================
# Path to your raw input data spreadsheet
DATA_PATH = "data/emails2.csv"

# Where to save your final learned network parameters
MODEL_SAVE_PATH = "src/neural_network/model_weights.pth"

# Path to the shared dictionary map reference
VOCAB_JSON_PATH = "src/neural_network/vocab/map.json"

# Path to the shared dictionary map reference
VOCAB_JSON_LEN = 85


# ========================================================
# 2. DATA PROCESSING HYPERPARAMETERS
# ========================================================
# Emails will be padded or cut off at exactly this many characters
MAX_EMAIL_LEN = 50 


# ========================================================
# 3. NEURAL NETWORK ARCHITECTURE SETTINGS
# ========================================================
# The column size of your character vector lookup table
EMBEDDING_DIM = 16  

# Number of sliding window filters for your 1D CNN layer
NUM_FILTERS = 64


# ========================================================
# 4. TRAINING ENGINE CONFIGURATIONS
# ========================================================
# Total number of times to cycle through all 20,000 emails
EPOCHS = 5

# Number of email matrices to feed the network simultaneously
BATCH_SIZE = 32

# The learning rate multiplier for your weight adjustments (AdamW)
LEARNING_RATE = 0.001