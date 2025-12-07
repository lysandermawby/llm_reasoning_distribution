#!/bin/sh

: << EOF
Deleting the data downloaded and stored in these directories
Leaves the code and the analysis directories alone
EOF

# defining colour variables
RED='\033[0;31m'
NC='\033[0m'

# saving the initial directory, to return the shell session to it after execution
INITIAL_DIR="$(pwd)"

trap "cd '$INITIAL_DIR'" EXIT INT TERM

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_PROCESSING_DIR="$SCRIPT_DIR/data_processing"

# Check if the directory exists
if [ -d "$DATA_PROCESSING_DIR" ]; then
    echo "Found data_processing directory"
    echo "Searching for data_processing/*/data directories to remove..."
    
    # Remove all data_processing/*/data directories
    for data_dir in "$DATA_PROCESSING_DIR"/*/data; do
        if [ -d "$data_dir" ]; then
            echo "Removing: $data_dir"
            rm -rf "$data_dir"
        fi
    done
    
    echo "Removal complete"
else
    echo "${RED}Warning: $DATA_PROCESSING_DIR does not exist${NC}"
fi
