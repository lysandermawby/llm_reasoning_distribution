#!/bin/zsh

: << EOF
Downloading and processing all data directores in the data_processing dir
Runs the download script followed by the process script
By default, this also deletes the data downloaded while leaving the analysis as the script progresses to avoid using up too much disk space
This script should be run after the setup.sh script 
EOF

# defining colour variables
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# defining script version
VERSIon="0.1.0"

# saving the initial directory, to return the shell session to it after execution
INITIAL_DIR="$(pwd)"

trap "cd '$INITIAL_DIR'" EXIT INT TERM

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_PROCESSING_NAME="data_processing"
DATA_PROCESSING_DIR="$SCRIPT_DIR/$DATA_PROCESSING_NAME"

DOWNLOAD_SCRIPT_NAME="download_data.py"
PROCESS_SCRIPT_NAME="process_data.py"

NUMBER=""
DELETE='false'

show_help() {
    echo "run.sh - Downloads, processes, and analyses all datasets specified in the $DATA_PROCESSING_NAME directory"
    echo ""
    echo "Usage: run.sh <options>"
    echo ""
    echo "Options:"
    echo "      -h|--help       Show this help message"
    echo "      -v|--version    Show the script version"
    echo "      -n|--number     The number of samples from each dataset to be downloaded"
    echo "      -d|--delete     Delete downloaded data once processed to free up disk space"
    echo ""
    echo "Note that without the -n argument, the number of samples is given by the default values in the download scripts $DOWNLOAD_SCRIPT_NAME"
}

parse_arguments() {
    POSITIONAL_COUNT=1
    while [ $# -gt 0 ]; do
        case "$1" in 
            -h|--help)
                show_help 2>&1
                return 1
                shift 2
                ;;
            -v|--version)
                echo "Version: $VERSION"
                return 1
                shift 2
                ;;
            -d|--delete)
                DELETE='true'
                shift 2
                ;;
            -n|--number)
                NUMBER="$2"
                shift 2
                ;;
            -*)
                echo "${RED}Error: Unrecognised argument $1${NC}"
                show_help 2>&1
                return 1
                shift 2
                ;;
            *)
                POSITIONAL_COUNT=$((POSITIONAL_COUNT+1))
                echo "${RED}Error: Positional argument found${NC}"
                return 1
                shift
                ;;
        esac
    done
    return 0
}

# list all file names in a directory
list_file_names() {
    DIRECTORY="$1"
    filename_arr=()
    for file in "$data_dir"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            filename_arr+=("$filename")
        fi
    done
}

# see whether the download and process scripts are in the directory
files_in_arr() {
    filename_arr=("$@")
    if (( ${filename_arr[(Ie)$DOWNLOAD_SCRIPT_NAME]} )) && (( ${filename_arr[(Ie)$PROCESS_SCRIPT_NAME]} )); then
        return 0
    else
        return 1
    fi
}

# deleting raw data
delete_raw_data() {
    data_dir="$1"
    data_name="$data_dir/data"
    if [ -d "$data_name" ]; then
        rm -rf "$data_name"
        echo "${GREEN}Deleted raw data at $data_name${NC}"
    else
        echo "${RED}Warning: No data directory $data_name found${NC}"
    fi
}

# main script logic
main() {
    # parse command line arguments
    parse_arguments "$@"
    local ret="$?"

    # parse_arguments returns 0 for normal functioning, 2 if the script should not continue but there has been no error, and 1 if there has been an error
    if [ $ret -eq 0 ]; then

        # Check if the directory exists
        if [ -d "$DATA_PROCESSING_DIR" ]; then
            echo "Found data_processing directory"
            echo "Searching for data to download, process, and analyse...\n"
            
            # running the download and processing scripts in the respective scripts
            count=0
            for data_dir in "$DATA_PROCESSING_DIR"/*; do # note that * in zsh will not match hidden files
                if [ -d "$data_dir" ]; then
                    echo "${GREEN}Found data directory: $data_dir${NC}"
                    count=$((count + 1))
                    # find all files in data_dir
                    list_file_names "$data_dir"
                    # echo ${filename_arr[@]}
                    files_in_arr "${filename_arr[@]}"
                    is_in_arr="$?"
                    if [ $is_in_arr -eq 0 ]; then
                        echo "Expected download and processing scripts were found in the ${basename data_dir} directory\n"
                        download_script_path="$data_dir/$DOWNLOAD_SCRIPT_NAME"
                        process_script_path="$data_dir/$PROCESS_SCRIPT_NAME"
                        echo "Downloading the files..."
                        if [ -n "$NUMBER" ]; then
                            uv run python "$download_script_path" -n "$NUMBER"
                        else
                            uv run python "$download_script_path"  # Use default from download script
                        fi
                        echo "Processing the files..."
                        uv run python "$process_script_path"
                        echo ""
                        echo "${GREEN}Success: Downloaded and processed the files${NC}"
                        if [ "$DELETE" = 'true' ]; then
                            delete_raw_data "$data_dir"
                        fi
                    else
                        echo "${RED}Warning: Did not find appropriate scripts in $data_dir${NC}"
                        echo "Did not find the download script $DOWNLOAD_SCRIPT_NAME and the process script $PROCESS_SCRIPT_NAME. Skipping the processing of directory $data_dir..."
                    fi
                else
                    echo "Non-directory file found: $data_dir"
                fi
            done
            
            echo "${GREEN}Success: Data processing complete! Found a total of $count repositories to process. ${NC}"
        else
            echo "${RED}Warning: $DATA_PROCESSING_DIR does not exist${NC}"
        fi
    fi

    [[ $ret -eq 2 || $ret -eq 0 ]] || return 0
    return 1
}

main "$@"
