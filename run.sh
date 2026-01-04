#!/bin/sh

: << EOF
Downloading and processing all data directories in the data_processing dir
Runs the download script followed by the process script
By default, this also deletes the data downloaded while leaving the analysis
This script should be run after the setup.sh script
EOF

# defining colour variables (POSIX-safe)
RED="$(printf '\033[0;31m')"
GREEN="$(printf '\033[0;32m')"
NC="$(printf '\033[0m')"

# defining script version
VERSION="0.2.0"

# saving the initial directory
INITIAL_DIR="$(pwd)"

# return the user to the initial directory
trap 'cd "$INITIAL_DIR"' EXIT INT TERM

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_PROCESSING_NAME="data_processing"
DATA_PROCESSING_DIR="$SCRIPT_DIR/$DATA_PROCESSING_NAME"

DOWNLOAD_SCRIPT_NAME="download_data.py"
PROCESS_SCRIPT_NAME="process_data.py"

NUMBER=""
DELETE=false

show_help() {
    echo "run.sh - Downloads, processes, and analyses all datasets in $DATA_PROCESSING_NAME"
    echo ""
    echo "Usage: run.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -v, --version     Show the script version"
    echo "  -n, --number      Number of samples to download overriding script defaults"
    echo "  -d, --delete      Delete downloaded data after processing"
    echo ""
}

parse_arguments() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -h|--help)
                show_help >&2
                return 2
                ;;
            -v|--version)
                echo "Version: $VERSION"
                return 2
                ;;
            -d|--delete)
                DELETE=true
                shift
                ;;
            -n|--number)
                if [ -z "$2" ]; then
                    echo "${RED}Error: -n requires an argument${NC}"
                    return 1
                fi
                NUMBER="$2"
                shift 2
                ;;
            -*)
                echo "${RED}Error: Unrecognised argument $1${NC}"
                show_help >&2
                return 1
                ;;
            *)
                echo "${RED}Error: Positional arguments are not supported${NC}"
                return 1
                ;;
        esac
    done
    return 0
}

# check that required scripts exist in a directory
check_required_files() {
    dir="$1"
    [ -f "$dir/$DOWNLOAD_SCRIPT_NAME" ] &&
    [ -f "$dir/$PROCESS_SCRIPT_NAME" ]
}

delete_raw_data() {
    data_dir="$1"
    data_path="$data_dir/data"

    if [ -d "$data_path" ]; then
        rm -rf "$data_path"
        echo "${GREEN}Deleted raw data at $data_path${NC}"
    else
        echo "${RED}Warning: No data directory $data_path found${NC}"
    fi
}

main() {
    parse_arguments "$@"
    ret="$?"

    # 2 = clean exit (help/version)
    if [ "$ret" -eq 2 ]; then
        return 0
    fi

    if [ "$ret" -ne 0 ]; then
        return 1
    fi

    if [ ! -d "$DATA_PROCESSING_DIR" ]; then
        echo "${RED}Warning: $DATA_PROCESSING_DIR does not exist${NC}"
        return 1
    fi

    echo "Found data_processing directory"
    echo "Searching for data to download and process..."
    echo ""

    count=0

    for data_dir in "$DATA_PROCESSING_DIR"/*; do
        [ -d "$data_dir" ] || continue

        echo "${GREEN}Found data directory: $data_dir${NC}"
        count=$((count + 1))

        if check_required_files "$data_dir"; then
            # Change to dataset directory before running scripts
            cd "$data_dir" || {
                echo "${RED}Error: Could not change to $data_dir${NC}"
                continue
            }

            echo "Downloading files..."
            if [ -n "$NUMBER" ]; then
                uv run python "$DOWNLOAD_SCRIPT_NAME" -n "$NUMBER"
            else
                uv run python "$DOWNLOAD_SCRIPT_NAME"
            fi

            echo "Processing files..."
            uv run python "$PROCESS_SCRIPT_NAME" -p -i

            echo "${GREEN}Success: Downloaded and processed files${NC}"

            if [ "$DELETE" = true ]; then
                delete_raw_data "$data_dir"
            fi

            # Return to script directory
            cd "$SCRIPT_DIR" || {
                echo "${RED}Error: Could not return to script directory${NC}"
                return 1
            }
        else
            echo "${RED}Warning: Missing scripts in $data_dir${NC}"
            echo "Expected $DOWNLOAD_SCRIPT_NAME and $PROCESS_SCRIPT_NAME"
        fi

        echo ""
    done

    echo "${GREEN}Success: Processed $count datasets${NC}"
    return 0
}

main "$@"
