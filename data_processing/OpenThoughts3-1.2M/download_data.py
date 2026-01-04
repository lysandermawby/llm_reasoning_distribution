#!/usr/bin/env python3
"""downloading the OpenThoughts3 dataset"""

# Disable HuggingFace's background processes and caching to ensure clean exit
import os
os.environ['HF_DATASETS_OFFLINE'] = '0'  # Allow downloads but don't cache aggressively
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'  # Disable telemetry
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism

# package imports
import sys
from pathlib import Path
import click

# add root directory to path for download_utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# local imports
import download_utils


@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('-n', '--number', type=int, help='number of samples to download', default=100000)
@click.option('-a', '--download-all', 'download_all', is_flag=True, default=False, help='download all samples')
def main(number, download_all):
    """download OpenThoughts3-1.2M dataset from HuggingFace"""

    # Set number to None if downloading all samples
    if download_all:
        number = None

    # Download and save dataset using shared utility function
    if download_all:
        click.echo("Downloading all samples...")
    else:
        click.echo(f"Downloading {number} samples...")

    download_utils.download_hf_dataset_streaming(
        hf_repo_name="open-thoughts/OpenThoughts3-1.2M",
        output_filename="OpenThoughts3-1.2M.jsonl",
        data_dir=Path("data"),
        n=number
    )

    # Force immediate exit - os._exit() doesn't wait for background threads
    # This is safe here because all data is saved and file is closed
    os._exit(0)


if __name__ == '__main__':
    main()
