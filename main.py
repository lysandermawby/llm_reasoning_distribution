#!/usr/bin/env python3
"""Main entry point for downloading and processing LLM reasoning datasets

This provides a Python CLI interface to the dataset processing pipeline.
For shell script automation, use run.sh instead.
"""

# package imports
import click
import subprocess
import sys
from pathlib import Path


def find_available_datasets():
    """find all datasets that have download_data.py and process_data.py scripts"""
    data_processing_dir = Path(__file__).parent / "data_processing"

    if not data_processing_dir.exists():
        return []

    datasets = []
    for dataset_dir in data_processing_dir.iterdir():
        if dataset_dir.is_dir():
            download_script = dataset_dir / "download_data.py"
            process_script = dataset_dir / "process_data.py"
            if download_script.exists() and process_script.exists():
                datasets.append(dataset_dir.name)

    return sorted(datasets)


def run_dataset_pipeline(dataset_name: str, num_samples: int = None, plot: bool = False,
                         individual: bool = False, delete_data: bool = False):
    """run the download and process pipeline for a specific dataset"""
    dataset_dir = Path(__file__).parent / "data_processing" / dataset_name

    if not dataset_dir.exists():
        click.echo(f"Error: Dataset directory '{dataset_name}' not found", err=True)
        return False

    download_script = dataset_dir / "download_data.py"
    process_script = dataset_dir / "process_data.py"

    if not (download_script.exists() and process_script.exists()):
        click.echo(f"Error: Required scripts not found in '{dataset_name}'", err=True)
        return False

    # Change to dataset directory
    original_dir = Path.cwd()

    # default to the original directory if an error is encountered
    try:
        import os
        os.chdir(dataset_dir)

        # Run download script using uv
        click.echo(f"\n{click.style('Downloading', fg='green')} {dataset_name}...")
        download_cmd = ["uv", "run", "python", "download_data.py"]
        if num_samples is not None:
            download_cmd.extend(["-n", str(num_samples)])

        result = subprocess.run(download_cmd, capture_output=False)
        if result.returncode != 0:
            click.echo(f"{click.style('Error:', fg='red')} Download failed for {dataset_name}", err=True)
            return False

        # Run process script using uv
        click.echo(f"\n{click.style('Processing', fg='green')} {dataset_name}...")
        process_cmd = ["uv", "run", "python", "process_data.py"]
        if plot:
            process_cmd.append("-p")
        if individual:
            process_cmd.append("-i")

        result = subprocess.run(process_cmd, capture_output=False)
        if result.returncode != 0:
            click.echo(f"{click.style('Error:', fg='red')} Processing failed for {dataset_name}", err=True)
            return False

        # Delete raw data if requested
        if delete_data:
            data_dir = dataset_dir / "data"
            if data_dir.exists():
                import shutil
                shutil.rmtree(data_dir)
                click.echo(f"{click.style('Deleted', fg='green')} raw data at {data_dir}")

        click.echo(f"\n{click.style('Success:', fg='green')} Completed {dataset_name}")
        return True

    finally:
        os.chdir(original_dir)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-l', '--list', 'list_datasets', is_flag=True, help='List all available datasets')
@click.option('-d', '--dataset', help='Process a specific dataset')
@click.option('-a', '--all', 'process_all', is_flag=True, help='Process all available datasets')
@click.option('-n', '--number', type=int, help='Number of samples to download')
@click.option('-p', '--plot', is_flag=True, help='Create frequency plots')
@click.option('-i', '--individual', is_flag=True, help='Create individual plots (requires --plot)')
@click.option('--delete', 'delete_data', is_flag=True, help='Delete raw data after processing')
def main(list_datasets, dataset, process_all, number, plot, individual, delete_data):
    """Main entry point for LLM reasoning distribution analysis

    This tool downloads and analyzes word frequency distributions in LLM reasoning datasets, comparing thinking tokens vs regular output tokens
    """

    # List available datasets
    if list_datasets:
        datasets = find_available_datasets()
        if datasets:
            click.echo(f"\n{click.style('Available datasets:', fg='green', bold=True)}")
            for ds in datasets:
                click.echo(f"  - {ds}")
            if len(datasets) == 1:
                click.echo(f"\nTotal: {len(datasets)} dataset")
            else:
                click.echo(f"\nTotal: {len(datasets)} datasets")
        else:
            click.echo(click.style("No datasets found", fg='yellow'))
        return

    # Validate individual flag
    if individual and not plot:
        click.echo(click.style("Error: --individual requires --plot", fg='red'), err=True)
        sys.exit(1)

    # Process specific dataset
    if dataset:
        success = run_dataset_pipeline(dataset, number, plot, individual, delete_data)
        sys.exit(0 if success else 1)

    # Process all datasets
    if process_all:
        datasets = find_available_datasets()
        if not datasets:
            click.echo(click.style("No datasets found", fg='yellow'))
            sys.exit(1)

        click.echo(f"\n{click.style(f'Processing {len(datasets)} dataset(s)...', fg='green', bold=True)}\n")

        successes = 0
        failures = []

        for ds in datasets:
            if run_dataset_pipeline(ds, number, plot, individual, delete_data):
                successes += 1
            else:
                failures.append(ds)

        # Summary
        click.echo(f"\n{click.style('='*60, fg='cyan')}")
        click.echo(f"{click.style('Summary:', fg='cyan', bold=True)}")
        click.echo(f"  Successful: {click.style(str(successes), fg='green')}/{len(datasets)}")
        if failures:
            click.echo(f"  Failed: {click.style(', '.join(failures), fg='red')}")
        click.echo(f"{click.style('='*60, fg='cyan')}\n")

        sys.exit(0 if not failures else 1)

    # No action specified
    click.echo("No action specified. Use --help for usage information.")


if __name__ == "__main__":
    main()
