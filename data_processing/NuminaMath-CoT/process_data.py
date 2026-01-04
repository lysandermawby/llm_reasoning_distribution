#!/usr/bin/env python3
"""finds the word distribution of the text in the NuminaMath-CoT dataset. Note that this is human generated data and does not have a separation between thinking and non-thinking text"""

# package imports
import sys
from pathlib import Path

# add root directory to path for process_utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import click

# local imports
import process_utils

@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('-s', '--save-dict', help='save frequencies dictionaries', default=False, is_flag=True)
@click.option('-p', '--plot', help='create plots for frequencies', default=False, is_flag=True)
@click.option('-i', '--individual', help='generate individual graphs', default=False, is_flag=True)
def main(save_dict, plot, individual):
    """process data for NuminaMath-CoT dataset"""

    # raise an error if individual plots have been requested without plot flag
    assert not (individual and not plot), "You have entered 'individual' and not 'plot'. Individual graphs cannot be made unless the plot option is selected also"

    # defining the path to downloaded data
    data_path = Path("data") / "NuminaMath-CoT.jsonl"

    # defining analysis directory
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    # loading jsonl
    json_data = process_utils.load_jsonl(data_path)

    text_chunks = []
    for entry in json_data:
        solution = entry["solution"]
        text_chunks.append(solution)

    # prevent individual plots. option preserved for compatability with shell automation
    if individual:
        print("Warning: Setting individual=false as the data in NuminaMath-CoT does not have a distinction between regular and thinking text")
        individual=False

    # Process and analyze using shared utility function
    process_utils.process_and_analyze_text_chunks(
        text_chunks=text_chunks,
        dataset_name="NuminaMath-CoT",
        analysis_dir=analysis_dir,
        save_dict=save_dict,
        plot=plot,
        individual=individual
    )


if __name__ == '__main__':
    main()
