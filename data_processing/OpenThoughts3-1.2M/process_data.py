#!/usr/bin/env python3
"""finds the word distribution of the thinking and non-thinking text in the OpenThoughts3-1.2M dataset"""

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
    """process data for OpenThoughts3-1.2M dataset"""

    # raise an error if individual plots have been requested without plot flag
    assert not (individual and not plot), "You have entered 'individual' and not 'plot'. Individual graphs cannot be made unless the plot option is selected also"

    # defining the path to downloaded data
    data_path = Path("data") / "OpenThoughts3-1.2M.jsonl"

    # defining analysis directory
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    # loading jsonl
    json_data = process_utils.load_jsonl(data_path)

    # extracting the chunks of llm-generated text (dataset-specific logic)
    text_chunks = []
    for entry in json_data:
        conversation = entry["conversations"]
        for conv in conversation:
            if conv['from'] == 'gpt':
                text_chunks.append(conv['value'])

    # Process and analyze using shared utility function
    process_utils.process_and_analyze_text_chunks(
        text_chunks=text_chunks,
        dataset_name="OpenThoughts3-1.2M",
        analysis_dir=analysis_dir,
        save_dict=save_dict,
        plot=plot,
        individual=individual
    )


if __name__ == '__main__':
    main()
