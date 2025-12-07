#!/usr/bin/env python3
"""finds the word distribution of the thinking and non-thinking text"""

# package imports
import json
from pathlib import Path
from tqdm import tqdm
import re
from typing import List, Dict
import matplotlib.pyplot as plt
import math
import click
from scipy.optimize import curve_fit
import numpy as np


"""
TODO:

Combine the distributions into a single plot. Show the Zipfian coefficient for each of them and their respective fits
Hypothesis is that for models which have undergone more RL / are more speaking in neuralese, we expect a different Zipfian coefficient
"""


def find_word_freq(text: List[str]) -> Dict:
    """
    find the frequency of different words
    
    Input: text
        List of strings

    Output: frequency_dict
        dictionary with words as keys and counts as values
    """

    frequency_dict = {}
    for section in text:
        for word in section.split():
            frequency_dict[word] = frequency_dict.get(word, 0) + 1

    return frequency_dict


def frequency_historam(frequency_dict, saved_path):
    """saving the data as a frequency histogram"""
    counts = [[key, value] for key, value in frequency_dict.items()]
    sorted_counts = sorted(counts, key=lambda x: x[1])[::-1]
    # words_sorted = [x[0] for x in sorted_counts]
    frequencies = [x[1] for x in sorted_counts]

    # creating plot
    plt.figure(figsize=(12,6))
    plt.bar(range(len(frequencies)), frequencies)
    plt.xlabel("Word Rank")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Distribution")

    plt.tight_layout()
    plt.savefig(saved_path, dpi=300, bbox_inches='tight')
    # plt.show()


def powerlaw_func(x, k, alpha):
    """defining the general powerlaw function being fit"""
    return k * np.power(x + 1, alpha)  # Note: using x+1 to avoid x=0


def fit_general_powerlaw(counts):
    """fix a zipfian law onto the data"""
    X = np.arange(len(counts))  # Convert to numpy array
    Y = counts
    
    popt, _ = curve_fit(powerlaw_func, X, Y)
    k, alpha = popt

    return k, alpha


def show_frequencies(thinking_freq, non_thinking_freq, saved_path):
    """show the frequency distributions side by side"""

    def get_counts(freq_dict):
        counts = [[key, value] for key, value in freq_dict.items()]
        sorted_counts = sorted(counts, key = lambda x: x[1])[::-1]
        return [x[1] for x in sorted_counts]


    def find_first_below(sorted_list: List, threshold: float):
        """Find the index of the FIRST element < threshold in a DESCENDING list"""
        left, right = 0, len(sorted_list) - 1
        result = len(sorted_list)  # Default: all elements are >= threshold
        
        while left <= right:
            mid = (left + right) // 2
            if sorted_list[mid] < threshold:
                result = mid  # This could be the answer
                right = mid - 1  # Look for an earlier element < threshold
            else:
                left = mid + 1  # Element is >= threshold, search right
        
        return result

    # finding the sorted counts 
    thinking_counts = get_counts(thinking_freq)
    non_thinking_counts = get_counts(non_thinking_freq)

    k_thinking, alpha_thinking = fit_general_powerlaw(thinking_counts)
    k_non_thinking, alpha_non_thinking = fit_general_powerlaw(non_thinking_counts)

    # remove elements which have frequency lower than the maximum frequency / cutoff_frac - for appearance
    cutoff_frac = 100 
    # cutoff_frac = None # set to none to disable this feature

    cutoff_count = math.ceil(min(thinking_counts[0], non_thinking_counts[0]) / cutoff_frac) # minimum count being considered
    cutoff_index = min(find_first_below(thinking_counts, cutoff_count), find_first_below(non_thinking_counts, cutoff_count))
                       
    # removing elements after this index
    thinking_counts, non_thinking_counts = thinking_counts[:cutoff_index], non_thinking_counts[:cutoff_index]

    # Generate power law curves for the visible range
    x_range = np.arange(len(thinking_counts))
    thinking_powerlaw = powerlaw_func(x_range, k_thinking, alpha_thinking)
    non_thinking_powerlaw = powerlaw_func(x_range, k_non_thinking, alpha_non_thinking)

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot thinking on left y-axis
    ax1.bar(range(len(thinking_counts)), thinking_counts, alpha=0.6, 
            label=f'Thinking (k={k_thinking:.2f}, α={alpha_thinking:.2f})', color='blue')
    ax1.plot(x_range, thinking_powerlaw, 'b-', linewidth=2, 
             label='Thinking Power Law')
    ax1.set_xlabel('Word Rank')
    ax1.set_ylabel('Thinking Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create second y-axis for non-thinking
    ax2 = ax1.twinx()
    ax2.bar(range(len(non_thinking_counts)), non_thinking_counts, alpha=0.6, 
            label=f'Non-Thinking (k={k_non_thinking:.2f}, α={alpha_non_thinking:.2f})', color='red')
    ax2.plot(x_range, non_thinking_powerlaw, 'r-', linewidth=2, 
             label='Non-Thinking Power Law')
    ax2.set_ylabel('Non-Thinking Frequency', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Word Frequency Comparison with Power Law Fits')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.tight_layout()
    plt.savefig(saved_path, dpi=300, bbox_inches='tight')


@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('-i', '--individual', help='generate individual graphs', default=False, is_flag=True)
def main(individual):
    """main script logic"""

    # defining the data path
    data_dir = Path("data")
    data_file_name = "SYNTHETIC-1.jsonl"
    data_path = data_dir / data_file_name
    
    # defining the figure path
    figure_dir = Path("analysis")
    figure_dir.mkdir(exist_ok=True)

    data_json = []
    with open(data_path, "r") as f:
        for line in tqdm(f, desc="Loading JSON..."):
            data_parsed = json.loads(line)  # Parse each line as JSON
            data_json.append(data_parsed)

    # each entry in data_json
    llm_responses = []
    for entry in data_json:
        llm_responses.append(entry["llm_response"])

    thinking_text = []
    non_thinking_text = []

    for text in llm_responses:
        think_blocks = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
        non_think_blocks = re.sub(r'<think>(.*?)</think>', '', text, flags=re.DOTALL)

        thinking_text.extend(think_blocks)
        non_thinking_text.append(non_think_blocks.strip())

    thinking_text_freq = find_word_freq(thinking_text)
    non_thinking_text_freq = find_word_freq(non_thinking_text)

    if individual:
        thinking_fig_path = figure_dir / "thinking_frequency.png"
        frequency_historam(thinking_text_freq, thinking_fig_path)

        non_thinking_fig_path = figure_dir / "non_thinking_frequency.png"
        frequency_historam(non_thinking_text_freq, non_thinking_fig_path)

    comb_fig = figure_dir / "frequencies.png"
    show_frequencies(thinking_text_freq, non_thinking_text_freq, comb_fig)


if __name__ == '__main__':
    main()
