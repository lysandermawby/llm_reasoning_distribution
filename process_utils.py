#!/usr/bin/env python3
"""helper functions for consuming and processing data"""

# package imports
import re
import string
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Any, Optional
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime


"""
TODO:

(clean!)
"""


def clean_word(word: str) -> Optional[str]:
    """clean a word for frequency counting, incl removing punctuation and capitalisation"""
    # Convert to lowercase
    word = word.lower()

    # Strip punctuation from both ends
    word = word.strip(string.punctuation)

    # Skip empty strings or single characters that are just punctuation
    if not word or (len(word) == 1 and word in string.punctuation):
        return None

    # Skip if the word is only digits
    if word.isdigit():
        return None

    return word


# returns the count dictionaries for a block of text where some information is stored inside thinking tags
def counts_from_text(text_chunks: str, think_open_tag: str = '<think>', think_close_tag: str = '</think>'
) -> tuple[dict[str, int], dict[str, int]]:
    """Finds the frequencies of words in the thinking and regular portions of text

    Takes text assuming that the thinking portion is contained within <think></think> tags
    Provides the counts of that text in dictionary form, mapping the word to its own frequency
    """

    # Escape special regex characters and build pattern
    open_tag = re.escape(think_open_tag)
    close_tag = re.escape(think_close_tag)
    pattern = f'{open_tag}(.*?){close_tag}'

    # Process chunks with progress bar
    think_parts = []
    reg_parts = []

    for text in tqdm(text_chunks, desc="Processing chunks", unit="chunk"):
        think_parts.append(''.join(re.findall(pattern, text, re.DOTALL)))
        reg_parts.append(re.sub(pattern, '', text, flags=re.DOTALL))

    think_text = ''.join(think_parts)
    reg_text = ''.join(reg_parts)

    print(f"\nDebug: Total thinking text length: {len(think_text)} characters")
    print(f"Debug: Total regular text length: {len(reg_text)} characters")

    think_freq, reg_freq = frequencies_from_text(think_text), frequencies_from_text(reg_text)

    print(f"Debug: Thinking vocabulary size: {len(think_freq)} unique words")
    print(f"Debug: Regular vocabulary size: {len(reg_freq)} unique words")
    print(f"Debug: Total thinking words: {sum(think_freq.values())}")
    print(f"Debug: Total regular words: {sum(reg_freq.values())}")

    return think_freq, reg_freq


def frequencies_from_text(text: str) -> Dict:
    """finds a dictionary mapping a word to its frequency in text

    Words are normalized (lowercased, punctuation removed) before counting.
    """
    frequency_dict = {}
    for raw_word in text.split():
        word = clean_word(raw_word)
        if word:  # Only count if word is not None (i.e., not filtered out)
            frequency_dict[word] = frequency_dict.get(word, 0) + 1

    return frequency_dict


def load_jsonl(data_path: Path) -> List[Dict[str, Any]]:
    """loads a jsonl file into list of dictionaries"""
    json_data = []
    with open(data_path, "r") as f:
        for line in tqdm(f, desc=f"Loading JSON from {data_path}..."):
            data_parsed = json.loads(line)  # Parse each line as JSON
            json_data.append(data_parsed)
    return json_data


def process_and_analyze_text_chunks(text_chunks: List[str], dataset_name: str, analysis_dir: Path, save_dict: bool = False,
    plot: bool = False, individual: bool = False, think_open_tag: str = '<think>', think_close_tag: str = '</think>'):
    """Process text chunks and perform complete analysis with optional plotting

    This is the main processing function that handles everything after text extraction
    It performs frequency analysis, powerlaw fitting, saves results, and optionally creates plots
    """
    # Create analysis directory
    analysis_dir.mkdir(exist_ok=True)

    # Extract thinking and regular text frequencies
    think_freq, reg_freq = counts_from_text(text_chunks, think_open_tag, think_close_tag)

    # Save frequency dictionaries if requested
    if save_dict:
        print("Saving frequencies")
        think_freq_file = save_freq_dict(think_freq, analysis_dir, file_name="thinking_frequencies.json")
        reg_freq_file = save_freq_dict(reg_freq, analysis_dir, file_name="regular_frequencies.json")
        print(f"Saved the thinking text frequencies to {think_freq_file} and the regular text frequencies to {reg_freq_file}")

    # Get counts and calculate powerlaw coefficients
    think_counts = get_counts(think_freq)
    reg_counts = get_counts(reg_freq)
    k_think, alpha_think, k_reg, alpha_reg = calculate_powerlaw_coefficients(think_counts, reg_counts)

    # Save analysis results
    print("Saving analysis results")
    results_path = save_analysis_results(
        dataset_name=dataset_name,
        think_freq=think_freq,
        reg_freq=reg_freq,
        k_think=k_think,
        alpha_think=alpha_think,
        k_reg=k_reg,
        alpha_reg=alpha_reg,
        analysis_dir=analysis_dir
    )
    print(f"Saved analysis results to {results_path}")

    # Create plots if requested
    if plot:
        print("Plotting frequencies")

        # Create individual plots if requested
        if individual:
            think_fig_path = analysis_dir / "thinking_frequency.png"
            plot_individual_freq(think_counts, think_fig_path, title="Thinking Text Frequency Distribution")

            reg_fig_path = analysis_dir / "non_thinking_frequency.png"
            plot_individual_freq(reg_counts, reg_fig_path, title="Non-Thinking Text Frequency Distribution")

        # Create combined plot
        comb_fig_path = analysis_dir / "frequencies.png"
        plot_freq_counts(think_counts, reg_counts, comb_fig_path, k_think, alpha_think, k_reg, alpha_reg)


def save_freq_dict(freq_dict: Dict, analysis_dir: Path, file_name: str = "freq_dict.json") -> Path:
    """save the dictionary of frequencies"""
    analysis_dir.mkdir(exist_ok=True) # make the analysis directory if it doesn't exist
    file_path = analysis_dir / file_name
    with open(file_path, "w") as f:
        json.dump(freq_dict, f, indent=2)
    return file_path


def save_analysis_results(dataset_name: str, think_freq: Dict, reg_freq: Dict, k_think: float,
    alpha_think: float, k_reg: float, alpha_reg: float, analysis_dir: Path, file_name: str = "analysis_results.json",
    top_n_words: Optional[int] = 1000) -> Path:
    """Save analysis results including powerlaw coefficients, statistics, and top N word frequencies to JSON

    Returns a path to the saved file.
    """
    analysis_dir.mkdir(exist_ok=True)

    # Calculate statistics
    total_think_words = sum(think_freq.values())
    total_reg_words = sum(reg_freq.values())
    unique_think_words = len(think_freq)
    unique_reg_words = len(reg_freq)

    # Get top N words sorted by frequency
    think_sorted = sorted(think_freq.items(), key=lambda x: x[1], reverse=True)
    reg_sorted = sorted(reg_freq.items(), key=lambda x: x[1], reverse=True)

    # Take top N words (or all if less than N)
    top_n_words = top_n_words if top_n_words else float('int')
    top_think = dict(think_sorted[:min(top_n_words, len(think_sorted))])
    top_reg = dict(reg_sorted[:min(top_n_words, len(reg_sorted))])

    results = {
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "thinking_text": {
            "powerlaw_k": float(k_think),
            "powerlaw_alpha": float(alpha_think),
            "total_words": total_think_words,
            "unique_words": unique_think_words,
            "type_token_ratio": unique_think_words / total_think_words if total_think_words > 0 else 0,
            f"top_{top_n_words}_words": top_think
        },
        "regular_text": {
            "powerlaw_k": float(k_reg),
            "powerlaw_alpha": float(alpha_reg),
            "total_words": total_reg_words,
            "unique_words": unique_reg_words,
            "type_token_ratio": unique_reg_words / total_reg_words if total_reg_words > 0 else 0,
            f"top_{top_n_words}_words": top_reg
        }
    }

    file_path = analysis_dir / file_name
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return file_path


def get_counts(freq_dict: Dict) -> List[int]:
    """takes the frequencies dictionary and returns the sorted counts"""
    counts = [[key, value] for key, value in freq_dict.items()]
    sorted_counts = sorted(counts, key = lambda x: x[1])[::-1]
    return [x[1] for x in sorted_counts]


def powerlaw_func(x, k, alpha):
    """defining the general powerlaw function being fit"""
    return k * np.power(x + 1, alpha)  # Note: using x+1 to avoid x=0


def fit_general_powerlaw(counts):
    """fit a zipfian law onto the data"""
    X = np.arange(len(counts))
    Y = np.array(counts)

    # Provide better initial guesses and bounds for the optimization
    # k should be close to the first count (most frequent word)
    # alpha should be negative (power law decay)
    k_initial = Y[0] if len(Y) > 0 else 1
    alpha_initial = -1.0

    try:
        # Add bounds to constrain the search space
        # k should be positive and reasonably close to max frequency
        # alpha should be negative (decay)
        popt, pcov = curve_fit(
            powerlaw_func,
            X,
            Y,
            p0=[k_initial, alpha_initial],
            bounds=([Y[0] * 0.1, -10], [Y[0] * 10, 0]),
            maxfev=10000
        )
        k, alpha = popt
    except Exception as e:
        print(f"Warning: curve_fit failed with error: {e}")
        print("Using fallback linear regression on log-log scale")
        # Fallback: use log-log linear regression
        # log(y) = log(k) + alpha * log(x+1)
        log_x = np.log(X + 1)
        log_y = np.log(Y)
        # Remove any -inf or nan values
        valid = np.isfinite(log_y)
        if np.sum(valid) > 1:
            coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
            alpha = coeffs[0]
            k = np.exp(coeffs[1])
        else:
            # Last resort fallback
            k = k_initial
            alpha = alpha_initial

    return k, alpha



def find_first_below(sorted_list: List, threshold: float) -> int:
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


def calculate_powerlaw_coefficients(think_counts: List[int], reg_counts: List[int]):
    """Calculate powerlaw coefficients for thinking and regular text counts

    This function fits powerlaw curves to the FULL dataset (no cutoff applied).
    Cutoffs are only applied during plotting for readability.

    Args:
        think_counts: Sorted frequency counts for thinking text
        reg_counts: Sorted frequency counts for regular text
    """
    print(f"\nDebug: Thinking counts length: {len(think_counts)}")
    print(f"Debug: Regular counts length: {len(reg_counts)}")
    if len(think_counts) > 0:
        print(f"Debug: Thinking counts range: {think_counts[-1]} to {think_counts[0]}")
    if len(reg_counts) > 0:
        print(f"Debug: Regular counts range: {reg_counts[-1]} to {reg_counts[0]}")

    # Calculate powerlaw coefficients on FULL dataset
    k_think, alpha_think = fit_general_powerlaw(think_counts) if think_counts else (1.0, -1.0)
    k_reg, alpha_reg = fit_general_powerlaw(reg_counts) if reg_counts else (1.0, -1.0)

    print(f"Debug: Powerlaw fit results - k_think: {k_think:.4f}, alpha_think: {alpha_think:.4f}")
    print(f"Debug: Powerlaw fit results - k_reg: {k_reg:.4f}, alpha_reg: {alpha_reg:.4f}")

    return k_think, alpha_think, k_reg, alpha_reg


def plot_individual_freq(counts: List[int], saved_path: Path, title: str = "Word Frequency Distribution", cutoff_frac: Optional[float] = 100):
    """plot individual frequency histogram"""
    # Apply cutoff if specified
    if cutoff_frac and len(counts) > 0:
        max_freq = counts[0]
        cutoff_count = math.ceil(max_freq / cutoff_frac)
        cutoff_index = find_first_below(counts, cutoff_count)
        counts = counts[:cutoff_index]
        print(f"Debug: Individual plot cutoff at index {cutoff_index} (count threshold: {cutoff_count})")

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(counts)), counts)
    plt.xlabel("Word Rank")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(saved_path, dpi=300, bbox_inches='tight')
    print(f"Saved individual frequency plot to {saved_path}")
    plt.close()


def plot_freq_counts(think_counts: List[int], reg_counts: List[int], saved_path: Path, k_think: float, alpha_think: float, k_reg: float, alpha_reg: float, cutoff_frac: Optional[float] = 100):
    """plot the counts and their Zipfian fits"""

    # Handle empty lists
    if not think_counts and not reg_counts:
        print("Warning: Both thinking and regular counts are empty. Skipping plot.")
        return

    # Apply cutoff for display if specified
    if cutoff_frac and (think_counts or reg_counts):
        # Get max frequency from whichever list is non-empty
        if think_counts and reg_counts:
            max_freq = max(think_counts[0], reg_counts[0])
        elif think_counts:
            max_freq = think_counts[0]
        else:
            max_freq = reg_counts[0]

        cutoff_count = math.ceil(max_freq / cutoff_frac)

        # Apply cutoff to each list independently
        if think_counts:
            think_cutoff = find_first_below(think_counts, cutoff_count)
            think_counts = think_counts[:think_cutoff]
        if reg_counts:
            reg_cutoff = find_first_below(reg_counts, cutoff_count)
            reg_counts = reg_counts[:reg_cutoff]

    print(f"Powerlaw coefficients for thinking counts: k = {k_think:.2f} ; alpha = {alpha_think:.2f}")
    print(f"Powerlaw coefficients for regular counts: k = {k_reg:.2f} ; alpha = {alpha_reg:.2f}")

    # Determine the range for plotting (use the longer list)
    max_len = max(len(think_counts) if think_counts else 0, len(reg_counts) if reg_counts else 0)
    if max_len == 0:
        print("Warning: No data to plot after cutoff.")
        return

    # Generate power law curves
    x_range = np.arange(max_len)
    think_powerlaw = powerlaw_func(x_range[:len(think_counts)], k_think, alpha_think) if think_counts else np.array([])
    reg_powerlaw = powerlaw_func(x_range[:len(reg_counts)], k_reg, alpha_reg) if reg_counts else np.array([])

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot thinking on left y-axis
    if think_counts:
        ax1.bar(range(len(think_counts)), think_counts, alpha=0.6,
                label=f'Thinking (k={k_think:.2f}, α={alpha_think:.2f})', color='blue')
        ax1.plot(range(len(think_counts)), think_powerlaw, 'b-', linewidth=2,
                 label='Thinking Power Law')
        ax1.set_ylabel('Thinking Frequency', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

    ax1.set_xlabel('Word Rank')

    # create second y-axis for non-thinking
    if reg_counts:
        if think_counts:
            # use twinx only if we have both datasets
            ax2 = ax1.twinx()
        else:
            # use single axis if only regular text
            ax2 = ax1

        ax2.bar(range(len(reg_counts)), reg_counts, alpha=0.6,
                label=f'Non-Thinking (k={k_reg:.2f}, α={alpha_reg:.2f})', color='red')
        ax2.plot(range(len(reg_counts)), reg_powerlaw, 'r-', linewidth=2,
                 label='Non-Thinking Power Law')

        if think_counts:
            ax2.set_ylabel('Non-Thinking Frequency', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        else:
            ax2.set_ylabel('Frequency', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Word Frequency Comparison with Power Law Fits')

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if think_counts and reg_counts:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(lines1, labels1, loc='upper right')

    fig.tight_layout()
    plt.savefig(saved_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined frequency plot to {saved_path}")
    plt.close()


# test function using the SYNTHETIC-1 data
def main():
    # defining the data path
    data_dir = Path("data_processing/SYNTHETIC-1/data")
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

    think_text_lst = []
    reg_text_lst = []
    text_lst = []
    for text in llm_responses:
        think_text, reg_text = counts_from_text(text)
        think_text_lst.append(think_text)
        reg_text_lst.append(reg_text)
        text_lst.append(text)

    all_think_text = ''.join(think_text_lst)
    all_reg_text = ''.join(reg_text_lst)
    all_text = ''.join(text_lst)

    print(len(all_think_text))
    print(len(all_reg_text))

    num_think_blocks = len(re.findall(r'<think>.*?</think>', all_text, re.DOTALL))
    tag_chars = num_think_blocks * (len('<think>') + len('</think>'))
    print(f"Number of thinking blocks: {num_think_blocks}")
    print(f"Characters used by tags: {tag_chars}")

    print(f"Remaining text: {len(all_text) - len(all_think_text) - len(all_reg_text)}")


if __name__ == '__main__':
    main()

