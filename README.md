# LLM Reasoning Distribution

Chain of thought (CoT) is an invaluable technique for interpreting and understanding the reasoning in large language models.

However, there is broad concern over the longevity of techniques relying on a human interpretable CoT. AI labs are increasingly applying optimisation pressures to the chain of thought, such that instead of giving an honest view into the mind of a model it is reflective of the intentions of the developer as much as the model's thoughts.

To get a glimpse into the future, it is always necessary to take a detour back into the past. Here, we attempt to shed some light on where we expect the future of CoT monitoring to go by looking at the distribution of word frequencies in model reasoning traces. 

We expect to see two things:
1. Models should tend towards using tokens with equal probability if RL with a length penalty is applied to the CoT during post-training.
2. Certain words or bigrams should fall out of use entirely after sufficient post training (assuming that there is a fixed set of concepts being represented, which is smaller than the total number of bigrams available given a token vocabulary).

## About the Data

We are interested in large datasets of reasoning from various different models. Each dataset will be from some reasoning model, giving us insight into how these distribution of text in the chain of thought evolves over time.

| **Dataset** | **Model(s)** | **Size** | **Notes** |
| ----------- | ------------ | -------- | --------- |
| [SYNTHETIC-1](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1) | Deepseek-R1 | 26.4 GB | **Warning**: Extremely large dataset. You are advised to either partially load this, or otherwise ensure that you have the capacity available |
| [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | QwQ-32B | 28.2 GB | Contains full conversations, not just the reasoning section of the outputs |
| [NuminaMath-Cot](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Humans | 1.23 GB | Mathematical problem solving, based on exam papers and online discussion forums |
| [NuminaMath-QwQ-CoT-5M](https://huggingface.co/datasets/PrimeIntellect/NuminaMath-QwQ-CoT-5M) | QwQ | 18Gb | Mathematical problem solving and reasoning |
| [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) | GPT-4o | 247Mb | Medical situational reasoning and tests of knowledge |


## Quick Start

To download the relevant depedencies, run the `setup.sh` command. Note that this project uses [uv package management](https://docs.astral.sh/uv/).

```bash
chmod +x setup.sh
./setup.sh
```

The `run.sh` script will download and process all relevant datasets.
If you run this command with the `./run.sh -d` flag, downloaded datasets will be deleted immediately after processing is complete.

```bash
chmod +x run.sh
./run.sh
```

To clean-up data once it's been downloaded to free up disk space, run the `cleanup_data.sh` command.
This will remove everything in the `/data_processing/*/data/` directories, leaving the scripts and the analysis untouched.

```bash
chmod + cleanup_data.sh
./cleanup_data.sh
```
