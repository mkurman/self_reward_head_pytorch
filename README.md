# Self-Reward Head for Language Models

This repository contains the implementation of a self-reward head designed for language models. The self-reward head enables the model to autonomously score its generated outputs, promoting self-assessment and iterative improvement. This innovative feature can be integrated into various language models to enhance their performance in text generation, translation, and summarization tasks.

## Features

- Autonomous scoring: The self-reward head evaluates the model's outputs without external input.
Improved output quality: By iteratively scoring and refining its outputs, the model can achieve higher-quality and more coherent results.
- Easy integration: Compatible with popular language models, making it simple to integrate into existing projects.
- Configurable scoring metrics: Allows customization of scoring criteria to fit specific use cases and performance goals.

## Installation

To install the self-reward head, clone the repository and follow the installation instructions in the README.

sh
git clone [https://github.com/mkurman/self_reward_head_pytorch](https://github.com/mkurman/self_reward_head_pytorch)
cd self-reward-head-pytorch

## Usage

The documentation provides detailed usage instructions and examples to help you effectively integrate and utilize the self-reward head.

You can copy the code from the `head.py` file and insert it into your models' architecture.

## Contributing

I welcome contributions!

## License

This project is licensed under the Apache 2.0 License.

## Citation
If you use this codebase, or otherwise found my work valuable, please cite:

```
@inproceedings{self_reward_head_pytorch,
  title={Self Reward Head: Autonomous evaluation of model results to self-improve its efficiency},
  author={Mariusz Kurman @ MedIT Solutions Kurman i Wspolnicy Sp. z o. o.},
  year={2024}
}
```
