# B0Mapping: Matlab to Python Translation

## Overview

This project focuses on translating existing Matlab code for **B0 mapping** into Python. It represents the first step in a broader pipeline aimed at streamlining **functional MRI (fMRI)** workflows for neuroscientists. Our goal is to significantly reduce the time between acquiring MRI data and obtaining processed results, ultimately improving efficiency in neuroscience research.

Currently, the process of generating B0 maps from raw MRI data takes **1 to 2 weeks**, creating a bottleneck in research timelines. By converting the Matlab codebase to Python, we aim to lay the foundation for a faster, more accessible, and open-source solution that integrates seamlessly into a modern fMRI processing pipeline.

## Purpose

The motivation behind this project is simple: **to make neuroscientists' lives better**. By reducing the lag between data acquisition and analysis, we hope to accelerate research progress, enabling scientists to focus on discoveries rather than waiting for processed outputs. This translation effort is the initial building block in a larger vision to optimize the entire fMRI workflow.

## What is B0 Mapping?

B0 mapping refers to the process of measuring and mapping the **static magnetic field (B0)** variations in an MRI scanner. These variations can cause distortions in fMRI images, and B0 maps are essential for correcting these distortions to ensure accurate brain imaging and analysis.

## Features

- **Matlab-to-Python Translation**: Converts legacy Matlab B0 mapping scripts into Python for broader accessibility and compatibility.
- **First Step in a Pipeline**: Designed as the starting point for an end-to-end fMRI processing solution.
- **Open-Source Friendly**: Python implementation encourages collaboration and customization by the neuroscience community.

## Why Python?

- **Accessibility**: Python is widely used, free, and supported by a vast ecosystem of scientific libraries (e.g., NumPy, SciPy, and NiBabel).
- **Performance**: Future optimizations can leverage Python’s tools to reduce processing times.
- **Community**: Transitioning to Python opens the door for contributions from a larger pool of developers and researchers.

## Current Status

This project is in its early stages, focusing on faithfully translating the Matlab B0 mapping code into Python. While it’s not yet a complete pipeline, it sets the stage for future enhancements, such as:
- Integration with additional fMRI processing steps.
- Optimization to cut down the 1–2 week processing delay.
- User-friendly documentation and tools for neuroscientists.

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: (To be specified as the project develops—e.g., NumPy, Matplotlib, etc.)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/femaleprog/b0_mapping.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the translated Python Script:
   ```bash
   python python_code.py

## Conributing
We welcome contributions from the community! Whether you’re a neuroscientist, a Python developer, or just passionate about improving research tools, feel free to:
- Submit issues or feature requests.
- Fork the repo and send pull requests with improvements.

## Future Goals 
- Connect this code to the Sparkling reconstruction code.
- Load raw data instead of DICOM.
- Enhance performance using deep learning tools
## Contact
For questions, suggestions, or collaboration inquiries, please reach out to [soukaina.tichirra@cea.fr (mailto:soukaina.tichirra@cea.fr)] or open an issue on this repository.
