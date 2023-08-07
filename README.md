# RefDBC - Uplink-Assist Downlink RS Image Compression

## Overview

RefDBC is a cutting-edge remote sensing (RS) image compression approach, presented in the IEEE TGRS paper titled "Uplink-Assist Downlink Remote Sensing Image Compression via Historical Referencing". This repository serves as the official implementation of the RefDBC method.

## Features

- Uplink-Assist Compression: RefDBC takes advantage of historical reference images received through the uplink to enhance on-orbit compression efficiency, reducing spatio-temporal redundancy in RS images.

- Dual-End Referencing Downsampling-Based Coding: RefDBC introduces a novel dual-end referencing framework, effectively handling radiation variations among RS images captured on different dates. Relevance embedding at the encoder and relevance-based super-resolution at the decoder ensure precise texture detail restoration.

- Fake Texture Mitigation: RefDBC efficiently suppresses fake texture generation caused by downsampling and compression, thanks to the incorporation of relevance referencing.

## Instructions

1. Requirements: Ensure that you have the required dependencies installed. Check the `requirements.txt` file for the necessary libraries, including Pytorch.

2. Data Preparation: Prepare your RS image dataset from the Spot-5 and Luojia3 satellites. Organize the images and reference data following the guidelines provided in the "data" folder.

3. Training: Train the RefDBC models with different compression baselines (e.g., RefDBC+JPEG2000, RefDBC+HEVC, and RefDBC+GMM) using the Adam optimizer and relevant loss functions. Use the provided training scripts for ease of use.

4. Testing: Evaluate the trained models on the test dataset and calculate the bitrate savings compared to standard, learning-based, and DBC compression methods.

## Citation

If you find this code useful in your research, please consider citing the following paper:
[Insert Citation for the IEEE TGRS paper]

## Contributions

We welcome any feedback, contributions, or issue reports related to the RefDBC implementation. Together, we can improve and extend the capabilities of this novel RS image compression approach. Happy coding!
