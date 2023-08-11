# RefDBC - Uplink-Assist Downlink RS Image Compression

## Overview

RefDBC is a cutting-edge remote sensing (RS) image compression approach, presented in the IEEE TGRS paper titled "Uplink-Assist Downlink Remote Sensing Image Compression via Historical Referencing". This repository serves as the official implementation of the RefDBC method.

## Features

- Uplink-Assist Compression: RefDBC takes advantage of historical reference images received through the uplink to enhance on-orbit compression efficiency, reducing spatio-temporal redundancy in RS images.

- Dual-End Referencing Downsampling-Based Coding: RefDBC introduces a novel dual-end referencing framework, effectively handling radiation variations among RS images captured on different dates. Relevance embedding at the encoder and relevance-based super-resolution at the decoder ensure precise texture detail restoration.

- Fake Texture Mitigation: RefDBC efficiently suppresses fake texture generation caused by downsampling and compression, thanks to the incorporation of relevance referencing.

## Instructions

1. Requirements: Ensure that you have the required dependencies installed. Check the `requirements.txt` file for the necessary libraries, including Pytorch.

2. Data Preparation: Prepare your RS image dataset according to the guidelines provided in the `./dataset` folder. Organize the training and test data accordingly.
链接：https://pan.baidu.com/s/1e-9wvRqM51kuh-X7FU84LQ  提取码：bqo2
3. Pretrained Model: Download the pretrained model using the provided link from Baidu Netdisk. 链接: https://pan.baidu.com/s/1gcLrcMHK5uPX3RQeYmqZrA 
提取码:sg18

4. Test: to test the pretrained model on Spot-5 test dataset
   ```bash
   python main.py --save_dir ./eval/TTSR_GMM --reset True  --log_file_name eval.log  --eval True  --eval_save_results True  --usegmm True  --num_workers 0  --dataset SATELITE  --dataset_dir /path/of/your/data/set  --model_path /path/of/your/pretrained/model

## Citation

If you find this code useful in your research, please consider citing the following paper:
[Insert Citation for the IEEE TGRS paper]

## Contributions

We welcome any feedback, contributions, or issue reports related to the RefDBC implementation. Together, we can improve and extend the capabilities of this novel RS image compression approach. Happy coding!
