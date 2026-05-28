# RefDBC - Uplink-Assist Downlink RS Image Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

RefDBC is a cutting-edge remote sensing (RS) image compression approach, presented in the IEEE TGRS paper titled **"Uplink-Assist Downlink Remote Sensing Image Compression via Historical Referencing"**. This repository serves as the official implementation of the RefDBC method.

---

## ✨ Features

- **Uplink-Assist Compression:** RefDBC takes advantage of historical reference images received through the uplink to enhance on-orbit compression efficiency, reducing spatio-temporal redundancy in RS images.
- **Dual-End Referencing Downsampling-Based Coding:** Introduces a novel dual-end referencing framework, effectively handling radiation variations among RS images captured on different dates. Relevance embedding at the encoder and relevance-based super-resolution at the decoder ensure precise texture detail restoration.
- **Fake Texture Mitigation:** Efficiently suppresses fake texture generation caused by downsampling and compression, thanks to the incorporation of relevance referencing.
- **CPU Inference Support:** The code has been optimized and updated to allow seamless evaluation and testing on CPU-only devices (such as NVIDIA Jetson Orin's CPU mode).

---

## 🚀 Getting Started

### 1. Requirements & Installation

We recommend using a virtual environment (e.g., Anaconda or Python `venv`). The codebase has been updated to fix deprecation bugs and CPU evaluation compatibility.

Clone the repository and install dependencies:

```bash
git clone https://github.com/WHW1233/RefDBC.git
cd RefDBC
pip install -r requirements.txt
```

### 2. Data Preparation

Prepare your RS image dataset according to the guidelines provided in the `./dataset` folder. Organize the training and test data accordingly.

- **Dataset Link (Baidu Netdisk):** [https://pan.baidu.com/s/1e-9wvRqM51kuh-X7FU84LQ](https://pan.baidu.com/s/1e-9wvRqM51kuh-X7FU84LQ)
- **Extraction Code:** `bqo2`

### 3. Pretrained Model

Download the pretrained models using the provided link from Baidu Netdisk.

- **Model Link:** [https://pan.baidu.com/s/1gcLrcMHK5uPX3RQeYmqZrA](https://pan.baidu.com/s/1gcLrcMHK5uPX3RQeYmqZrA)
- **Extraction Code:** `sg18`

Place the downloaded models into the `./pretrained` directory (or specify the path using `--model_path`).

---

## 💻 Evaluation & Testing

You can easily evaluate the model on the provided Spot-5 test dataset or your own custom images.

### Quick Test on Local CPU
If you are running on an edge device without a dedicated GPU (e.g., Orin CPU), add `--cpu True` to your command:

```bash
python main.py \
    --test True \
    --cpu True \
    --lr_path ./test/demo/lr/lr.png \
    --ref_path ./test/demo/ref/ref.png \
    --reset True
```
> **Note:** The `--reset True` flag is useful to automatically overwrite previous output directories. Outputs will be saved in `save_dir/save_results/`.

### Evaluate Entire Dataset
To evaluate the model over the full SATELITE dataset using TTSR+GMM:

```bash
python main.py \
    --save_dir ./eval/TTSR_GMM \
    --reset True \
    --log_file_name eval.log \
    --eval True \
    --eval_save_results True \
    --usegmm True \
    --num_workers 0 \
    --dataset SATELITE \
    --dataset_dir /path/to/your/dataset \
    --model_path ./pretrained/model_x2_compress.pt
```

*(Optional)* Append `--cpu True` if running without a CUDA device.

---

## 📚 Citation

If you find this code useful in your research, please consider citing the following paper:

```bibtex
@article{RefDBC2023,
  title={Uplink-Assist Downlink Remote Sensing Image Compression via Historical Referencing},
  author={WHW et al.},
  journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
  year={2023},
  doi={10.1109/TGRS.2023.3315725}
}
```
*(Please replace with the official citation once available)*

---

## 🤝 Contributions

We welcome any feedback, contributions, or issue reports related to the RefDBC implementation! Together, we can improve and extend the capabilities of this novel RS image compression approach. Happy coding!
