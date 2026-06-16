# Adaptive Action Chunking at Inference-time for Vision-Language-Action Models (RoboCasa Evaluation)

This is the official codebase of AAC.

[**[Home page]**](https://lance-lot.github.io/adaptive-chunking.github.io/) &ensp; [**[Paper]**](https://arxiv.org/abs/2604.04161)

-------
## Overview
For a quick review of the implentation of AAC algorithm, see the function select_chunk_size in [action_entropy_v2.py](https://github.com/Adaptive-Action-Chunking/robocasa/blob/main/robocasa/demos/action_optimization/action_entropy_v2.py).

This repo is built on offcial [RoboCasa](https://github.com/robocasa/robocasa) with modifications to support AAC. If you already have RoboCasa installed, simply copy and replace the folder robocasa/demos from this repo to your original robocasa projects. If you don't have RoboCasa installed before, follow the full installation instructions below. Then replace robocasa/demos with the folder from this repo.

Note:
If you want to align the RoboCasa version with our experiments, you can use the command below to specify the RoboCasa code base version.

In the project folder of RoboCasa:
```
git checkout 0eae0634a61ad2be33962c9de7000a2dd1ee573f
```

-------
## Installation of Official RoboCasa
RoboCasa works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.10
   ```
2. Activate conda environment:
   ```sh
   conda activate robocasa
   ```
3. Clone and setup robosuite dependency (**important: use the master branch!**):

   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd ..
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install           # Optional: set up code formatter.

   (optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
   ```
5. Install the package and download assets:
   ```sh
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.
   ```

## Usage
Our implementation is based on server-client mode. This repo only includes the client part. To run a policy server, we provide the implementation of [GR00T N1.5](https://github.com/Adaptive-Action-Chunking/gr00t-multi-sample) (with modification to support sampling mutiple action chunks in parallel.)

example to run the evaluation client for one task with AAC:
```sh
cd robocasa/demos/action_optimization
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/gr00t15_action_entropy_test --task PnPCounterToCab --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081
```

Configure --task for different tasks in RoboCasa, set --port according to your policy server.

Refer to [robocasa/demos/action_optimization/AAC_evaluation.sh](https://github.com/Adaptive-Action-Chunking/robocasa/blob/main/robocasa/demos/action_optimization/AAC_evaluation.sh) for more example scripts.

Note that default max_steps is 600, but for the task OpenDoubleDoor max_steps is set to 1200 to allow enough time to finish this long horizon task.
 
-------
## Citation
```bibtex
@inproceedings{liang2026adaptive,
  title={Adaptive action chunking at inference-time for vision-language-action models},
  author={Liang, Yuanchang and Wang, Xiaobo and Wang, Kai and Wang, Shuo and Peng, Xiaojiang and Chen, Haoyu and Chua, David Kim Huat and Vadakkepat, Prahlad},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20802--20811},
  year={2026}
}
```
