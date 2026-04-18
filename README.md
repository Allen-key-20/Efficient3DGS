We conducted the experiment on Ubuntu 22.04, with the graphics card being RTX 3090, the driver version being nvidia-driver-570-server, the CUDA version being 11.8, and the Python version being 3.10.
## Setup

### Local Setup

```shell
git clone https://github.com/Allen-key-20/Efficient3DGS.git
cd Efficient3DGS

conda env create -n e3d python=3.10 -y
conda activate e3d

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

```
### Dataset
You can find three datasets in [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) . Put the downloaded dataset into the "data" folder

```
data
  ├── Mip-NeRF360
  │   ├── bicycle
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
  ├── Deep Blending
  │   ├── playroom
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
  ├── Tanks&Temples
  │   ├── truck
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
```

### Running

Train a single scene. Set `model = ' '` in the `run.py` file to the name of the scenario to be trained. 
```shell
python run.py 
```

Train the entire dataset. Train the default three datasets.
```shell
python run_folder.py 
```
### Note

Our code is modified based on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) . The simpleknn in the submodules folder is also from this project. We have made modifications to [fast_ssim](https://github.com/rahul-goel/fused-ssim/tree/1272e21a282342e89537159e4bad508b19b34157) to support images in the (H, W, C) format.
An initial open-source version of our method is already available, which demonstrates the core methodology and enables reproduction of the main experimental results.    To ensure code quality and usability, we are adopting a progressive open-source strategy: more complete and optimized implementations will be released as the development stabilizes.
