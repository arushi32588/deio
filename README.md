# Clone 
git clone 
cd DEIO

# Check if NVIDIA driver is installed
nvidia-smi

# If not installed, download & install driver from NVIDIA site

# Install CUDA Toolkit (matching your driver & PyTorch version)
e.g., https://developer.nvidia.com/cuda-downloads


# conda environment
conda env create -f environment.yml
conda activate DEIO

#Python packages
pip install .
pip install numpy-quaternion==2022.4.3

# Install GTSAM (needed for optimization)
cd thirdparty/gtsam
mkdir build
cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.10.11
make python-install
cd ../../..

## Dataset

Download the DAVIS240c shapes-rotation dataset:
https://rpg.ifi.uzh.ch/davis_data.html

Download DEVO.pth model and place it within models/:
cd /Users/arushiagrawal/Desktop/DEIO/models && wget https://cvg.cit.tum.de/webshare/g/evs/DEVO.pth

Place it here:
DEIO/shapes_rotation

## Run Inference

CUDA_VISIBLE_DEVICES=0 python script/eval_deio/davis240c.py \
    --inputdir=shapes_rotation \
    --config=config/davis240c.yaml \
    --val_split=script/splits/davis240c/shapes_rotation_val.txt \
    --enable_event \
    --network=models/DEVO.pth \
    --plot \
    --save_trajectory \
    --trials=1

## visualization

pip install evo

# Compare your estimated trajectory against ground truth
evo_traj tum results/DAVIS240C/shapes-rotation/estimated_trajectory_1.txt \
             shapes_rotation/groundtruth.txt \
             -p --plot
