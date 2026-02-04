# Environment Setup
All of our training and inference have been performed on Windows x64 systems. Below, we describe the process of manually establishing an environment suitable for running these scripts. We also detail the specific CUDA, CUDNN, and library versions utilized for the processing of hilcorp data. The exact environment setup for CIBA2/3 are thus far unknown but will be determined at a later date.

# Dependencies
## Binaries to Install
* CUDA-Compatiable GPU
* CUDNN 11.2 Windows X64
* CUDA 11.2.2
* Anaconda

## Python Packages
### PIP
* openpyxl
* scipy
* tensorflow=
* opencv-contrib-python==4.6.0.66
* numpy
* tensorflow<2.11
* pillow

This can be achieved by running 
`pip install "tensorflow<2.11" opencv-contrib-python numpy pandas openpyxl scipy pillow`
### Conda

`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

# Procedure
1. Install CUDA, then unzip CUDNN and merge with the CUDA folder
2. Install Anaconda, set up a new python environment (3.9.x). Name it something intuitive to you (e.g., ciba_ai)
3. Install python packages via PIP in the target virtual environment
4. Install the cuda toolkit and cudnn packages via conda
5. Open spyder (or your IDE of choice and run `import tensorflow as tf`
6. Then `print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))`
7. You should get an output that indicates that there are at least 1 GPUs available to you
If this works, that means we've (likely) had success with installing the requisite dependencies needed to run the scripts.

