# RL4Echo

RL4Echo is a reinforcement learning framework for domain adaptation of echocardiography segmentation.

Key elements of single timestep segmentation RL are defined as follows:
- State (s): input image
- Action (a): segmentation for a given image
- Reward r(s, a): pixelwise error map for a given segmentation and image.

It's iterative 3 step loop allows the user to fine-tune a segmentation policy (pretrained on a source domain) on the target domain without the need for expert annotations:
1. Predict on the target domain with the current segmentation policy. Create a reward network with valid, invalid segmentations and input images using anatomical metrics, distortions, etc.
2. Train the reward network (supervised) using the reward dataset.
3. Fine-tune the segmentation policy with PPO against the newly trained reward network. (Return to step 1.)

The reward network can be used as a reliable uncertainty estimator once training is complete.

While it has been tested and used on echocardiography images, it can be used with any other image modality.

## References
Arnaud Judge, Thierry Judge, Nicolas Duchateau, Roman A. Sandler, Joseph Z. Sokol, Olivier Bernard, Pierre-Marc Jodoin. Domain Adaptation of Echocardiography Segmentation Via Reinforcement Learning. International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2024, Marrakesh, Morocco.

Preprint available at: https://www.arxiv.org/abs/2406.17902

## Install

1. Download the repository:
   ```bash
   # clone project
   git clone --recurse-submodules https://github.com/arnaudjudge/RL4Echo
   cd RL4Echo
   ```
2. Create a virtual environment (Conda is strongly recommended):
   ```bash
   # create conda environment
   conda create -n rl python=3.10
   conda activate rl
   ```
3. Install [PyTorch](https://pytorch.org/get-started/locally/) according to instructions. Grab the one with GPU for faster training:
   ```bash
   # example for linux or Windows
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
4. Install the project in editable mode and its dependencies:
   ```bash
   pip install -e .
   ```
5. Install vital submodule in editable mode (without dependencies):
    ```bash
    cd vital
    pip install -e . --no-deps
    cd ../patchless-nnUnet
    pip install -e . --no-deps
    cd ..
    ```

## Run
   To obtain predictions from an input folder and a pre-trained checkpoint, use the following command:
   By default, it will use TTA and automatic TTO, see config file for all options
   ```bash
   predict_3d input_folder=<PATH/TO/DATA/> output_folder=<OUT/> ckpt_path=<checkpoint_file>
   ```

   Alternatively, for fastest results, no test-time augmentation nor optimization
   ```bash
   predict_3d input_folder=<PATH/TO/DATA/> output_folder=<OUT/> ckpt_path=<checkpoint_file> tta=False tto=off
   ```

To run training, use the runner.py script for individual runs, 
or the auto_iteration.py script for full RL loops. (WILL BE IMPROVED FOR BETTER USABILITY!)