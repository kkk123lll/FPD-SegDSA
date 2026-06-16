# FPD-SegDSA

PyTorch implementation for DSA sequence vessel segmentation. The current public version focuses on fully supervised segmentation with the `New_Mamba_Net` architecture.

## Repository Layout

```text
.
├── train.py                        # training entry point
├── test.py                         # evaluation entry point
├── fpd_seg/                        # project package
│   ├── architectures/              # New_Mamba_Net architecture
│   ├── config/                     # default configuration
│   ├── data/                       # datasets and dataloaders
│   ├── training/                   # trainer, tester, scheduler, and optimizer
│   ├── objectives/                 # losses
│   └── common/                     # metrics, augmentation, and helpers
├── tools/preprocess/               # optional preprocessing scripts
├── scripts/                        # experiment launch helpers
├── docs/                           # lightweight figures and PDFs
├── notebooks/                      # exploratory notebooks
└── environment.yml                 # conda environment export
```

Experiment outputs such as checkpoints, TensorBoard runs, debug batches, caches, and local datasets are intentionally ignored by Git.

## Data Format

This repository does not include the dataset. Configure your local paths in `fpd_seg/config/config.py` or override them from the command line.

Expected directory structure:

```text
data/
└── FPD-SegDSA/
    ├── train/
    │   ├── images/   # sequence files, for example 0.npy, 1.npy, ...
    │   └── labels/   # masks, for example label_s0.png, label_s1.png, ...
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

The dataloader expects each image sample as a NumPy sequence array and a corresponding binary vessel mask.

## Installation

```bash
conda env create -f environment.yml
conda activate fpd-maxip
```

## Training

```bash
python train.py \
  --model_type New_Mamba_Net \
  --batch_size 64 \
  --tag FPD_SegDSA
```

TensorBoard logs are written to `runs/<experiment_id>/`:

```bash
tensorboard --logdir runs
```

You can override dataset paths without editing source code:

```bash
python train.py --opts \
  DATASET.TRAIN_IMAGE_PATH /path/to/train/images \
  DATASET.TRAIN_LABEL_PATH /path/to/train/labels \
  DATASET.VAL_IMAGE_PATH /path/to/val/images \
  DATASET.VAL_LABEL_PATH /path/to/val/labels
```

## Testing

Due to the large file size, the pretrained model checkpoint is provided via Google Drive. Download the pretrained checkpoint from the release link below, then place it at:

```text
checkpoints/New_Mamba_Net_FPD_SegDSA/best_model.pth
```

Checkpoint link: 

- **Checkpoint**: [Download from Google Drive](https://drive.google.com/drive/folders/1hFghqzO3E4eeEjH8v-giJsi9qwZjgrdi?dmr=1&ec=wgc-drive-hero-goto)


Checkpoint files are ignored by Git because the released weight is larger than GitHub's normal file-size limit.

```bash
python test.py \
  --model_path checkpoints/New_Mamba_Net_FPD_SegDSA \
  --opts \
  DATASET.TEST_IMAGE_PATH /path/to/test/images \
  DATASET.TEST_LABEL_PATH /path/to/test/labels
```

## Before Publishing to GitHub

Keep only source code, configs, lightweight figures, and documentation in the repository. Do not upload:

- `runs/`
- `__pycache__/`
- `outputs/predictions/`
- `checkpoints/`
- `debug_batches/`
- local datasets
- model checkpoints unless they are intentionally released through GitHub Releases or another model hosting service

## Acknowledgements

This project was developed with reference to the public [DIAS](https://github.com/lseventeen/DIAS) codebase and has been reorganized around the FPD-SegDSA training workflow, package layout, configuration defaults, and experiment outputs.
