# Temporal Phase-Difference Guided Spatiotemporal Learning for DSA Vessel Segmentation

**🎉🎉🎉  The paper associated with this project has been provisionally accepted by [MICCAI 2026](https://conferences.miccai.org/2026/en/default.asp) after the rebuttal process. We are grateful for the reviewers' constructive feedback and are currently preparing the camera-ready version.**

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

This repository does not include the dataset and you can get datasets on **[DIAS](https://zenodo.org/records/11637181)** and **[DSCA](https://zenodo.org/records/11255024)**. Configure your local paths in `fpd_seg/config/config.py` or override them from the command line.

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

## Citation

If you find this repository useful for your research, please consider citing our work:

```text
@inproceedings{liu2026fpdsegdsa,
  title     = {Temporal Phase-Difference Guided Spatiotemporal Learning for DSA Vessel Segmentation},
  author    = {Liu, Kun and He, Ziyang and Zheng, Bin and Zhao, Wenyi and Zhu, Mengke and Xu, Weijin and Liu, Wentao and He, Zijun and Chen, Hanlin and Lu, Bofeng and Liang, Zhiyuan and Yang, Huihua},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2026},
  note      = {Provisionally Accepted}
}
```

The citation information will be updated after the final camera-ready version and publication details become available.

## Acknowledgements

This project was developed with reference to the public [DIAS](https://github.com/lseventeen/DIAS) codebase and has been reorganized around the FPD-SegDSA training workflow, package layout, configuration defaults, and experiment outputs.

The main innovative design, algorithmic ideas, and experimental implementation of this work were independently completed by the author. For any questions regarding the technical details or implementation of this work, please contact the author directly.

The author would also like to express sincere gratitude to [He Ziyang](https://github.com/hzyBupt)，[Xu Weijin](https://github.com/xwjBupt) for their guidance, constructive comments, and support during the preparation of this manuscript. Their valuable feedback on academic writing and paper revision greatly improved the clarity and presentation of this work.


