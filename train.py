import os
import argparse
from loguru import logger
from fpd_seg.data import build_train_loader
from fpd_seg.training.trainer import Trainer
from fpd_seg.common.helpers import seed_torch
from fpd_seg.objectives.losses import *
from datetime import datetime
from fpd_seg.config.config import get_config
from fpd_seg.architectures import build_model
from fpd_seg.training.lr_scheduler import build_scheduler
from fpd_seg.training.optimizer import build_optimizer
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_option():
    parser = argparse.ArgumentParser("FPD-SegDSA training")
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument("--tag", default="FPD_SegDSA", help='tag of experiment')
    parser.add_argument("-mt", "--model_type", default="New_Mamba_Net")
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help="batch size for single GPU")
    parser.add_argument('-ed', '--enable_distributed', help="training without DDP",
                        required=False, action="store_true")
    parser.add_argument('-ws', '--world_size', type=int,
                        help="process number for DDP")
    args = parser.parse_args()
    config = get_config(args)

    return args, config


def main(config):
    if config.DIS:
        mp.spawn(main_worker,
                 args=(config,),
                 nprocs=config.WORLD_SIZE,)
    else:
        main_worker(0, config)


def main_worker(local_rank, config):
    if local_rank == 0:
        config.defrost()
        config.EXPERIMENT_ID = f"{config.EXPERIMENT_ID}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        config.freeze()
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    torch.cuda.set_device(local_rank)
    if config.DIS:
        dist.init_process_group(
            "nccl", init_method='env://', rank=local_rank, world_size=config.WORLD_SIZE)
    seed = config.SEED + local_rank
    seed_torch(seed)
    cudnn.benchmark = True

    train_loader, val_loader = build_train_loader(config)  # 数据加载
    model, is_2d = build_model(config)  # 模型构建

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    if config.DIS:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    logger.info(f'\n{model}\n')
    # loss = CE_DiceLoss()
    # loss = SoftDiceLoss()
    loss = DC_and_CE_loss({}, {})  # 损失函数
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    trainer = Trainer(config=config,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      model=model.cuda(),
                      is_2d=is_2d,
                      loss=loss,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10000"
    _, config = parse_option()

    main(config)
