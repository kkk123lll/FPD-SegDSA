import os
import torch
from thop import profile
import argparse
from loguru import logger
from fpd_seg.data import build_test_loader
from fpd_seg.training.tester import Tester
from fpd_seg.common.helpers import load_checkpoint
from fpd_seg.config.config import get_val_config
from fpd_seg.architectures import build_model
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser("FPD-SegDSA evaluation")
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('-mp', '--model_path', type=str,
                        default='checkpoints/New_Mamba_Net_FPD_SegDSA',
                        help='path to a checkpoint file or experiment directory')
    args = parser.parse_args()
    config = get_val_config(args)

    return args, config

# /path/to/project/outputs/predictions/New_Mamba_Net_FPD_SegDSA
# checkpoints/New_Mamba_Net_FPD_SegDSA
def main(config):
    save_dir = config.MODEL_PATH.split('/')[-1]
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    test_loader = build_test_loader(config)

    model_checkpoint = load_checkpoint(config.MODEL_PATH, True)
    config_chk = model_checkpoint["config"]
    model_name = config_chk.MODEL.TYPE
    model, is_2d = build_model(config_chk)
    
    model = model.to("cuda:0")
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters: {:.2f} M".format(total_params / 1e6))

    # ✅ Correct input shape: (B, C=1, D, H, W)
    dummy_input = torch.randn(1, 1, 8, 64, 64).to("cuda:0")

    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print("Total FLOPs: {:.2f} G".format(flops / 1e9))
        print("Total Params (from thop): {:.2f} M".format(params / 1e6))
    
    model.load_state_dict({k.replace('module.', ''): v for k,
                          v in model_checkpoint['state_dict'].items()})
    logger.info(f'\n{model}\n')
    tester = Tester(config=config,
                    test_loader=test_loader,
                    model=model.eval().cuda(),
                    save_dir=save_dir,
                    is_2d=is_2d,
                    model_name=model_name)
    tester.test()


if __name__ == '__main__':

    _, config = parse_option()
    main(config)
