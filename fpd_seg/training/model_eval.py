from fpd_seg.architectures import build_model


def print_model_parm_nums(model):
    total = sum(param.nelement() for param in model.parameters())
    print("  + Number of params: %.2fM" % (total / 1e6))
