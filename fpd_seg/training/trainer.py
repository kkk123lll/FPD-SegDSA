import os
from re import L
import time
import math
import torch
from loguru import logger
from tqdm import tqdm
from fpd_seg.common.helpers import to_cuda
from fpd_seg.common.metrics import AverageMeter, get_metrics, get_metrics
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from fpd_seg.objectives.losses import *
import imageio
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, config, train_loader, val_loader, model, is_2d, loss, optimizer, lr_scheduler):
        self.config = config

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        self.loss_ce = DC_and_CE_loss({}, {}, weight_dice=0)
        self.model = model
        self.is_2d = is_2d
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_steps = len(self.train_loader)
        if self._get_rank() == 0:
            self.checkpoint_dir = os.path.join(
                config.SAVE_DIR, config.EXPERIMENT_ID)

            os.makedirs(self.checkpoint_dir)
            self.writer = SummaryWriter(
                log_dir=os.path.join(config.TENSORBOARD.LOG_DIR, config.EXPERIMENT_ID)
            )
        else:
            self.writer = None
          # MONITORING
        self.improved = True
        self.not_improved_count = 0
        self.mnt_best = -math.inf if self.config.TRAIN.MNT_MODE == 'max' else math.inf

    def train(self):

        for epoch in range(1, self.config.TRAIN.EPOCHS+1):

            if self.config.DIS:
                self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.config.TRAIN.VAL_NUM_EPOCHS == 0:
                results = self._valid_epoch(epoch)
                if self._get_rank() == 0:
                    logger.info(f'## Info for epoch {epoch} ## ')
                    for k, v in results.items():
                        logger.info(f'{str(k):15s}: {v}')
                    if self.config.TRAIN.MNT_MODE != 'off' and epoch >= 10:
                        try:
                            if self.config.TRAIN.MNT_MODE == 'min':
                                self.improved = (
                                    results[self.config.TRAIN.MNT_METRIC] <= self.mnt_best)
                            else:
                                self.improved = (
                                    results[self.config.TRAIN.MNT_METRIC] >= self.mnt_best)
                        except KeyError:
                            logger.warning(
                                f'The metrics being tracked ({self.config.TRAIN.MNT_METRIC}) has not been calculated. Training stops.')
                            break

                        if self.improved:
                            self.mnt_best = results[self.config.TRAIN.MNT_METRIC]
                            self.not_improved_count = 0
                        else:
                            self.not_improved_count += 1
                        if self.not_improved_count >= self.config.TRAIN.EARLY_STOPPING:
                            logger.info(
                                f'\nPerformance didn\'t improve for {self.config.TRAIN.EARLY_STOPPING} epochs')
                            logger.warning('Training Stoped')
                            break

            # SAVE CHECKPOINT
            if self._get_rank() == 0:
                self._save_checkpoint(epoch, save_best=self.improved)
        if self.writer is not None:
            self.writer.close()
        return self.checkpoint_dir

    def _train_epoch(self, epoch):
        wrt_mode = "train"
        self.model.train()

        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        tic = time.time()

        for idx, (img, gt) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            img = to_cuda(img)
            gt = to_cuda(gt)
            if not self.is_2d:
                img = img.unsqueeze(1)  # 在第1维增加一个维度，不改内容只改形状，适合3D模型的输入，原来是(N,C,H,W)变成(N,1,C,H,W)
            
            # 🧠 检查输入是否异常
            if torch.isnan(img).any() or torch.isinf(img).any():
                print(f"[Warning] NaN/Inf detected in input at epoch {epoch}, batch {idx}, skipping batch.")
                continue

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.config.AMP):
                # pre, pre_gru, fused = self.model(img)
                # pre = torch.sigmoid(pre)
                pre= self.model(img)

                # 🧠 检查模型输出
                if torch.isnan(pre).any() or torch.isinf(pre).any():
                    print(f"[Warning] NaN/Inf detected in model output at epoch {epoch}, batch {idx}, skipping batch.")
                    # # === 创建保存目录 ===
                    # save_dir = "debug_batches"
                    # os.makedirs(save_dir, exist_ok=True)

                    # # 保存tensor以便后续复现
                    # torch.save({"inputs": img.detach().cpu(), "predict": pre.detach().cpu()},
                    #         f"{save_dir}/batch_epoch{epoch}_idx{idx}.pt")

                    # # ======================
                    # # 1️⃣ 取输入序列的第一个样本
                    # # ======================
                    # inp = img[0, 0].detach().cpu().numpy()  # shape [D,H,W]
                    # D, H, W = inp.shape

                    # # ======================
                    # # 2️⃣ 模型输出是2D
                    # # ======================
                    # pred = pre[0, 0].detach().cpu().numpy()  # shape [H,W]

                    # # ========== 安全归一化函数 ==========
                    # def norm_to_uint8(x):
                    #     if np.isnan(x).all() or np.isinf(x).all():
                    #         return np.zeros_like(x, dtype=np.uint8)
                    #     x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                    #     vmin, vmax = x.min(), x.max()
                    #     if vmax - vmin < 1e-8:
                    #         return np.zeros_like(x, dtype=np.uint8)
                    #     x = (x - vmin) / (vmax - vmin)
                    #     return (x * 255).astype(np.uint8)

                    # # ======================
                    # # 3️⃣ 保存输入GIF
                    # # ======================
                    # inp_frames = [norm_to_uint8(inp[d]) for d in range(D)]
                    # gif_inp_path = f"{save_dir}/batch{idx}_epoch{epoch}_input.gif"
                    # imageio.mimsave(gif_inp_path, inp_frames, fps=2)
                    # print(f"Saved input GIF: {gif_inp_path}")

                    # # ======================
                    # # 4️⃣ 保存预测结果（单帧GIF）
                    # # ======================
                    # pred_frame = norm_to_uint8(pred)
                    # gif_pred_path = f"{save_dir}/batch{idx}_epoch{epoch}_predict.gif"
                    # imageio.mimsave(gif_pred_path, [pred_frame], fps=1)
                    # print(f"Saved predict GIF: {gif_pred_path}")

                    # # ======================
                    # # 5️⃣ 合成对比图（选用中间帧 + 预测结果）
                    # # ======================
                    # mid = D // 2
                    # inp_mid = norm_to_uint8(inp[mid])
                    # pred_resized = pred_frame
                    # if pred_resized.shape != inp_mid.shape:
                    #     from skimage.transform import resize
                    #     pred_resized = resize(pred_frame, inp_mid.shape, preserve_range=True).astype(np.uint8)

                    # combined = np.concatenate([inp_mid, pred_resized], axis=1)
                    # gif_combined_path = f"{save_dir}/batch{idx}_epoch{epoch}_compare.gif"
                    # imageio.mimsave(gif_combined_path, [combined], fps=1)
                    # print(f"Saved combined compare GIF: {gif_combined_path}")

                    # # ======================
                    # # 6️⃣ 保存静态拼图（8帧输入 + 预测）
                    # # ======================
                    # fig, axes = plt.subplots(2, D, figsize=(D*2, 4))
                    # for d in range(D):
                    #     axes[0, d].imshow(inp[d], cmap='gray')
                    #     axes[0, d].axis('off')
                    #     axes[0, d].set_title(f"In {d}")
                    # for d in range(D):
                    #     axes[1, d].imshow(pred, cmap='gray')
                    #     axes[1, d].axis('off')
                    #     axes[1, d].set_title("Pred")
                    # plt.tight_layout()
                    # plt.savefig(f"{save_dir}/batch{idx}_epoch{epoch}_grid.png")
                    # plt.close(fig)

                    continue

                loss = self.loss(pre, gt)
                # loss_s = self.loss(pre, gt)
                # loss_t = self.loss(pre_gru, gt)
                # loss_con = self.loss(pre_gru, pre)
                # loss = loss_s + loss_t + loss_con

                # 🧠 检查 loss 是否异常
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Warning] NaN/Inf loss detected at epoch {epoch}, batch {idx}, skipping batch.")
                    continue

            if self.config.AMP:
                self.scaler.scale(loss).backward()
                if self.config.TRAIN.DO_BACKPROP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.TRAIN.DO_BACKPROP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)

            # ✅ 使用稳定 softmax 并检测 NaN
            probs = torch.softmax(pre - pre.max(dim=1, keepdim=True)[0], dim=1)
            if torch.isnan(probs).any():
                print(f"[Warning] NaN detected in probs after softmax at epoch {epoch}, batch {idx}, skipping metrics.")
                continue

            preds_np = probs.cpu().detach().numpy()[:, 1, :, :]
            gts_np = gt.cpu().detach().numpy()

            # ⚙️ 封装 metrics，安全计算
            try:
                metrics = get_metrics(preds_np, gts_np)
                self._update_metrics(*metrics.values())
            except ValueError:
                print(f"[Warning] Invalid metric computation (NaN or single class) at epoch {epoch}, batch {idx}.")
                continue

            # self._update_metrics(
            #     *get_metrics(torch.softmax(pre, dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
            #     # *get_metrics(torch.softmax(pre - pre.max(dim=1, keepdim=True)[0], dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
            
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} |DSC {:.4f}  Acc {:.4f}  Sen {:.4f} Spe {:.4f}  IOU {:.4f} AUC {:.4f} clDice {:.4f}|B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.mean, *self._get_metrics_mean().values(), self.batch_time.mean, self.data_time.mean))
            tic = time.time()
            self.lr_scheduler.step_update(epoch * self.num_steps + idx)
        if self._get_rank() == 0:
            self.writer.add_scalar(f'{wrt_mode}/loss', self.total_loss.mean, epoch)
            for k, v in list(self._get_metrics_mean().items())[:-1]:
                self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
            for i, opt_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(
                    f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        # tbar = tqdm(self.val_loader, ncols=160)
        tbar = tqdm(self.val_loader)
        with torch.no_grad():
            for idx, (img, gt) in enumerate(tbar):
                img = to_cuda(img)
                gt = to_cuda(gt)
                if not self.is_2d:
                    img = img.unsqueeze(1)
                with torch.amp.autocast('cuda', enabled=self.config.AMP):

                    predict = self.model(img)
                    
                    # predict, pre_gru, fused = self.model(img)
                    # predict = torch.sigmoid(predict)
                    # loss_s = self.loss(predict, gt)
                    # loss_t = self.loss(pre_gru, gt)
                    # loss_con = self.loss(pre_gru, predict)
                    # loss = loss_s + loss_t + loss_con

                    loss = self.loss(predict, gt)

                self.total_loss.update(loss.item())
                
                self._update_metrics(
                    *get_metrics(torch.softmax(predict - predict.max(dim=1, keepdim=True)[0], dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
                tbar.set_description(
                'EVAL ({})  | Loss: {:.4f} |DSC {:.4f}  Acc {:.4f}  Sen {:.4f} Spe {:.4f}  IOU {:.4f} AUC {:.4f} |'.format(
                    epoch, self.total_loss.mean, *self._get_metrics_mean().values()))

        if self._get_rank() == 0:

            self.writer.add_scalar(f'{wrt_mode}/loss', self.total_loss.mean, epoch)
            for k, v in list(self._get_metrics_mean().items())[:-1]:
                self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)

        log = {
            'val_loss': self.total_loss.mean,
            **self._get_metrics_mean()
        }
        return log

    def _save_checkpoint(self, epoch, save_best=True):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'final_checkpoint.pth')
        logger.info(f'Saving a checkpoint: {filename}')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
            logger.info(f"Saving current best: {filename}")
            torch.save(state, filename)

        return filename

    def _get_rank(self):
        """get gpu id in distribution training."""
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.Pre = AverageMeter()
        self.DSC = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.iou = AverageMeter()
        self.VC = AverageMeter()
        self.cldice = AverageMeter()

    def _update_metrics(self, DSC, acc, sen, spe, iou,auc, cldice):
        self.DSC.update(DSC)
        self.Pre.update(Pre)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.iou.update(iou)
        self.auc.update(auc)
        self.cldice.update(cldice)

    def _get_metrics_mean(self):

        return {
            
            "DSC_mean": self.DSC.mean,
            "Pre_mean": self.Pre.mean,
            "Acc_mean": self.acc.mean,
            "Sen_mean": self.sen.mean,
            "Spe_mean": self.spe.mean,
            "IOU_mean": self.iou.mean,
            "AUC_mean": self.auc.mean,
            "cldice_mean": self.cldice.mean,
        }
    def _get_metrics_std(self):

        return {
            
            "DSC_std": self.DSC.std,
            "Pre_std": self.Pre.std,
            "Acc_std": self.acc.std,
            "Sen_std": self.sen.std,
            "Spe_std": self.spe.std,
            "IOU_std": self.iou.std,
            "AUC_std": self.auc.std,
            "cldice_std": self.cldice.std,
        }

