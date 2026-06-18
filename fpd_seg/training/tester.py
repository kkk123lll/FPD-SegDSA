import time
from openpyxl import Workbook
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm
from fpd_seg.training.trainer import Trainer
import torch.nn.functional as F
from fpd_seg.common.helpers import dir_exists, remove_files,to_cuda,recompone_overlap
from fpd_seg.common.metrics import get_metrics, count_connect_component,get_color,AverageMeter
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd

class Tester(Trainer):
    def __init__(self,config, test_loader, model, save_dir, is_2d,  model_name):
        # super(Trainer, self).__init__()
        self.config = config
        self.test_loader = test_loader
        self.model = model
        self.is_2d = is_2d
        self.model_name = model_name
        self.save_path = "outputs/predictions/" + save_dir
        self.labels_path = config.DATASET.TEST_LABEL_PATH
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        dir_exists(self.save_path)
        remove_files(self.save_path)
        
        cudnn.benchmark = True

    def test(self):
        
        self.model.eval()
        self._reset_metrics()
        self.VC=AverageMeter()
        gts = self.get_labels()
        tbar = tqdm(self.test_loader, ncols=150)

        pres = []
        infer_times = []   # 新增：记录每个样本的推理时间（毫秒）
        with torch.no_grad():
            for img, _ in tbar:
                img = to_cuda(img)
                if not self.is_2d:
                    img = img.unsqueeze(1)

                # ------ 开始计时 ------
                torch.cuda.synchronize()
                start_time = time.time()
                # ----------------------

                with torch.amp.autocast('cuda', enabled=self.config.AMP):
                    pre = self.model(img)  # S,T,F

                # ------ 结束计时 ------
                torch.cuda.synchronize()
                end_time = time.time()
                infer_ms = (end_time - start_time) * 1000   # 转换为毫秒
                infer_times.append(infer_ms)
                # ----------------------

                pre = torch.softmax(pre, dim=1)[:,1,:,:]
                # ori, pre = self.hysteresis_from_softmax(pre, h_thresh=0.5, l_thresh=0.3)
                pres.extend(pre)
        
        patch_counts = self.test_loader.dataset.get_patch_counts()

        pres = torch.stack(pres, 0).cpu()

        all_preds = pres.cpu().detach().numpy()   # (N_patch, H, W)
        all_preds = np.expand_dims(all_preds, axis=1)

        start = 0
        predict_list = []
        predict_b_list = []

        for i, count in enumerate(patch_counts):
            end = start + count

            preds_img = all_preds[start:end]   # 当前图像的所有 patch
            gt = gts[i]
            H, W = gt.shape

            pad_h = (self.stride - (H - self.patch_size[0]) % self.stride) % self.stride
            pad_w = (self.stride - (W - self.patch_size[1]) % self.stride) % self.stride

            new_h = H + pad_h
            new_w = W + pad_w

            pres_img = recompone_overlap(
                preds_img,
                new_h, new_w,
                self.stride, self.stride
            )

            pred = pres_img[0, 0, :H, :W]
            pred_b = np.where(pred >= 0.6, 1, 0)

            predict_list.append(pred)
            predict_b_list.append(pred_b)

            start = end

        
        num_data = len(predict_list)

        for j in range(num_data):
            predict = predict_list[j]
            predict_b = predict_b_list[j]
            gt = gts[j]

            cv2.imwrite(self.save_path + f"/gt{j}.png", np.uint8(gt*255))
            cv2.imwrite(self.save_path + f"/pre{j}.png", np.uint8(predict*255))
            cv2.imwrite(self.save_path + f"/pre_b{j}.png", np.uint8(predict_b*255))
            cv2.imwrite(self.save_path + f"/color_b{j}.png", get_color(predict_b, gt))

            metrics = get_metrics(predict_b, gt, run_clDice=True)
            self._update_metrics(*metrics.values())

            dsc_value = metrics["DSC"]
            iou_value = metrics["IOU"]
            file_name = f"pre{j}_dsc{dsc_value:.2f}_iou{iou_value:.2f}.png"
            cv2.imwrite(self.save_path + "/" + file_name, np.uint8(predict * 255))

            self.VC.update(count_connect_component(predict_b, gt))


###########
        # 创建 Excel
        wb = Workbook()
        ws = wb.active
        ws.title = "Inference Speed"

        # 写表头
        ws.append(["Sample Index", "Inference Time (ms)"])

        # 写数据
        for idx, t in enumerate(infer_times):
            ws.append([idx, t])

        # 保存文件
        save_file = self.save_path + "/inference_speed.xlsx"
        wb.save(save_file)

        print(f"推理速度已保存到 {save_file}")
#############
        mean_data = list(self._get_metrics_mean().values())
        std_data = list(self._get_metrics_std().values())
        mean_data.append(self.VC.mean)
        std_data.append(self.VC.std)
        columns = list(self._get_metrics_mean().keys())
        columns.append("VC")


        formatted_data = [f"{mean}$\\pm${std}" for mean, std in zip(mean_data, std_data)]

        # 创建一个字典，用于构造DataFrame
        data_dict = {col: [val] for col, val in zip(columns, formatted_data)}

        # 创建DataFrame
        df = pd.DataFrame(data_dict)
        # df = pd.DataFrame(data=np.array(data).reshape(1, len(columns)), index=[self.model_name], columns = columns)
       
        # logger.info(f"###### TEST EVALUATION ######")
        # logger.info(f'test time:  {self.batch_time.average}')
        # logger.info(f'     VC:  {self.VC.average}')
  
        

        df.to_csv(join(self.save_path, f"{self.model_name}_result.cvs"))
        for k, v in self._get_metrics_mean().items():
            logger.info(f'{str(k):5s}: {v}')

        for k, v in self._get_metrics_std().items():
            logger.info(f'{str(k):5s}: {v}')
        
        logger.info(f'VC_mean: {self.VC.mean}')
       
        logger.info(f'VC_std: {self.VC.std}')
     
            

    def get_labels(self):
        labels = subfiles(self.labels_path, join=False, suffix='png')
        label_list = []
        for i in range(len(labels)):
            gt = cv2.imread(os.path.join(self.labels_path, f'label_s{i}.png'), 0)
            gt = np.array(gt/255)
            label_list.append(gt)
        return label_list
    
    def hysteresis_from_softmax(self, pre, h_thresh=0.2, l_thresh=0.1):
        """
        pre: torch.Tensor, shape = (B, H, W), 值域 (0,1)
        return:
            bin_img  : (B, H, W)
            gbin_img : (B, H, W)
        """

        # 补 channel 维 → (B,1,H,W)
        img = pre.unsqueeze(1)

        # high / low threshold
        high = img >= h_thresh
        low  = (img >= l_thresh) & (img < h_thresh)

        # 初始强边
        gbin = high.clone()

        # 8 邻域卷积核
        kernel = torch.ones((1, 1, 3, 3), device=pre.device)

        prev = torch.zeros_like(gbin)
        while not torch.equal(prev, gbin):
            prev = gbin.clone()

            # 8 邻域是否有激活
            neighbor = F.conv2d(
                gbin.float(), kernel, padding=1
            ) > 0

            # hysteresis 规则
            gbin = gbin | (low & neighbor)

        # 输出去掉 channel 维
        bin_img  = high.squeeze(1).float()
        gbin_img = gbin.squeeze(1).float()

        return bin_img, gbin_img
        
