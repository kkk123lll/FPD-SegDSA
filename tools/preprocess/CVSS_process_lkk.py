import os
import cv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import shutil
import re

class DIAS_process(object):
    def __init__(self, data_path, process_data_path, resample=True, new_slice=8, num_sequence=60, is_overwrite=False, save_png=True) -> None:
        if resample:
            assert new_slice is not None

        self.resample = resample
        self.data_path = data_path
        self.process_data_path = process_data_path
        self.num_sequence = num_sequence
        self.new_slice = new_slice
        self.save_png = save_png
        if is_overwrite and isdir(self.process_data_path):
            shutil.rmtree(self.process_data_path)
        # self.training_images_path = os.path.join(
        #     process_data_path, "training", "images")
        # self.training_labels_path = os.path.join(
        #     process_data_path, "training", "labels")
        # self.test_images_path = os.path.join(
        #     process_data_path, "test", "images")
        # self.test_labels_path = os.path.join(
        #     process_data_path, "test", "labels")
        # self.val_images_path = os.path.join(
        #     process_data_path, "validation", "images")
        # self.val_labels_path = os.path.join(
        #     process_data_path, "validation", "labels")
        self.images_path = os.path.join(
            process_data_path, "images")
        self.labels_path = os.path.join(
            process_data_path, "labels")

        # os.makedirs(self.training_images_path, exist_ok=True)
        # os.makedirs(self.training_labels_path, exist_ok=True)
        # os.makedirs(self.test_images_path, exist_ok=True)
        # os.makedirs(self.test_labels_path, exist_ok=True)
        # os.makedirs(self.val_images_path, exist_ok=True)
        # os.makedirs(self.val_labels_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.labels_path, exist_ok=True)

    def process(self):
        image_path = os.path.join(self.data_path, "images")
        label_path = os.path.join(self.data_path, "labels")
        
        image_files = list(sorted(os.listdir(image_path)))
        label_files = list(sorted(os.listdir(label_path)))
        slice_count = []
        sequences_list = []
        # image_num = 0
        # mean = 0
        # std = 0
        sequence_lists = self.extract_timestamp_info_image(image_path)
        label_lists = self.extract_timestamp_info_label(label_path)

        for i in range(len(sequence_lists)):
            slice_count_each_sequence = 0
            image_each_slice = []
            for j in image_files:
                # match = re.search(r'image_(.*)_i\d+', j)
                match = re.match(r'^image_(s\d+)_i\d+\.png$', j)
                if match:
                    result = match.group(1)  # → '20170103001655_06'
                if result == sequence_lists[i]:
                    slice_count_each_sequence += 1
                    img = cv2.imread(os.path.join(image_path, j), 0)/255
                    # image_num += 1
                    # mean += img.mean()
                    # std += img.std()
                    image_each_slice.append(img)
                    # print(j)
            slice_count.append(slice_count_each_sequence)
            sequences_list.append(np.array(image_each_slice))
        # mean /= image_num
        # std /= image_num
        # h, w = sequences_list[0].shape[1:]
        h, w = 800, 800
        if self.resample:
            new_shape = [self.new_slice, h, w]

        else:
            max_length = max(seq.shape[0] for seq in sequences_list)
        image_list = []
        for s in sequences_list:
            mean = np.mean(s)
            std_dev = np.std(s)
            # print(s)

            # 动态获取当前序列的高度和宽度
            original_shape = s.shape
            h, w = original_shape[1], original_shape[2]

            if self.resample:
                
                # 新形状：仅修改深度，保持高度和宽度不变
                new_shape = [self.new_slice, h, w]

                s = resize(s, new_shape, order=3,
                           mode="edge", anti_aliasing=False)
                if not self.save_png:

                    s = (s - mean) / std_dev
            else:
                aligned_array = np.full((max_length, h, w), 255)
            # aligned_array = np.ones((max_length, h, w), dtype=np.float32)
                aligned_array[:s.shape[0]] = s

                s = (aligned_array-mean) / \
                    std_dev if not self.save_png else aligned_array

            # mn = sequence.mean()
            # std = sequence.std()
            # print(sequence.shape, sequence.dtype, mn, std)
            # sequence = (sequence - mn) / (std + 1e-8)
            # image_full.append(ToTensor()(sequence))
            image_list.append(s)

        label_list = []
        for i in range(len(label_lists)):
            label_slice = []
            for j in label_files:
                result = j.replace("label_", "").replace(".png", "")
                if result == label_lists[i]:
                    label = cv2.imread(os.path.join(label_path, j), 0)
                    # # 调整标签分辨率为 800x800
                    # label = resize(label, (800, 800), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                    label_slice.append(
                        np.where(label >= 100, 255, 0).astype(np.uint8))
                    # print(j)
            label = np.array(label_slice).max(axis=0)
            label_list.append(label)
        # train_seq, test_seq, train_lab, test_lab = train_test_split(
        #     image_list, label_list, test_size=1/3, random_state=0)
        # train_seq, val_seq, train_lab, val_lab = train_test_split(
        #     train_seq, train_lab, test_size=0.25, random_state=0)
        # if self.save_png:
        #     self.save_seq_png(train_seq, self.training_images_path)
        #     self.save_seq_png(test_seq, self.test_images_path)
        #     self.save_seq_png(val_seq, self.val_images_path)

        # else:
        #     self.save_seq_npy(train_seq, self.training_images_path)
        #     self.save_seq_npy(test_seq, self.test_images_path)
        #     self.save_seq_npy(val_seq, self.val_images_path)
        # self.save_lab_png(train_lab, self.training_labels_path)
        # self.save_lab_png(test_lab, self.test_labels_path)
        # self.save_lab_png(val_lab, self.val_labels_path)
        
        if self.save_png:
            self.save_seq_png(image_list, self.images_path)
            self.save_lab_png(label_list, self.labels_path)
        else:
            self.save_seq_npy(image_list, self.images_path)
            self.save_lab_png(label_list, self.labels_path)

    def save_seq_png(self, seqs_list, path):
        for id_s, seq in enumerate(seqs_list):
            for id_i, img in enumerate(seq):
                file = f"image_s{id_s}_i{id_i}.png"
                cv2.imwrite(f"{path}/{file}", img*255)
                print(f'save_seqs : {file}')

    def save_lab_png(self, labs_list, path):
        for id_s, lab in enumerate(labs_list):
            file = f"label_s{id_s}.png"
            cv2.imwrite(f"{path}/{file}", lab)
            print(f'save_labs : {file}')

    def save_seq_npy(self, seqs_list, path):
        for id_s, seq in enumerate(seqs_list):
            np.save(os.path.join(path, f"{id_s}.npy"), seq)

            print(f'save_seqs : {id_s}.npy')

    def save_lab_npy(self, labs_list, path):
        for id_s, lab in enumerate(labs_list):
            np.save(os.path.join(path, f"{id_s}.npy"), lab/255)
            print(f'save_labs : {id_s}.npy')

    def extract_timestamp_info_image(self, folder_path):
        """
        提取文件夹中所有符合格式的文件名中的时间戳信息。
        
        Args:
            folder_path (str): 文件夹路径
        
        Returns:
            list: 包含提取的时间戳信息的列表
        """
        # 定义匹配的正则表达式
        # pattern = r'image_(\d{14}_\d{2,4})_i\d+\.png'
        pattern = r'image_(s\d+)_i\d+\.png'
        # pattern = r'image_(\d{3})_i\d+\.png'
        result = set()

        # 遍历文件夹中的文件
        for filename in os.listdir(folder_path):
            match = re.match(pattern, filename)
            if match:
                # 提取符合格式的部分
                result.add(match.group(1))
        
        return sorted(list(result))
    
    def extract_timestamp_info_label(self, folder_path):
        """
        从文件名列表中提取时间戳信息（适用于 label_*.png 格式）。
        
        Args:
            filenames (list): 文件名字符串列表，如 ['label_20170407000150_13.png', ...]
        
        Returns:
            list: 去重并排序后的时间戳信息列表，如 ['20170407000150_13', ...]
        """
        # 定义匹配的正则表达式
        # pattern = r'label_(\d{14}_\d{2,4})\.png'
        pattern = r'label_(s\d+)\.png'
        # pattern = r'label_(\d{3})\.png'
        result = set()

        # 遍历文件夹中的文件
        for filename in os.listdir(folder_path):
            match = re.match(pattern, filename)
            if match:
                # 提取符合格式的部分
                result.add(match.group(1))
        
        return sorted(list(result))


class DIAS_unlabel_process(DIAS_process):
    def __init__(self, data_path, process_data_path, new_slice=8, num_sequence=60, is_overwrite=True) -> None:
        self.data_path = data_path
        self.process_data_path = process_data_path
        self.num_sequence = num_sequence
        self.new_slice = new_slice

        if is_overwrite and isdir(self.process_data_path):
            shutil.rmtree(self.process_data_path)

        os.makedirs(self.process_data_path, exist_ok=True)

    # def process(self):

    #     image_files = list(sorted(os.listdir(self.data_path)))

    #     slice_count = []
    #     sequences_list = []

    #     for i in range(1, self.num_sequence + 1):
    #         slice_count_each_sequence = 0
    #         image_each_slice = []

    #         for j in image_files:
    #             if int(j[:2]) == i:
    #                 slice_count_each_sequence += 1
    #                 img = cv2.imread(os.path.join(self.data_path, j), 0)
    #                 image_each_slice.append(img)

    #         slice_count.append(slice_count_each_sequence)

    #         if len(image_each_slice) > 0:
    #             sequences_list.append(np.array(image_each_slice))
    #         else:
    #             print(f"Warning: sequence {i} has no images.")

    #     image_list = []

    #     for s in sequences_list:
    #         # s.shape = [T, H, W]
    #         original_shape = s.shape
    #         h, w = original_shape[1], original_shape[2]

    #         # 只修改时间帧数，不修改高度和宽度
    #         new_shape = [self.new_slice, h, w]

    #         sequence = resize(
    #             s,
    #             new_shape,
    #             order=3,
    #             mode="edge",
    #             anti_aliasing=False
    #         )

    #         image_list.append(sequence)

    #     self.save_seq_png(image_list, self.process_data_path)
    def process(self):

        image_files = os.listdir(self.data_path)

        # 匹配 image_s60_i0.png / image_s61_i7.png 这种格式
        pattern = re.compile(r'^image_s(\d+)_i(\d+)\.png$')

        # 按原始序列号分组，例如 60、61、62
        sequence_dict = {}

        for filename in image_files:
            match = pattern.match(filename)

            if not match:
                print(f"Skip unmatched file: {filename}")
                continue

            old_seq_id = int(match.group(1))    # 例如 s60 -> 60
            frame_id = int(match.group(2))      # 例如 i0 -> 0

            if old_seq_id not in sequence_dict:
                sequence_dict[old_seq_id] = []

            sequence_dict[old_seq_id].append((frame_id, filename))

        image_list = []

        # old_seq_id 会按照 60, 61, 62 ... 排序
        # append 到 image_list 后，save_seq_png 会自动从 s0 开始重新编号
        for new_seq_id, old_seq_id in enumerate(sorted(sequence_dict.keys())):

            frame_files = sorted(sequence_dict[old_seq_id], key=lambda x: x[0])

            image_each_slice = []

            for frame_id, filename in frame_files:
                img = cv2.imread(os.path.join(self.data_path, filename), 0)

                if img is None:
                    print(f"Warning: failed to read {filename}")
                    continue

                # 和 DIAS_process 保持一致，归一化到 0~1
                img = img / 255.0

                image_each_slice.append(img)

            if len(image_each_slice) == 0:
                print(f"Warning: sequence s{old_seq_id} has no valid images.")
                continue

            s = np.array(image_each_slice)

            # s.shape = [T, H, W]
            h, w = s.shape[1], s.shape[2]

            # 只修改帧数，不修改图像宽高
            new_shape = [self.new_slice, h, w]

            sequence = resize(
                s,
                new_shape,
                order=3,
                mode="edge",
                anti_aliasing=False
            )

            image_list.append(sequence)

            print(
                f"Processed: old s{old_seq_id} -> new s{new_seq_id}, "
                f"{s.shape} -> {sequence.shape}"
            )

        self.save_seq_png(image_list, self.process_data_path)


def main():
    # data_path = "/ai/data/data/vessel/DIAS/DSA"
    # process_data_path = "/ai/data/data/vessel/DIAS/DSA/data"
    data_path = "/ai/LKK/0_A_LKK/Datasets/DIAS/DIAS/unlabeled_DSA"
    process_data_path = "/ai/LKK/0_A_LKK/Datasets/DIAS/DIAS_lkk/png/unlabeled_DSA"
    # data_path = "/home/lwt/data/CVSS/DSA_new"
    # process_data_path="/home/lwt/data/CVSS/unlabel"
    # dp = DIAS_process(data_path, process_data_path, resample=True,
                    #   save_png=False, is_overwrite=True)
    dp = DIAS_unlabel_process(data_path, process_data_path)
    dp.process()


if __name__ == '__main__':
    main()
