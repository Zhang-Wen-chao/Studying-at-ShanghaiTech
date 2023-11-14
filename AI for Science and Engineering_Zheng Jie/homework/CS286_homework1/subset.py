import os
import shutil
import random

def create_small_dataset(source_dir, target_dir, num_samples_per_class):
    """
    从每个类别中随机选择 num_samples_per_class 个样本，并复制到目标目录。
    """
    classes = os.listdir(source_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for cls in classes:
        cls_source_dir = os.path.join(source_dir, cls)
        cls_target_dir = os.path.join(target_dir, cls)

        if not os.path.exists(cls_target_dir):
            os.makedirs(cls_target_dir)

        images = os.listdir(cls_source_dir)
        selected_images = random.sample(images, num_samples_per_class)

        for img in selected_images:
            src_path = os.path.join(cls_source_dir, img)
            dst_path = os.path.join(cls_target_dir, img)
            shutil.copy(src_path, dst_path)

# 设置参数
source_train_dir = '/home/student001/zhangwch/Studying-at-ShanghaiTech/AI for Science and Engineering_Zheng Jie/homework/CS286_homework1/BreakHis/train'
source_test_dir = '/home/student001/zhangwch/Studying-at-ShanghaiTech/AI for Science and Engineering_Zheng Jie/homework/CS286_homework1/BreakHis/test'
target_train_dir = '/home/student001/zhangwch/Studying-at-ShanghaiTech/AI for Science and Engineering_Zheng Jie/homework/CS286_homework1/BreakHis_small/train'
target_test_dir = '/home/student001/zhangwch/Studying-at-ShanghaiTech/AI for Science and Engineering_Zheng Jie/homework/CS286_homework1/BreakHis_small/test'
num_samples_per_class = 50  # 每个类别中要选择的样本数量

# 创建小型数据集
create_small_dataset(source_train_dir, target_train_dir, num_samples_per_class)
create_small_dataset(source_test_dir, target_test_dir, num_samples_per_class)
