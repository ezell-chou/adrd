import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import pathlib
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, split_ratio: float = 0.8, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.split_ratio = split_ratio
        self.is_train = is_train
        self.image_paths, self.labels = [], []

        # 1. 递归扫 real/ fake/ 子目录
        for label_name, folder_id in [('real', 0), ('fake', 1)]:
            label_dir = Path(root_dir) / label_name
            if not label_dir.is_dir():
                continue
            for pt_path in label_dir.rglob('*.pt'):   # 递归所有 .pt
                tensor, inner_label = self._load_label(str(pt_path))
                label = inner_label if inner_label is not None else folder_id
                self.image_paths.append(str(pt_path))
                self.labels.append(label)

        if len(self.image_paths) == 0:
            raise RuntimeError(f'No .pt files found in {root_dir}/(real|fake)')

        # 2. 训练/验证划分
        if split_ratio < 1.0:
            self.train_paths, self.valid_paths, self.train_labels, self.valid_labels = \
                train_test_split(self.image_paths, self.labels,
                                 test_size=1 - self.split_ratio,
                                 stratify=self.labels,
                                 random_state=42)
        else:
            self.train_paths, self.valid_paths = self.image_paths, []
            self.train_labels, self.valid_labels = self.labels, []
        
        # 3. 设置当前使用的indices（支持shuffle）
        self._set_current_indices()

    def _set_current_indices(self):
        """设置当前使用的indices"""
        if self.is_train:
            self.indices = np.arange(len(self.train_paths))
        else:
            self.indices = np.arange(len(self.valid_paths))

    def shuffle_indices(self):
        """用于评估时引入随机性"""
        np.random.shuffle(self.indices)

    # ---------- 工具：统一读 .pt 返回 (tensor, label) ----------
    def _load_label(self, pt_path):
        obj = torch.load(pt_path, map_location='cpu', weights_only=True)
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return obj[0], int(obj[1])
        if isinstance(obj, dict):
            tensor = obj.get('tensor') or obj.get('data')
            label  = obj.get('label')  or obj.get('is_fake')
            if label is None:
                raise ValueError(f'{pt_path} 里没有找到 label')
            return tensor, int(label)
        # 仅 tensor，无标签 → 抛错
        raise ValueError(f'{pt_path} 仅保存 tensor，未存标签')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        # 按训练/验证取路径
        real_idx = self.indices[idx]
        
        if self.is_train:
            path, label = self.train_paths[real_idx], self.train_labels[real_idx]
        else:
            path, label = self.valid_paths[real_idx], self.valid_labels[real_idx]

        tensor, _ = self._load_label(path)   # 这里只拿 tensor，label 用前面划分好的

        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        feat = tensor.float()
        if self.transform:
            feat = self.transform(feat)
        return feat, label

    def set_train_mode(self, is_train: bool = True):
        """切换训练/验证模式"""
        self.is_train = is_train
        self._set_current_indices()


class EarlyStopping:
    def __init__(self, patience=8, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.best_state = None          # 存最好权重

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience
