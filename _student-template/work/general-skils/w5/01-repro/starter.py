import json
import random
import numpy as np
from pathlib import Path


def seed_everything(seed):
    """设定随机种子确保实验可复现。返回一次抽样字典以证明确定性。"""
    random.seed(seed)
    np.random.seed(seed)
    return {
        "python": random.random(),
        "numpy": float(np.random.random())
    }


def save_run(path, config, metrics):
    """把一次实验的配置与指标保存为 JSON（确定性排序，便于比对）。"""
    data = {
        'config': config,
        'metrics': metrics
    }
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2)

def load_run(path):
    """读取实验记录，返回 (config, metrics)。"""
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    return data['config'], data['metrics']
