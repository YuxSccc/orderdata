import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from model_common import MainModel
from pathlib import Path
import logging
import time
import wandb
from config import *
import torch.nn.functional as F
import os
from torchmetrics import AUROC, F1Score
from sklearn.metrics import precision_score, recall_score, f1_score
import bisect
import sys        
import inspect

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stderr)
    ]
)

class TradingDataset(Dataset):
    def __init__(self, file_list: list[str], seq_len: int, begin_skip: int = 100, end_skip: int = 100, cache_size: int = 8):
        """
        Args:
            file_list: 数据文件列表，每个文件会对应读取 _number/_event/_target/_flow 四个parquet文件
            seq_len: 每组数据包含的时间步数
            begin_skip: 开始时跳过的行数
            end_skip: 结束时跳过的行数
            cache_size: 缓存的文件数量（建议为batch_size的2倍）
        """
        self.seq_len = seq_len
        self.file_list = file_list
        self.begin_skip = begin_skip
        self.end_skip = end_skip
        self.cache_size = cache_size
        self.cache = {}  # 文件缓存
        
        # 存储每个文件的有效序列数和累积序列数
        self.file_sequences = []  # [(file_path, start_idx, num_sequences), ...]
        total_sequences = 0
        
        for file in file_list:
            # 读取number文件以获取总行数（其他文件行数应该一致）
            number_df = pd.read_parquet(f"{FILE_PREFIX}/{file}_number.parquet")
            valid_rows = len(number_df) - begin_skip - end_skip
            
            if valid_rows >= seq_len:
                num_sequences = valid_rows - seq_len + 1
                self.file_sequences.append((file, total_sequences, num_sequences))
                total_sequences += num_sequences
        
        self.total_sequences = total_sequences
        self.feature_names = set()

        self.start_indices = [start_idx for (_, start_idx, _) in self.file_sequences]

        # 初始化特征名称（只需要读取一个文件）
        first_file = file_list[0]
        event_df = self.read_parquet(f"{first_file}_event.parquet")
        self.feature_names.update(set(col.rsplit('_', 1)[0] for col in event_df.columns))
    
        self.last_file_idx = None
        self.last_file_info = None  # (file_path, start_idx, num_sequences)

        self.load_times = []  # 记录加载时间
        self.access_count = 0  # 记录访问次数

    def get_number_feature_dim(self):
        number_df = self.read_parquet(f"{self.file_list[0]}_number.parquet")
        return number_df.shape[1]

    def get_feature_name_list(self):
        return list(self.feature_names)

    def read_parquet(self, file_path: str):
        """读取parquet文件，使用缓存机制"""
        if file_path in self.cache:
            return self.cache[file_path]
        
        # 如果缓存已满，删除最早的文件
        if len(self.cache) >= self.cache_size:
            # 获取所有缓存的文件基础名称（不包含_number/_event等后缀）
            cached_base_files = set(f.rsplit('_', 1)[0] for f in self.cache.keys())
            # 找到最早的基础文件名
            oldest_base_file = min(cached_base_files)
            # 删除该文件相关的所有缓存
            for suffix in ['_number.parquet', '_event.parquet', '_target.parquet', '_flow.parquet']:
                self.cache.pop(f"{oldest_base_file}{suffix}", None)
        
        # 读取新文件并加入缓存
        df = pd.read_parquet(f"{FILE_PREFIX}/{file_path}")
        self.cache[file_path] = df
        return df

    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        start_time = time.time()
        self.access_count += 1
        
        # 检查索引范围
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError("Index out of range")

        if self.last_file_info is not None:
            last_file_idx, (file_path, start_idx, num_sequences) = self.last_file_idx, self.last_file_info
            if start_idx <= idx < start_idx + num_sequences:
                # 缓存命中，直接使用
                local_idx = idx - start_idx
            else:
                # 缓存未命中，需要更新缓存
                file_idx = bisect.bisect_right(self.start_indices, idx) - 1
                file_path, start_idx, num_sequences = self.file_sequences[file_idx]
                local_idx = idx - start_idx
                # 更新缓存
                self.last_file_idx = file_idx
                self.last_file_info = (file_path, start_idx, num_sequences)
        else:
            # 缓存为空，首次访问，进行二分查找
            file_idx = bisect.bisect_right(self.start_indices, idx) - 1
            file_path, start_idx, num_sequences = self.file_sequences[file_idx]
            local_idx = idx - start_idx
            # 更新缓存
            self.last_file_idx = file_idx
            self.last_file_info = (file_path, start_idx, num_sequences)

        # 计算开始和结束行号
        start_row = self.begin_skip + local_idx
        end_row = start_row + self.seq_len
        
        # 1. 读取并处理 number features
        number_df = self.read_parquet(f"{file_path}_number.parquet")
        other_features = number_df.iloc[start_row:end_row].values
        
        # 2. 读取并处理 event features
        event_df = self.read_parquet(f"{file_path}_event.parquet")
        feature_data = []
        
        # 按时间步处理event特征
        for step in range(self.seq_len):
            current_idx = start_row + step
            step_features = {}
            
            # 获取当前时间步的所有特征
            row = event_df.iloc[current_idx]
            feature_names = set(col.rsplit('_', 1)[0] for col in event_df.columns)
            
            for feature_name in feature_names:
                params_col = f"{feature_name}_params"
                categories_col = f"{feature_name}_categories"
                
                params = row[params_col] if params_col in event_df.columns else None
                categories = row[categories_col] if categories_col in event_df.columns else None
                
                if params is not None and not (isinstance(params, list) and params[0] is None):
                    if categories is None:
                        step_features[feature_name] = (params.tolist(),)
                    else:
                        step_features[feature_name] = (params.tolist(), categories)
            
            feature_data.append(step_features)
        
        # 3. 读取并处理 flow features
        flow_df = self.read_parquet(f"{file_path}_flow.parquet")
        flow_start = local_idx * self.seq_len
        flow_end = flow_start + self.seq_len
        
        # 处理每个时间步的flow特征
        for step in range(self.seq_len):
            current_flow_idx = flow_start + step
            flow_row = flow_df.iloc[current_flow_idx]
            
            for col in flow_df.columns:
                feature_name = col
                value = flow_row[col]
                
                if value is not None and not (isinstance(value, list) and value[0] is None):
                    # value = np.array(value).reshape(-1, len(value[0]) if isinstance(value[0], list) else 1)
                    if feature_name in feature_data[step]:
                        params, categories = feature_data[step][feature_name]
                        params = np.concatenate([params, value], axis=1)
                        if categories is None:
                            feature_data[step][feature_name] = (params.tolist(),)
                        else:
                            feature_data[step][feature_name] = (params.tolist(), categories)
                    else:
                        feature_data[step][feature_name] = (value,)
        
        # 4. 读取target（使用序列末尾对应的目标）
        target_df = self.read_parquet(f"{file_path}_target.parquet")
        target = target_df.iloc[local_idx + self.seq_len - 1].values

        # 记录加载时间
        load_time = time.time() - start_time
        self.load_times.append(load_time)
        
        return {
            'other_features': torch.FloatTensor(other_features),
            'feature_data': feature_data
        }, torch.LongTensor(target)

    def get_stats(self):
        """获取数据集访问统计信息"""
        if not self.load_times:
            return "No data loaded yet"
        
        avg_time = np.mean(self.load_times)
        max_time = np.max(self.load_times)
        min_time = np.min(self.load_times)
        items_per_sec = 1.0 / avg_time
        
        return {
            "avg_load_time": avg_time,
            "max_load_time": max_time,
            "min_load_time": min_time,
            "items_per_sec": items_per_sec,
            "total_accesses": self.access_count
        }

class OffsetSampler(Sampler):
    def __init__(self, data_source, start_idx=0):
        self.data_source = data_source
        self.start_idx = start_idx
        
    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))
    
    def __len__(self):
        return len(self.data_source) - self.start_idx

    def reset(self):
        self.current_start_idx = 0

# 在创建 DataLoader 时使用
def create_data_loader(dataset, batch_size, start_batch=0):
    start_idx = start_batch * batch_size
    sampler = OffsetSampler(dataset, start_idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    ), sampler

is_profiling = False

def check_nan_inf(model, optimizer):
    flag = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"参数 {name} 中存在 NaN 或 Inf")
            flag = True
    for param_group in optimizer.param_groups:
        for key, value in param_group.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"优化器参数 {key} 中存在 NaN 或 Inf")
                    flag = True

    for i, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"优化器状态第 {i} 个参数的 {key} 中存在 NaN 或 Inf")
                    flag = True

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"梯度 {name} 中存在 NaN 或 Inf")
                flag = True
    if flag:
        stack = inspect.stack()
        previous_frame = stack[1]  # 上一层
        module = inspect.getmodule(previous_frame[0])
        filename = previous_frame.filename
        line_number = previous_frame.lineno
        function_name = previous_frame.function
        print(f"Called from {function_name} in {filename} at line {line_number}")
    return flag

def train_epoch(model, train_loader, optimizer, device, scaler, 
                epoch, checkpoint_path, best_val_loss, patience_counter, scheduler, start_batch, batch_size):
    model.train()
    total_loss = 0
    correct = torch.zeros(4).to(device)
    total = 0
    last_print_time = time.time()
    last_checkpoint_time = time.time()
    checkpoint_interval = 300

    batch_start_time = time.time()
    moving_avg_time = 0
    alpha = 0.98  # 用于移动平均的平滑系数

    # 创建用于累计每个任务的 TP、FP、FN
    task_metrics = {
        'TP': [0, 0, 0, 0],
        'FP': [0, 0, 0, 0],
        'FN': [0, 0, 0, 0],
        'TN': [0, 0, 0, 0]
    }

    profile_count = 0
    sample_count = 0
    for batch_idx, (features, targets) in enumerate(train_loader):
        print(f"Batch {start_batch + batch_idx}/{len(train_loader)}")
        start_batch += 1
        targets = targets.to(device)
        other_features = features['other_features'].to(device, non_blocking=True)
        feature_data = features['feature_data']
        flag = False
        optimizer.zero_grad()
        pos_weights = [4.0, 4.0, 4.0, 4.0]
        if is_profiling:
            from torch.profiler import profile, record_function, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(other_features, feature_data)
                    loss = calculate_multi_task_loss(outputs, targets, pos_weights=pos_weights)
                    profile_count += 1
            if profile_count == 1:
                prof.export_chrome_trace("small_trace.json")
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                exit(0)
        else:
            with torch.amp.autocast('cuda'):
                outputs = model(other_features, feature_data)
                loss = calculate_multi_task_loss(outputs, targets, pos_weights=pos_weights)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Loss is NaN or Inf, outputs: {outputs}, targets: {targets}")
            exit(0)

        sample_count += 1
        wandb.log({"train_loss": loss})

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()

        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()

        # 更新每个任务的 TP、FP、FN、TN
        for task_idx in range(4):
            preds = predictions[:, task_idx]
            trues = targets[:, task_idx]

            task_metrics['TP'][task_idx] += ((preds == 1) & (trues == 1)).sum().item()
            task_metrics['FP'][task_idx] += ((preds == 1) & (trues == 0)).sum().item()
            task_metrics['FN'][task_idx] += ((preds == 0) & (trues == 1)).sum().item()
            task_metrics['TN'][task_idx] += ((preds == 0) & (trues == 0)).sum().item()
        # 计算并更新批处理时间
        batch_time = time.time() - batch_start_time
        moving_avg_time = alpha * batch_time + (1 - alpha) * moving_avg_time if batch_idx > 0 else batch_time
        iter_per_sec = 1 / moving_avg_time if moving_avg_time > 0 else 0
        # 打印简要的训练信息
        print(f'\rBatch: {start_batch + batch_idx}/{len(train_loader)}, '
            f'Loss: {loss.item():.4f}, '
            f'Speed: {iter_per_sec:.2f} iter/s, '
            f'ItemSpeed: {iter_per_sec * batch_size:.2f} items/s ', end='', flush=True)

        current_time = time.time()
        batch_start_time = current_time
        # 每隔一定时间或批次，打印和记录详细的性能指标
        if batch_idx % 200 == 0 or current_time - last_print_time >= 300:
            # 计算每个任务的指标
            metrics_str = ''
            for task_idx in range(4):
                TP = task_metrics['TP'][task_idx]
                FP = task_metrics['FP'][task_idx]
                FN = task_metrics['FN'][task_idx]
                TN = task_metrics['TN'][task_idx]

                precision = TP / (TP + FP + 1e-8)
                recall = TP / (TP + FN + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-8)

                metrics_str += (f'Task{task_idx+1} - Acc: {accuracy*100:.1f}%, '
                                f'Prec: {precision*100:.1f}%, '
                                f'Recall: {recall*100:.1f}%, '
                                f'F1: {f1*100:.1f}%\n')

            avg_loss = total_loss / sample_count
            logging.info(f'\nBatch: {start_batch + batch_idx}/{len(train_loader)}, Avg Loss: {avg_loss:.4f}\n{metrics_str}')

            last_print_time = current_time

            # 重置统计变量（可选）
            task_metrics = {
                'TP': [0, 0, 0, 0],
                'FP': [0, 0, 0, 0],
                'FN': [0, 0, 0, 0],
                'TN': [0, 0, 0, 0]
            }
            total_loss = 0.0
            sample_count = 0
        if current_time - last_checkpoint_time >= checkpoint_interval:
            batch_progress = batch_idx / len(train_loader)
            logging.info(f"Saving checkpoint at epoch {epoch}, batch progress: {batch_progress:.2%}")

            torch.save({
                'epoch': epoch,
                'batch_idx': batch_idx + start_batch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'patience_counter': patience_counter,
                'best_val_loss': min(best_val_loss, loss.item()),
                'total_batches': len(train_loader),
            }, checkpoint_path)
            
            logging.info(f"Saving checkpoint at epoch {epoch}, batch progress: {batch_progress:.2%}")
            last_checkpoint_time = current_time
    task_accuracies = [100. * c / total for c in correct]
    return 0, task_accuracies

def validate(model, val_loader, device, num_tasks):
    model.eval()
    total_loss = 0
    correct = torch.zeros(4).to(device, non_blocking=True)
    total = 0
    all_preds = [[] for _ in range(num_tasks)]
    all_targets = [[] for _ in range(num_tasks)]
    with torch.no_grad():
        for features, targets in val_loader:
            targets = targets.to(device, non_blocking=True)
            other_features = features['other_features'].to(device, non_blocking=True)
            feature_data = features['feature_data']
                
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(other_features, feature_data)
                print(f"Outputs sample: {outputs[:5]}")
                print(f"Targets sample: {targets[:5]}")
                loss = calculate_multi_task_loss(outputs, targets)
                for i in range(num_tasks):
                    all_preds[i].append(torch.sigmoid(outputs[:, i]))
                    all_targets[i].append(targets[:, i])
        
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            for task_idx in range(4):
                correct[task_idx] += (predictions[:, task_idx] == targets[:, task_idx]).sum().item()
            total += targets.size(0)
    
    task_accuracies = [100. * c / total for c in correct]
    return total_loss / len(val_loader), task_accuracies, all_preds, all_targets

def calculate_multi_task_loss(outputs, targets, pos_weights=None, weights=None):
    """
    计算多任务二分类损失
    
    Args:
        outputs: 模型输出 [batch_size, 4]
        targets: 目标值 [batch_size, 4]
        weights: 可选的任务权重 [4]
    """
    if weights is None:
        weights = [1.0] * 4
    
    total_loss = 0
    for task_idx in range(4):
        if pos_weights is not None:
            pos_weight = torch.tensor(pos_weights[task_idx], device=outputs.device)
        else:
            pos_weight = None
        task_loss = F.binary_cross_entropy_with_logits(
            outputs[:, task_idx],
            targets[:, task_idx].float(),
            pos_weight=pos_weight,
            reduction='mean'
        )
        total_loss += weights[task_idx] * task_loss
    
    return total_loss

def custom_collate_fn(batch):
    other_features_batch = []
    feature_data_dict_list_batch = []
    target_batch = []
    for sample in batch:
        # 拆分 batch 中的每个样本数据
        other_features_batch.append(sample[0]['other_features'])
        feature_data_dict_list_batch.append(sample[0]['feature_data'])
        target_batch.append(sample[1])

    other_features_batch = torch.stack(other_features_batch, dim=1)  # [time_step, batch_size, other_feature_dim]

    time_step_count = len(feature_data_dict_list_batch[0])

    feature_data_dict_list_combined = []

    for t in range(time_step_count):
        combined_dict = {} # dict[feature_name, (dict[batch_id, params], dict[batch_id, categories])]
        for batch_idx, batch in enumerate(feature_data_dict_list_batch):
            for feature_name, data in batch[t].items():
                if feature_name not in combined_dict:
                    combined_dict[feature_name] = ({}, {})
                combined_dict[feature_name][0][batch_idx] = data[0]
                if len(data) > 1:
                    combined_dict[feature_name][1][batch_idx] = data[1]
        feature_data_dict_list_combined.append(combined_dict)
    target_batch = torch.stack([target.clone().detach() for target in target_batch])
    
    return ({'other_features': other_features_batch, 'feature_data': feature_data_dict_list_combined}, target_batch)

def get_file_list():
    import os
    all_files = set()
    for filename in os.listdir('{FILE_PREFIX}/'):
        if filename.endswith('.parquet'):
            # Split on _ and take first part as base filename
            base_name = filename.split('_')[0]
            all_files.add(base_name)
    
    # Convert to sorted list for deterministic splitting
    all_files = sorted(list(all_files))
    
    # Calculate split point at 80%
    split_idx = int(len(all_files) * 0.8)
    
    # Split into train and validation files
    train_file_list = all_files[:split_idx]
    val_file_list = all_files[split_idx:]
    return train_file_list, val_file_list

FILE_PREFIX = "./model_feature/"

def main():
    config = {
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'epochs': 40,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': Path('./checkpoints'),
        'early_stopping_patience': 10,
    }
    config['save_dir'].mkdir(exist_ok=True)
    scaler = torch.amp.GradScaler()
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    wandb.init(project="trading_model", config=config)

    all_files = []
    for file in os.listdir(FILE_PREFIX):
        if file.endswith('.parquet'):
            base_name = file.split('_')[0]
            all_files.append(base_name)
    all_files = sorted(list(set(all_files)))
    train_file_list = all_files[:int(len(all_files) * 0.8)]
    val_file_list = all_files[int(len(all_files) * 0.8):]

    # train_file_list = all_files
    # val_file_list = []

    # file_cache = {}
    # for file in os.listdir(FILE_PREFIX):
    #     if file.endswith('.parquet'):
    #         file_cache[file] = pd.read_parquet(f"{FILE_PREFIX}/{file}")

    checkpoint_path = config['save_dir'] / 'latest_checkpoint.pth'
    best_model_path = config['save_dir'] / 'best_model.pth'
    train_dataset = TradingDataset(train_file_list, seq_len, begin_skip, end_skip)
    val_dataset = TradingDataset(val_file_list, seq_len, begin_skip, end_skip)
    logging.info(f"Train dataset: {train_file_list}")
    logging.info(f"Val dataset: {val_file_list}")
    logging.info(f"Train Length: {len(train_dataset)}")
    logging.info(f"Val Length: {len(val_dataset)}")

    dim = train_dataset.get_number_feature_dim()
    feature_config = get_feature_embedding_config(train_dataset.get_feature_name_list())

    model = MainModel(
        other_feature_dim=dim,
        feature_configs=feature_config,
        total_time_steps=seq_len,
        output_dim=4
    ).to(config['device'])

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['early_stopping_patience'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        logging.info(f"Resuming from epoch {start_epoch}")
        if checkpoint['batch_idx'] == -1:
            start_epoch += 1
        start_batch = checkpoint['batch_idx'] + 1
        logging.info(f"Resuming from epoch {start_epoch}, batch {start_batch}")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['early_stopping_patience'])
        best_val_loss = float('inf')
        patience_counter = 0
        start_epoch = 0
        start_batch = 0

    # start_batch = 630

    train_loader, sampler = create_data_loader(train_dataset, config['batch_size'], start_batch)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)

    num_tasks = 4
    auc_metrics = [AUROC(task="binary").to(config['device']) for _ in range(num_tasks)]
    f1_metrics = [F1Score(task="binary").to(config['device']) for _ in range(num_tasks)]

    for epoch in range(start_epoch, config['epochs']):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, config['device'], scaler, 
                                            epoch, checkpoint_path, best_val_loss, 
                                            patience_counter, scheduler, start_batch, config['batch_size'])
        sampler.reset()
        start_batch = 0
        val_loss, val_acc, all_preds, all_targets = validate(model, val_loader, config['device'], num_tasks)
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        avg_val_auc = 0.0
        avg_val_f1 = 0.0
        for i in range(num_tasks):
            task_preds = torch.cat(all_preds[i])
            task_targets = torch.cat(all_targets[i])
            task_auc = auc_metrics[i](task_preds, task_targets)
            task_f1 = f1_metrics[i](task_preds, task_targets)
            avg_val_auc += task_auc
            avg_val_f1 += task_f1
        avg_val_auc /= num_tasks
        avg_val_f1 /= num_tasks

        val_loss /= len(val_loader)

        logging.info(f'Epoch: {epoch+1}/{config["epochs"]}')
        # logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc[0]:.2f}%, {train_acc[1]:.2f}%, {train_acc[2]:.2f}%, {train_acc[3]:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc[0]:.2f}%, {val_acc[1]:.2f}%, {val_acc[2]:.2f}%, {val_acc[3]:.2f}% val_auc: {avg_val_auc}, val_f1: {avg_val_f1}')
        logging.info(f'Time: {epoch_time:.2f}s')
    
        # wandb.log({
        #     'train_loss': train_loss,
        #     'train_acc': train_acc,
        #     'val_loss': val_loss,
        #     'val_acc': val_acc,
        #     'val_auc': avg_val_auc,
        #     'val_f1': avg_val_f1,
        #     'learning_rate': optimizer.param_groups[0]['lr']
        # })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'patience_counter': patience_counter,
                'best_val_loss': best_val_loss,
                'batch_idx': -1,
                'total_batches': len(train_loader),
            }, best_model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'patience_counter': patience_counter,
                'best_val_loss': best_val_loss,
                'batch_idx': -1,
                'total_batches': len(train_loader),
            }, config['save_dir'] / f'checkpoint_epoch_{epoch+1}.pth')
        
        if patience_counter >= config['early_stopping_patience']:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    wandb.finish()
    if checkpoint_path.exists():
        os.remove(checkpoint_path)
        logging.info("Removed checkpoint after successful training completion")

if __name__ == '__main__':
    main()
