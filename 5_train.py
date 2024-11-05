import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
import cProfile
import pstats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
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

        # 初始化特征名称（只需要读取一个文件）
        first_file = file_list[0]
        event_df = self.read_parquet(f"{first_file}_event.parquet")
        self.feature_names.update(set(col.rsplit('_', 1)[0] for col in event_df.columns))
    
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
        file_path = None
        local_idx = None
        for file, start_idx, num_sequences in self.file_sequences:
            if idx < start_idx + num_sequences:
                file_path = file
                local_idx = idx - start_idx
                break
        
        if file_path is None:
            raise IndexError("Index out of range")
        
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

        return {
            'other_features': torch.FloatTensor(other_features),
            'feature_data': feature_data
        }, torch.LongTensor(target)

def train_epoch(model, train_loader, optimizer, device, scaler, 
                epoch, checkpoint_path, best_val_loss, patience_counter, scheduler, start_batch):
    model.train()
    total_loss = 0
    correct = torch.zeros(4).to(device)
    total = 0
    
    last_print_time = time.time()
    last_checkpoint_time = time.time()
    checkpoint_interval = 300

    for batch_idx, (features, targets) in enumerate(train_loader):
        if batch_idx < start_batch:
            continue

        targets = targets.to(device)
        other_features = features['other_features'].to(device, non_blocking=True)
        feature_data = features['feature_data']
        
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(other_features, feature_data)
            loss = calculate_multi_task_loss(outputs, targets)
        wandb.log({"train_loss": loss})
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        for task_idx in range(4):
            correct[task_idx] += (predictions[:, task_idx] == targets[:, task_idx]).sum().item()
        total += targets.size(0)
        
        current_time = time.time()

        if batch_idx % 100 == 0 or current_time - last_print_time >= 120:
            acc_str = ' '.join([f'Task{i+1}: {100.*c/total:.1f}%' for i, c in enumerate(correct)])
            logging.info(f'Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc_str}')
            last_print_time = time.time()

        if current_time - last_checkpoint_time >= checkpoint_interval:
            batch_progress = batch_idx / len(train_loader)
            logging.info(f"Saving checkpoint at epoch {epoch}, batch progress: {batch_progress:.2%}")

            torch.save({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'patience_counter': patience_counter,
                'best_val_loss': best_val_loss,
                'total_batches': len(train_loader),
            }, checkpoint_path)
            
            logging.info(f"Saving checkpoint at epoch {epoch}, batch progress: {batch_progress:.2%}")
            last_checkpoint_time = current_time

    task_accuracies = [100. * c / total for c in correct]
    return total_loss / len(train_loader), task_accuracies

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

def calculate_multi_task_loss(outputs, targets, weights=None):
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
        task_loss = F.binary_cross_entropy_with_logits(
            outputs[:, task_idx],
            targets[:, task_idx].float(),
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
        'batch_size': 64,
        'learning_rate': 0.0015,
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

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
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

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, num_workers=4)

    num_tasks = 4
    auc_metrics = [AUROC(task="binary").to(config['device']) for _ in range(num_tasks)]
    f1_metrics = [F1Score(task="binary").to(config['device']) for _ in range(num_tasks)]

    for epoch in range(start_epoch, config['epochs']):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, config['device'], scaler, 
                                            epoch, checkpoint_path, best_val_loss, 
                                            patience_counter, scheduler, start_batch)
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
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc[0]:.2f}%, {train_acc[1]:.2f}%, {train_acc[2]:.2f}%, {train_acc[3]:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc[0]:.2f}%, {val_acc[1]:.2f}%, {val_acc[2]:.2f}%, {val_acc[3]:.2f}% val_auc: {avg_val_auc}, val_f1: {avg_val_f1}')
        logging.info(f'Time: {epoch_time:.2f}s')
    
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': avg_val_auc,
            'val_f1': avg_val_f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
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
    # cProfile.run('main()', 'output.prof')
    main()
    # p = pstats.Stats('output.prof')
    # p.sort_stats('cumtime').print_stats(100)
