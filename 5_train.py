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
import wandb  # 可选，用于实验跟踪
from config import *
import torch.nn.functional as F
import os
from torchmetrics import AUROC, F1Score
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

FILE_PREFIX = "./model_feature_test"

class TradingDataset(Dataset):
    def __init__(self, file_list: list[str], seq_len: int, begin_skip: int = 100, end_skip: int = 100):
        """
        Args:
            file_list: 数据文件列表，每个文件会对应读取 _number/_event/_target/_flow 四个parquet文件
            seq_len: 每组数据包含的时间步数
            begin_skip: 开始时跳过的行数
            end_skip: 结束时跳过的行数
        """
        self.seq_len = seq_len
        self.file_list = file_list
        self.begin_skip = begin_skip
        self.end_skip = end_skip
        
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
        self.cache = {}
        self.feature_names = set()

        for file in file_list:
            event_df = self.read_parquet(f"{FILE_PREFIX}/{file}_event.parquet")
            self.feature_names.update(set(col.rsplit('_', 1)[0] for col in event_df.columns))
    
    def get_number_feature_dim(self):
        number_df = pd.read_parquet(f"{FILE_PREFIX}/{self.file_list[0]}_number.parquet")
        return number_df.shape[1]

    def get_feature_name_list(self):
        return list(self.feature_names)

    def read_parquet(self, file_path: str):
        if file_path in self.cache:
            return self.cache[file_path]
        df = pd.read_parquet(file_path)
        self.cache[file_path] = df
        return df

    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # 找到对应的文件和局部索引
        file_path = None
        local_idx = None
        for file, start_idx, num_sequences in self.file_sequences:
            if idx < start_idx + num_sequences:
                file_path = file
                local_idx = idx - start_idx
                break
        
        if file_path is None:
            raise IndexError("Index out of range")
        
        # 计算实际的数据范围
        start_row = self.begin_skip + local_idx
        end_row = start_row + self.seq_len
        
        # 1. 读取并处理 number features
        number_df = self.read_parquet(f"{FILE_PREFIX}/{file_path}_number.parquet")
        other_features = number_df.iloc[start_row:end_row].values
        
        # 2. 读取并处理 event features
        event_df = self.read_parquet(f"{FILE_PREFIX}/{file_path}_event.parquet")
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
                    # 确保params是numpy数组并具有正确的形状
                    # params = np.array(params).reshape(-1, len(params[0]) if isinstance(params[0], list) else 1)
                    # categories = np.array(categories) if categories is not None else None
                    if categories is None:
                        step_features[feature_name] = (params.tolist(),)
                    else:
                        step_features[feature_name] = (params.tolist(), categories)
            
            feature_data.append(step_features)
        
        # 3. 读取并处理 flow features
        flow_df = self.read_parquet(f"{FILE_PREFIX}/{file_path}_flow.parquet")
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
                        # 如果特征已存在，扩展参数列表
                        params, categories = feature_data[step][feature_name]
                        params = np.concatenate([params, value], axis=1)
                        if categories is None:
                            feature_data[step][feature_name] = (params.tolist(),)
                        else:
                            feature_data[step][feature_name] = (params.tolist(), categories)
                    else:
                        # 如果特征不存在，创建新特征
                        feature_data[step][feature_name] = (value,)
        
        # 4. 读取target（使用序列末尾对应的目标）
        target_df = self.read_parquet(f"{FILE_PREFIX}/{file_path}_target.parquet")
        target = target_df.iloc[local_idx + self.seq_len - 1].values

        # for time_step in range(self.seq_len):
        #     for feature_name in self.feature_names:
        #         if feature_name not in feature_data[time_step]:
        #             feature_data[time_step][feature_name] = ([],)

        return {
            'other_features': torch.FloatTensor(other_features),
            'feature_data': feature_data
        }, torch.LongTensor(target)

def print_model_params(model, prefix=""):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{prefix}Parameter {name}: {param.shape}")

def get_state_hash(model, optimizer):
    """计算模型参数和优化器状态的哈希值"""
    import hashlib
    import json
    
    def tensor_to_list(tensor):
        """将tensor转换为可哈希的形式"""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy().tolist()
        return tensor

    def get_state_dict(obj):
        """获取状态字典的可哈希形式"""
        if isinstance(obj, dict):
            return {str(k): get_state_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [get_state_dict(x) for x in obj]
        elif torch.is_tensor(obj):
            return tensor_to_list(obj)
        return obj

    # 获取模型状态
    model_state = get_state_dict(model.state_dict())
    
    # 获取优化器状态
    optimizer_state = {
        'param_groups': get_state_dict(optimizer.state_dict()['param_groups']),
        'state': get_state_dict(optimizer.state_dict()['state'])
    }
    
    # 计算哈希值
    model_hash = hashlib.sha256(json.dumps(model_state, sort_keys=True).encode()).hexdigest()
    optimizer_hash = hashlib.sha256(json.dumps(optimizer_state, sort_keys=True).encode()).hexdigest()
    
    return {
        'model': model_hash,
        'optimizer': optimizer_hash
    }

# 在关键点检查状态
def check_state_consistency(model, optimizer, checkpoint_path, train_dataset):
    print("\nBefore saving checkpoint:")
    before_save = get_state_hash(model, optimizer)
    print(f"Model hash: {before_save['model'][:10]}...")
    print(f"Optimizer hash: {before_save['optimizer'][:10]}...")
    
    # 保存checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # ... 其他状态 ...
    }, checkpoint_path)
    
    print("\nAfter saving, before loading:")
    after_save = get_state_hash(model, optimizer)
    print(f"Model hash: {after_save['model'][:10]}...")
    print(f"Optimizer hash: {after_save['optimizer'][:10]}...")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)

    model = MainModel(
        other_feature_dim=train_dataset.get_number_feature_dim(),
        feature_configs=get_feature_embedding_config(train_dataset.get_feature_name_list()),
        total_time_steps=seq_len,
        output_dim=4
    ).to('cuda')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("\nAfter loading checkpoint:")
    after_load = get_state_hash(model, optimizer)
    print(f"Model hash: {after_load['model'][:10]}...")
    print(f"Optimizer hash: {after_load['optimizer'][:10]}...")
    
    # 检查一致性
    is_model_consistent = (before_save['model'] == after_load['model'])
    is_optimizer_consistent = (before_save['optimizer'] == after_load['optimizer'])
    
    print("\nConsistency check:")
    print(f"Model state consistent: {is_model_consistent}")
    print(f"Optimizer state consistent: {is_optimizer_consistent}")
    
    return is_model_consistent and is_optimizer_consistent, model, optimizer

def train_epoch(model, train_loader, optimizer, device, scaler, 
                epoch, checkpoint_path, best_val_loss, patience_counter, scheduler, start_batch, gen, train_dataset):
    model.train()
    total_loss = 0
    correct = torch.zeros(4).to(device)  # 每个任务的正确预测数
    total = 0
    
    last_print_time = time.time()
    last_checkpoint_time = time.time()
    checkpoint_interval = 60  # 每300秒（5分钟）保存一次检查点

    for batch_idx, (features, targets) in enumerate(train_loader):
        if batch_idx < start_batch:
            continue
        # 将数据移到GPU

        print(f"\nBatch {batch_idx} data check:")
        print(f"Features type: {type(features)}")
        print(f"Other features shape: {features['other_features'].shape}")
        tensor_bytes = features['other_features'].numpy().tobytes()
        import hashlib
        hash_object = hashlib.sha256(tensor_bytes)
        hash_hex = hash_object.hexdigest()
        print(f"Other features hash: {hash_hex[:10]}...")

        targets = targets.to(device)
        other_features = features['other_features'].to(device)
        feature_data = features['feature_data']
        
        # with torch.amp.autocast('cuda'):
        outputs = model(other_features, feature_data)
        loss = calculate_multi_task_loss(outputs, targets)
        wandb.log({"train_loss": loss})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # 计算每个任务的准确率
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        for task_idx in range(4):
            correct[task_idx] += (predictions[:, task_idx] == targets[:, task_idx]).sum().item()
        total += targets.size(0)
        
        current_time = time.time()

        if batch_idx % 100 == 0 or current_time - last_print_time >= 30:
            acc_str = ' '.join([f'Task{i+1}: {100.*c/total:.1f}%' for i, c in enumerate(correct)])
            logging.info(f'Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc_str}')
            last_print_time = time.time()


        # 定期保存检查点
        if current_time - last_checkpoint_time >= checkpoint_interval:
            batch_progress = batch_idx / len(train_loader)  # 当前epoch的进度
            logging.info(f"Saving checkpoint at epoch {epoch}, batch progress: {batch_progress:.2%}")
            
            print("\nBefore save checkpoint:")
            before_save = get_state_hash(model, optimizer)
            is_consistent, model, optimizer = check_state_consistency(model, optimizer, './checkpoints/aaa.pth', train_dataset)

            torch.save({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'scaler_state_dict': scaler.state_dict(),
                'patience_counter': patience_counter,
                'best_val_loss': best_val_loss,
                'total_batches': len(train_loader),
                'gen_state': gen.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
            }, checkpoint_path)
            
            print("\nAfter save checkpoint:")
            after_save = get_state_hash(model, optimizer)


            last_checkpoint_time = current_time

    # 返回平均损失和每个任务的准确率
    task_accuracies = [100. * c / total for c in correct]
    return total_loss / len(train_loader), task_accuracies, model, optimizer

def validate(model, val_loader, criterion, device, scaler, num_tasks):
    model.eval()
    total_loss = 0
    correct = torch.zeros(4).to(device)
    total = 0
    all_preds = [[] for _ in range(num_tasks)]
    all_targets = [[] for _ in range(num_tasks)]
    with torch.no_grad():
        for features, targets in val_loader:
            targets = targets.to(device)
            other_features = features['other_features'].to(device)
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

def check_optimizer_state(optimizer, prefix=""):
    """检查优化器状态的一致性"""
    param_shapes = {}
    state_shapes = {}
    
    for i, group in enumerate(optimizer.param_groups):
        for j, p in enumerate(group['params']):
            param_shapes[f"param_{i}_{j}"] = p.shape
            if p in optimizer.state:
                state = optimizer.state[p]
                state_shapes[f"param_{i}_{j}"] = {
                    k: v.shape if torch.is_tensor(v) else v
                    for k, v in state.items()
                }
    return param_shapes, state_shapes

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # 配置参数
    config = {
        'batch_size': 2,
        'learning_rate': 0.001,
        'epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': Path('./checkpoints'),
        'early_stopping_patience': 10,
        # 添加其他配置参数...
    }
    seed = 42  # 固定的随机种子
    # 创建保存目录
    config['save_dir'].mkdir(exist_ok=True)
    scaler = torch.amp.GradScaler()
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 初始化wandb（可选）
    wandb.init(project="trading_model", config=config)
    gen = torch.Generator()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    begin_skip = 100
    end_skip = 50
    seq_len = 100
    # Get all unique file names from model_feature directory
    train_file_list = ['output']
    val_file_list = ['output']
    checkpoint_path = config['save_dir'] / 'latest_checkpoint.pth'
    best_model_path = config['save_dir'] / 'best_model.pth'

    # checkpoint_path = config['save_dir'] / 'best_model.pth'
    # 尝试加载最近的检查点
    train_dataset = TradingDataset(train_file_list, seq_len, begin_skip, end_skip)
    val_dataset = TradingDataset(val_file_list, seq_len, begin_skip, end_skip)
    model = MainModel(
        other_feature_dim=train_dataset.get_number_feature_dim(),
        feature_configs=get_feature_embedding_config(train_dataset.get_feature_name_list()),
        total_time_steps=seq_len,
        output_dim=4
    ).to(config['device'])

    # print_model_params(model, prefix="Model Parameters: ")


    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    if checkpoint_path.exists():
        before_save = get_state_hash(model, optimizer)

        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        gen.set_state(checkpoint['gen_state'])
        logging.info(f"Resuming from epoch {start_epoch}")
        if checkpoint['batch_idx'] == -1:
            start_epoch += 1
        start_batch = checkpoint['batch_idx'] + 1
        logging.info(f"Resuming from epoch {start_epoch}, batch {start_batch}")

        after_save = get_state_hash(model, optimizer)
        print("before load ", before_save)
        print("after load ", after_save)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        best_val_loss = float('inf')
        patience_counter = 0
        start_epoch = 0
        start_batch = 0
    # 创建数据加载器
    # train_file_list, val_file_list = get_file_list()
    
    print(start_epoch, start_batch)

    # print_model_params(model, prefix="Model Parameters: ")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn)


    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    num_tasks = 4
    auc_metrics = [AUROC(task="binary").to(config['device']) for _ in range(num_tasks)]
    f1_metrics = [F1Score(task="binary").to(config['device']) for _ in range(num_tasks)]
    # 训练循环
    
    for epoch in range(start_epoch, config['epochs']):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss, train_acc, model, optimizer = train_epoch(model, train_loader, optimizer, config['device'], scaler, 
                                            epoch, checkpoint_path, best_val_loss, 
                                            patience_counter, scheduler, start_batch, gen, train_dataset)
        
        # 验证
        val_loss, val_acc, all_preds, all_targets = validate(model, val_loader, criterion, config['device'], scaler, num_tasks)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录训练信息
        epoch_time = time.time() - start_time
        # 计算每个任务的 AUC 和 F1，然后取平均
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
    
        # 记录到wandb（可选）
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': avg_val_auc,
            'val_f1': avg_val_f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'scaler_state_dict': scaler.state_dict(),
                'patience_counter': patience_counter,
                'best_val_loss': best_val_loss,
                'batch_idx': -1,
                'total_batches': len(train_loader),
                'gen_state': gen.get_state(),
            }, config['save_dir'] / 'best_model.pth')
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'scaler_state_dict': scaler.state_dict(),
                'patience_counter': patience_counter,
                'best_val_loss': best_val_loss,
                'batch_idx': -1,
                'total_batches': len(train_loader),
                'gen_state': gen.get_state(),
            }, config['save_dir'] / f'checkpoint_epoch_{epoch+1}.pth')
        
        # 早停
        if patience_counter >= config['early_stopping_patience']:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    wandb.finish()  # 如果使用wandb
    if checkpoint_path.exists():
        os.remove(checkpoint_path)
        logging.info("Removed checkpoint after successful training completion")

if __name__ == '__main__':
    main()