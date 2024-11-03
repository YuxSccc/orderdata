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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

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
            number_df = pd.read_parquet(f"./model_feature/{file}_number.parquet")
            valid_rows = len(number_df) - begin_skip - end_skip
            
            if valid_rows >= seq_len:
                num_sequences = valid_rows - seq_len + 1
                self.file_sequences.append((file, total_sequences, num_sequences))
                total_sequences += num_sequences
        
        self.total_sequences = total_sequences
        self.cache = {}
        self.feature_names = set()

        for file in file_list:
            event_df = self.read_parquet(f"./model_feature/{file}_event.parquet")
            self.feature_names.update(set(col.rsplit('_', 1)[0] for col in event_df.columns))
    
    def get_number_feature_dim(self):
        number_df = pd.read_parquet(f"./model_feature/{self.file_list[0]}_number.parquet")
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
        number_df = self.read_parquet(f"./model_feature/{file_path}_number.parquet")
        other_features = number_df.iloc[start_row:end_row].values
        
        # 2. 读取并处理 event features
        event_df = self.read_parquet(f"./model_feature/{file_path}_event.parquet")
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
        flow_df = self.read_parquet(f"./model_feature/{file_path}_flow.parquet")
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
        target_df = self.read_parquet(f"./model_feature/{file_path}_target.parquet")
        target = target_df.iloc[local_idx + self.seq_len - 1].values

        # for time_step in range(self.seq_len):
        #     for feature_name in self.feature_names:
        #         if feature_name not in feature_data[time_step]:
        #             feature_data[time_step][feature_name] = ([],)

        return {
            'other_features': torch.FloatTensor(other_features),
            'feature_data': feature_data
        }, torch.LongTensor(target)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = torch.zeros(4).to(device)  # 每个任务的正确预测数
    total = 0
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        # 将数据移到GPU
        targets = targets.to(device)
        other_features = features['other_features'].to(device)
        feature_data = features['feature_data']
            
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(other_features, feature_data)
            for name, param in model.named_parameters():
                print(f"Parameter {name} has dtype: {param.dtype}")
            loss = calculate_multi_task_loss(outputs, targets)


        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # 计算每个任务的准确率
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        for task_idx in range(4):
            correct[task_idx] += (predictions[:, task_idx] == targets[:, task_idx]).sum().item()
        total += targets.size(0)
        
        if batch_idx % 100 == 0:
            acc_str = ' '.join([f'Task{i+1}: {100.*c/total:.1f}%' for i, c in enumerate(correct)])
            logging.info(f'Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc_str}')
    
    # 返回平均损失和每个任务的准确率
    task_accuracies = [100. * c / total for c in correct]
    return total_loss / len(train_loader), task_accuracies

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = torch.zeros(4).to(device)
    total = 0
    
    with torch.no_grad():
        for features, targets in val_loader:
            targets = targets.to(device)
            other_features = features['other_features'].to(device)
            feature_data = features['feature_data']
                
            outputs = model(other_features, feature_data)
            loss = calculate_multi_task_loss(outputs, targets)
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            for task_idx in range(4):
                correct[task_idx] += (predictions[:, task_idx] == targets[:, task_idx]).sum().item()
            total += targets.size(0)
    
    task_accuracies = [100. * c / total for c in correct]
    return total_loss / len(val_loader), task_accuracies

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

def main():
    # 配置参数
    config = {
        'batch_size': 2,
        'learning_rate': 0.001,
        'epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': Path('./checkpoints'),
        'early_stopping_patience': 10,
        # 添加其他配置参数...
    }
    
    # 创建保存目录
    config['save_dir'].mkdir(exist_ok=True)
    
    # 初始化wandb（可选）
    wandb.init(project="trading_model", config=config)
    
    begin_skip = 30
    end_skip = 10
    seq_len = 100
    train_file_list = ['output']
    val_file_list = ['output']
    # 创建数据加载器
    train_dataset = TradingDataset(train_file_list, seq_len, begin_skip, end_skip)
    val_dataset = TradingDataset(val_file_list, seq_len, begin_skip, end_skip)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=custom_collate_fn)

    # 初始化模型
    model = MainModel(
        other_feature_dim=train_dataset.get_number_feature_dim(),
        feature_configs=get_feature_embedding_config(train_dataset.get_feature_name_list()),
        total_time_steps=seq_len,
        output_dim=4  # 根据目标类别数设置
    ).to(config['device'])


    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, config['device'])
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录训练信息
        epoch_time = time.time() - start_time
        logging.info(f'Epoch: {epoch+1}/{config["epochs"]}')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logging.info(f'Time: {epoch_time:.2f}s')
        
        # 记录到wandb（可选）
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
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
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, config['save_dir'] / 'best_model.pth')
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, config['save_dir'] / f'checkpoint_epoch_{epoch+1}.pth')
        
        # 早停
        if patience_counter >= config['early_stopping_patience']:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    wandb.finish()  # 如果使用wandb

if __name__ == '__main__':
    main()