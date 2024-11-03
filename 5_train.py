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
from sklearn.model_selection import train_test_split
import wandb  # 可选，用于实验跟踪
from config import *

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
    def __init__(self, features_dict, targets):
        """
        Args:
            features_dict: {
                'other_features': list[time_steps][batch_size, other_feature_dim],
                'feature_data': list[time_steps][dict] 每个dict包含各种特征的参数和类别
            }
            targets: [num_samples, num_classes] 或 [num_samples] (如果是单类别标签)
        """
        self.features_dict = features_dict
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # 对于每个时间步，获取该样本的特征
        other_features = [step_features[idx] for step_features in self.features_dict['other_features']]
        
        # 对于每个时间步，获取该样本的特征数据字典
        feature_data = []
        for step_data in self.features_dict['feature_data']:
            step_dict = {}
            for feature_name, (params, categories) in step_data.items():
                # 获取该样本的参数和类别
                sample_params = params[idx] if params else []
                sample_categories = categories[idx] if categories else None
                step_dict[feature_name] = (sample_params, sample_categories)
            feature_data.append(step_dict)
        
        return {
            'other_features': other_features,  # list[time_steps][other_feature_dim]
            'feature_data': feature_data       # list[time_steps][dict]
        }, self.targets[idx]

def load_and_preprocess_data(config):
    """
    从parquet文件加载数据并预处理

    Returns:
        processed_features: {
            'other_features': list[time_steps][num_samples, other_feature_dim],
            'feature_data': list[time_steps][dict] 每个dict的格式为:
                {
                    'feature_name': (
                        feature_params: list[num_samples][num_features, param_dim],
                        feature_categories: list[num_samples][num_features] or None
                    )
                }
        }
        targets: [num_samples, num_classes] 或 [num_samples]
        feature_configs: {
            'feature_name': {
                'param_dim': int,
                'embedding_dim': int,
                'category_dim': int,
                'max_feature_length': int,
                'num_categories': int,
                'use_attention': bool
            }
        }
    """
    # 加载各个parquet文件
    bar_df = pd.read_parquet('./model_feature/output_bar.parquet')
    bar_feature_df = pd.read_parquet('./model_feature/output_bar_feature.parquet')
    onehot_df = pd.read_parquet('./model_feature/output_onehot.parquet')
    price_level_df = pd.read_parquet('./model_feature/output_price_level.parquet')
    target_df = pd.read_parquet('./model_feature/output_target.parquet')
    flow_df = pd.read_parquet('./model_feature/output_flow.parquet')

    # TODO: 实现数据处理逻辑
    
    return processed_features, targets, feature_configs

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        # 将数据移到GPU
        targets = targets.to(device)
        other_features = [feat.to(device) for feat in features['other_features']]
        
        # feature_data中的数据不需要移到GPU，因为在模型内部会处理
        feature_data = features['feature_data']
            
        optimizer.zero_grad()
        outputs = model(other_features, feature_data)  # 修正：传入两个参数
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            logging.info(f'Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, targets in val_loader:
            targets = targets.to(device)
            for k in features:
                features[k] = features[k].to(device)
                
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(val_loader), 100.*correct/total

def main():
    # 配置参数
    config = {
        'batch_size': 32,
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
    
    # 加载数据
    processed_features, targets, feature_configs = load_and_preprocess_data(config)
    
    # 分割训练集和验证集
    train_features, val_features, train_targets, val_targets = train_test_split(
        processed_features, targets, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = TradingDataset(train_features, train_targets)
    val_dataset = TradingDataset(val_features, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # 初始化模型
    model = MainModel(
        other_feature_dim=processed_features['other_features'].shape[1],
        feature_configs=feature_configs,
        total_time_steps=config['time_steps'],
        output_dim=4  # 根据目标类别数设置
    ).to(config['device'])
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
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