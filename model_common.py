import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状：[1, max_len, embedding_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class FeatureEmbedding(nn.Module):
    def __init__(self, param_dim, embedding_dim, category_dim, max_feature_length, num_categories, use_attention=True):
        """
        初始化特征嵌入层。

        参数：
        - param_dim (int): 每个特征参数的维度。
        - embedding_dim (int): 嵌入向量的维度。
        - max_feature_length (int): 单个时间步内最多的特征数量（用于填充）。
        - num_categories (int): 特征类别的总数（类别索引从 0 开始）。
          如果 num_categories == 1，则不使用类别嵌入。
        - use_attention (bool): 是否使用注意力机制。
        """
        super(FeatureEmbedding, self).__init__()
        self.param_dim = param_dim
        self.embedding_dim = embedding_dim
        self.category_dim = category_dim
        self.max_feature_length = max_feature_length
        self.num_categories = num_categories
        self.use_attention = use_attention

        if num_categories > 1:
            self.use_category_embedding = True
            self.category_embedding = nn.Embedding(num_categories, category_dim)
        else:
            self.use_category_embedding = False
            category_dim = 0

        self.param_embedding = nn.Linear(param_dim, embedding_dim)
        self.combined_embedding_dim = embedding_dim + category_dim
        # 位置编码
        self.positional_encoding = PositionalEncoding(self.combined_embedding_dim, max_len=max_feature_length)

        if self.use_attention:
            # 使用 TransformerEncoder 作为注意力机制
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.combined_embedding_dim, nhead=max(2, self.combined_embedding_dim // 16), batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            # 如果不使用注意力机制，可以在这里添加其他处理方式
            pass

    def forward(self, batch_feature_params, batch_feature_categories=None):
        """
        Args:
            batch_feature_params: list[list[list[float]]], shape: [batch_size, num_features, param_dim]
            batch_feature_categories: list[list[int]], shape: [batch_size, num_features] (optional)
        """
        device = next(self.parameters()).device
        batch_size = len(batch_feature_params)
        
        # 找到批次中最大的特征数量
        max_num_features = self.max_feature_length
        
        # 将所有特征参数和类别索引展开为一维列表
        all_params = []
        all_categories = []
        all_masks = []
        for features in batch_feature_params:
            num_features = len(features)
            all_params.extend(features)
            all_masks.extend([1] * num_features)
        if batch_feature_categories is not None:
            for categories in batch_feature_categories:
                all_categories.extend(categories)
        
        total_num_features = len(all_params)
        param_dim = self.param_dim
        
        # 将所有特征参数一次性转换为张量
        all_params_tensor = torch.tensor(
            np.array(all_params), 
            dtype=torch.float32, 
            device=device
        )  # [total_num_features, param_dim]
        
        # 进行参数嵌入
        param_embeds = self.param_embedding(all_params_tensor) if len(all_params) > 0 else torch.zeros(0, self.embedding_dim, device=device)  # [total_num_features, embedding_dim]
        
        if self.use_category_embedding and batch_feature_categories is not None:
            if len(all_categories) > 0:
                all_categories_tensor = torch.tensor(
                    all_categories,
                    dtype=torch.long,
                    device=device
                )  # [total_num_features]
            else:
                all_categories_tensor = torch.zeros(0, dtype=torch.long, device=device)
            category_embeds = self.category_embedding(all_categories_tensor)  # [total_num_features, category_dim]
            all_embeds = torch.cat([param_embeds, category_embeds], dim=1)  # [total_num_features, combined_embedding_dim]
        else:
            all_embeds = param_embeds  # [total_num_features, combined_embedding_dim]
        
        # 将嵌入向量重新组织回批次和特征维度

        all_embeds = all_embeds.to('cpu')
        embedded_features = torch.zeros(batch_size, max_num_features, self.combined_embedding_dim, device='cpu')
        masks = torch.zeros(batch_size, max_num_features, dtype=torch.bool, device='cpu')

        index = 0
        for b in range(batch_size):
            num_features = len(batch_feature_params[b])
            padding_length = min(max_num_features, num_features)
            if num_features > 0:
                embedded_features[b, :padding_length] = all_embeds[index:index+padding_length]
                masks[b, :padding_length] = True
                index += num_features
        
        embedded_features = embedded_features.to(device)
        masks = masks.to(device)
        embedded_features = self.positional_encoding(embedded_features)
        if self.use_attention:
                # 批量处理注意力机制
            with torch.autocast("cuda", enabled=False):
                transformer_output = self.transformer_encoder(
                    embedded_features,
                    src_key_padding_mask=masks
                )  # [batch_size, max_feature_length, combined_embedding_dim]
            # 批量计算注意力权重
            feature_weights = F.softmax(
                transformer_output.mean(dim=-1),
                dim=-1
            ) * (~masks).float()  # [batch_size, max_feature_length]
            # 批量加权聚合
            with torch.autocast("cuda", enabled=False):
                aggregated_features = torch.bmm(
                    feature_weights.unsqueeze(1),
                    transformer_output
                ).squeeze(1)  # [batch_size, combined_embedding_dim]
        else:
            # 简单的批量平均
            masked_features = embedded_features * (~masks).float().unsqueeze(-1)
            aggregated_features = masked_features.sum(dim=1) / \
                (~masks).float().sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        return aggregated_features  # [batch_size, combined_embedding_dim]


class MainModel(nn.Module):
    def __init__(self, other_feature_dim, feature_configs, total_time_steps, output_dim):
        """
        参数：
        - other_feature_dim (int): 其他特征的维度（每个时间步）。
        - feature_configs (dict): 特征配置的字典，键为特征名称，值为配置字典。
          配置字典包含：
          - 'param_dim': 特征参数维度。
          - 'embedding_dim': 嵌入维度。
          - 'max_feature_length': 最大特征长度。
          - 'category_dim': 类别维度。
          - 'num_categories': 类别数量。
          - 'use_attention': 是否使用注意力机制（bool）
        - total_time_steps (int): 总的时间步数，即时间序列的长度。
        - output_dim (int): 模型的输出维度（预测目标的维度）。
        """
        super(MainModel, self).__init__()
        self.feature_embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for feature_name, config in feature_configs.items():
            self.feature_embeddings[feature_name] = FeatureEmbedding(
                param_dim=config['param_dim'],
                embedding_dim=config['embedding_dim'],
                category_dim=config['category_dim'],
                max_feature_length=config['max_feature_length'],
                num_categories=config['num_categories'],
                use_attention=config.get('use_attention', True)
            )
            total_embedding_dim += config['embedding_dim']
            total_embedding_dim += config['category_dim']

        # 每个时间步的特征维度
        self.time_step_feature_dim = other_feature_dim + total_embedding_dim

        nhead = 32
        if self.time_step_feature_dim % nhead != 0:
            self.pad_dim = nhead - (self.time_step_feature_dim % nhead)
        else:
            self.pad_dim = 0
        self.padded_feature_dim = self.time_step_feature_dim + self.pad_dim
        # 时间序列的 Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.padded_feature_dim, nhead=16, batch_first=True)
        self.time_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.positional_encoding = PositionalEncoding(self.padded_feature_dim, max_len=total_time_steps)

        # 全连接层，用于输出预测结果
        self.fc_layers = nn.Sequential(
            nn.Linear(self.time_step_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, other_features_list, feature_data_dict_list):
        """
        参数：
        - other_features_list: 其他特征列表，长度为 time_steps，每个元素形状为 [batch_size, other_feature_dim]
        - feature_data_dict_list: 特征数据的字典列表，长度为 time_steps，每个元素是一个特征数据字典
          特征数据字典的键为特征名称，值为 (feature_params_batch, feature_categories_batch)
            - feature_params_batch: 批次中每个样本的特征参数列表
            - feature_categories_batch: 批次中每个样本的特征类别索引列表（可选）

        返回：
        - output: 模型输出，形状为 [batch_size, output_dim]
        """
        batch_size = other_features_list[0].size(0)
        time_steps = len(other_features_list)

        # 存储每个时间步的特征表示
        time_step_embeddings = []

        for t in range(time_steps):
            other_features = other_features_list[t]  # [batch_size, other_feature_dim]
            feature_data_dict = feature_data_dict_list[t]

            embedded_feature_list = []
            for feature_name, feature_embedding in self.feature_embeddings.items():
                feature_params_batch, feature_categories_batch = feature_data_dict.get(feature_name, ({}, {}))
                # 准备批量数据
                batch_params = []
                batch_categories = []
                for i in range(batch_size):
                    batch_params.append(feature_params_batch.get(i, []))
                    if feature_categories_batch:
                        batch_categories.append(feature_categories_batch.get(i, []))
                
                # 批量处理特征嵌入
                if feature_categories_batch:
                    aggregated_features = feature_embedding(batch_params, batch_categories)
                else:
                    aggregated_features = feature_embedding(batch_params)
                embedded_feature_list.append(aggregated_features)

            # 拼接其他特征和嵌入特征，得到当前时间步的表示
            time_step_input = torch.cat([other_features] + embedded_feature_list, dim=1)  # [batch_size, time_step_feature_dim]
            if self.pad_dim > 0:
                time_step_input = F.pad(time_step_input, (0, self.pad_dim))  # 填充到 [batch_size, time_steps, padded_feature_dim]

            time_step_embeddings.append(time_step_input.unsqueeze(1))  # [batch_size, 1, padded_feature_dim]
            # 将所有时间步的表示拼接成序列
            sequence_input = torch.cat(time_step_embeddings, dim=1)  # [batch_size, time_steps, padded_feature_dim]

            # 添加位置编码
            sequence_input = self.positional_encoding(sequence_input)  # [batch_size, time_steps, padded_feature_dim]
            # 转换形状以适应 Transformer：[seq_len, batch_size, feature_dim]
            # sequence_input = sequence_input.transpose(0, 1)  # [time_steps, batch_size, feature_dim]

            # 定义序列掩码（可选，如果有填充的时间步）
            # 此处假设所有样本的时间步长度相同，不需要序列掩码
            # 通过时间序列 Transformer
            with torch.autocast("cuda", enabled=False):
                transformer_output = self.time_transformer(sequence_input)  # [batch_size, time_steps, feature_dim]
            # 如果进行了填充，则去掉填充部分
            if self.pad_dim > 0:
                transformer_output = transformer_output[:, :, :self.time_step_feature_dim]  # 去除填充，形状为 [batch_size, time_steps, time_step_feature_dim]


            # 取最后一个时间步的输出作为特征（也可以采用其他聚合方式）
            final_output = transformer_output[:, -1, :]  # [batch_size, feature_dim]
            # 通过全连接层，输出预测结果
            with torch.autocast("cuda", enabled=False):
                output = self.fc_layers(final_output)  # [batch_size, output_dim]
            return output
