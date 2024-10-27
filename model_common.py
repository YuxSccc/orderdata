import torch
import torch.nn as nn

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
    def __init__(self, param_dim, embedding_dim, max_feature_length, num_categories, use_attention=True):
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
        self.max_feature_length = max_feature_length
        self.num_categories = num_categories
        self.use_attention = use_attention

        if num_categories > 1:
            self.use_category_embedding = True
            self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        else:
            self.use_category_embedding = False

        self.param_embedding = nn.Linear(param_dim, embedding_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_feature_length)

        if self.use_attention:
            # 使用 TransformerEncoder 作为注意力机制
            encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            # 如果不使用注意力机制，可以在这里添加其他处理方式
            pass

    def forward(self, feature_params, feature_categories=None):
        """
        前向传播。

        参数：
        - feature_params (list of list of float): 特征参数列表，形状为 [num_features, param_dim]。
        - feature_categories (list of int, optional): 特征类别索引列表，形状为 [num_features]。
          如果 num_categories == 1，可以不提供该参数。

        返回：
        - aggregated_feature (torch.Tensor): 聚合后的特征向量，形状为 [1, embedding_dim]。
        """
        device = next(self.parameters()).device

        num_features = len(feature_params)

        if num_features == 0:
            # 如果没有特征，返回全零的向量
            aggregated_feature = torch.zeros(1, self.embedding_dim, device=device)
            return aggregated_feature

        # 将特征参数转换为张量
        feature_params_tensor = torch.tensor(feature_params, dtype=torch.float32, device=device)  # [num_features, param_dim]
        param_embeds = self.param_embedding(feature_params_tensor)  # [num_features, embedding_dim]

        if self.use_category_embedding:
            assert feature_categories is not None, "feature_categories must be provided when num_categories > 1"
            feature_categories_tensor = torch.tensor(feature_categories, dtype=torch.long, device=device)  # [num_features]
            category_embeds = self.category_embedding(feature_categories_tensor)  # [num_features, embedding_dim]
            embedded_features = category_embeds + param_embeds  # [num_features, embedding_dim]
        else:
            embedded_features = param_embeds  # [num_features, embedding_dim]

        # 创建掩码：填充位置为 True，有效数据为 False
        mask = torch.ones(self.max_feature_length, dtype=torch.bool, device=device)  # 全部初始化为填充
        mask[:num_features] = False  # 有效位置设为 False

        # 填充或截断 embedded_features 到 max_feature_length
        if num_features < self.max_feature_length:
            padding_length = self.max_feature_length - num_features
            padding_embeds = torch.zeros(padding_length, self.embedding_dim, device=device)
            embedded_features = torch.cat([embedded_features, padding_embeds], dim=0)  # [max_feature_length, embedding_dim]
        else:
            embedded_features = embedded_features[:self.max_feature_length, :]  # [max_feature_length, embedding_dim]
            mask = mask[:self.max_feature_length]

        # 添加批次维度
        embedded_features = embedded_features.unsqueeze(0)  # [1, max_feature_length, embedding_dim]
        mask = mask.unsqueeze(0)  # [1, max_feature_length]

        # 添加位置编码
        embedded_features = self.positional_encoding(embedded_features)  # [1, max_feature_length, embedding_dim]

        if self.use_attention:
            # 转换形状以适应 Transformer：[seq_len, batch_size, embedding_dim]
            embedded_features = embedded_features.transpose(0, 1)  # [max_feature_length, 1, embedding_dim]

            # Transformer 期望的 src_key_padding_mask 形状为 [batch_size, seq_len]
            # 我们需要传递 mask，填充位置为 True，有效位置为 False
            transformer_mask = mask  # [1, max_feature_length]

            # 通过 TransformerEncoder
            transformer_output = self.transformer_encoder(embedded_features, src_key_padding_mask=transformer_mask)  # [max_feature_length, 1, embedding_dim]

            # 转换回 [batch_size, seq_len, embedding_dim]
            transformer_output = transformer_output.transpose(0, 1)  # [1, max_feature_length, embedding_dim]

            # 应用掩码到 transformer_output
            masked_output = transformer_output * (~mask.unsqueeze(2)).float()  # 填充位置置零

            # 聚合有效位置的输出（例如，取平均值）
            sum_embeddings = masked_output.sum(dim=1)  # [1, embedding_dim]
            valid_token_count = (~mask).sum(dim=1, keepdim=True).float()  # [1, 1]
            aggregated_feature = sum_embeddings / (valid_token_count + 1e-8)  # [1, embedding_dim]
        else:
            # 如果不使用注意力机制，可以对嵌入特征进行其他聚合
            masked_embeddings = embedded_features * (~mask.unsqueeze(2)).float()
            sum_embeddings = masked_embeddings.sum(dim=1)  # [1, embedding_dim]
            valid_token_count = (~mask).sum(dim=1, keepdim=True).float()
            aggregated_feature = sum_embeddings / (valid_token_count + 1e-8)  # [1, embedding_dim]

        return aggregated_feature  # [1, embedding_dim]


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
                max_feature_length=config['max_feature_length'],
                num_categories=config['num_categories'],
                use_attention=config.get('use_attention', True)
            )
            total_embedding_dim += config['embedding_dim']

        # 每个时间步的特征维度
        self.time_step_feature_dim = other_feature_dim + total_embedding_dim

        # 时间序列的 Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.time_step_feature_dim, nhead=4)
        self.time_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.positional_encoding = PositionalEncoding(self.time_step_feature_dim, max_len=total_time_steps)

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
                feature_params_batch, feature_categories_batch = feature_data_dict.get(feature_name, ([], None))

                # 处理批次中的每个样本
                aggregated_features = []

                for i in range(batch_size):
                    feature_params = feature_params_batch[i]
                    feature_categories = feature_categories_batch[i] if feature_categories_batch is not None else None

                    aggregated_feature = feature_embedding(feature_params, feature_categories)  # [1, embedding_dim]
                    aggregated_features.append(aggregated_feature)

                # 堆叠聚合特征
                aggregated_features = torch.cat(aggregated_features, dim=0)  # [batch_size, embedding_dim]
                embedded_feature_list.append(aggregated_features)

            # 拼接其他特征和嵌入特征，得到当前时间步的表示
            time_step_input = torch.cat([other_features] + embedded_feature_list, dim=1)  # [batch_size, time_step_feature_dim]
            time_step_embeddings.append(time_step_input.unsqueeze(1))  # [batch_size, 1, time_step_feature_dim]

        # 将所有时间步的表示拼接成序列
        sequence_input = torch.cat(time_step_embeddings, dim=1)  # [batch_size, time_steps, time_step_feature_dim]

        # 添加位置编码
        sequence_input = self.positional_encoding(sequence_input)  # [batch_size, time_steps, time_step_feature_dim]

        # 转换形状以适应 Transformer：[seq_len, batch_size, feature_dim]
        sequence_input = sequence_input.transpose(0, 1)  # [time_steps, batch_size, feature_dim]

        # 定义序列掩码（可选，如果有填充的时间步）
        # 此处假设所有样本的时间步长度相同，不需要序列掩码

        # 通过时间序列 Transformer
        transformer_output = self.time_transformer(sequence_input)  # [time_steps, batch_size, feature_dim]

        # 取最后一个时间步的输出作为特征（也可以采用其他聚合方式）
        final_output = transformer_output[-1, :, :]  # [batch_size, feature_dim]

        # 通过全连接层，输出预测结果
        output = self.fc_layers(final_output)  # [batch_size, output_dim]

        return output
