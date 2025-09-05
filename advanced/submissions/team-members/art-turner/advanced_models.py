"""
Week 4: Advanced Neural Network Architectures for Optimization
Enhanced LSTM, GRU, and TCN models with various architectural improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM with proper handling of both directions
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [128, 64],
                 output_size: int = 3,
                 dropout_rate: float = 0.2):
        super(BidirectionalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_layers = len(hidden_sizes)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First bidirectional LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[0],
                batch_first=True,
                dropout=dropout_rate if len(hidden_sizes) > 1 else 0,
                bidirectional=True
            )
        )
        
        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            lstm_input_size = hidden_sizes[i-1] * 2  # *2 for bidirectional
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=hidden_sizes[i],
                    batch_first=True,
                    dropout=dropout_rate if i < len(hidden_sizes) - 1 else 0,
                    bidirectional=True
                )
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        final_hidden_size = hidden_sizes[-1] * 2  # *2 for bidirectional
        self.fc = nn.Linear(final_hidden_size, output_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                # Set forget gate bias to 1 for LSTM
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Pass through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:
                x = self.dropout(x)
        
        # Take the last time step output
        x = x[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Apply final dropout
        x = self.dropout(x)
        
        # Linear output layer
        output = self.fc(x)
        
        return output


class DeepLSTM(nn.Module):
    """
    Deep LSTM with 3-4 layers and advanced regularization
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64, 32],
                 output_size: int = 3,
                 dropout_rate: float = 0.3,
                 layer_norm: bool = True):
        super(DeepLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.num_layers = len(hidden_sizes)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if layer_norm else None
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[0],
                batch_first=True,
                dropout=dropout_rate if len(hidden_sizes) > 1 else 0
            )
        )
        
        if layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))
        
        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_sizes[i-1],
                    hidden_size=hidden_sizes[i],
                    batch_first=True,
                    dropout=dropout_rate if i < len(hidden_sizes) - 1 else 0
                )
            )
            
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_sizes[i]))
        
        # Dropout and batch norm
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer with residual connection
        self.fc1 = nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2)
        self.fc2 = nn.Linear(hidden_sizes[-1] // 2, output_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                if n >= 4:  # LSTM bias
                    param.data[n//4:n//2].fill_(1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Pass through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            
            # Apply layer normalization
            if self.layer_norm:
                x = self.layer_norms[i](x)
            
            # Apply dropout between layers
            if i < len(self.lstm_layers) - 1:
                x = self.dropout(x)
        
        # Take the last time step output
        x = x[:, -1, :]
        
        # MLP head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism to focus on important time steps
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 3,
                 dropout_rate: float = 0.2):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param.data)
                else:
                    nn.init.constant_(param.data, 0)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        """Forward pass"""
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # Self-attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out  # Residual connection
        
        # Take weighted average across time steps
        # Use last time step as query for final attention
        final_out = combined[:, -1, :]  # (batch, hidden)
        
        # Output layer
        output = self.fc(self.dropout(final_out))
        
        return output


class AdvancedTCN(nn.Module):
    """
    Advanced TCN with deeper dilations and attention
    """
    
    def __init__(self,
                 input_size: int,
                 num_channels: List[int] = [64, 128, 256, 128, 64],
                 output_size: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 use_attention: bool = True):
        super(AdvancedTCN, self).__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.output_size = output_size
        self.use_attention = use_attention
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            padding = (kernel_size - 1) * dilation_size
            
            layers += [AdvancedTemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            )]
        
        self.network = nn.Sequential(*layers)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=num_channels[-1],
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Output layers
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1] // 2)
        self.fc2 = nn.Linear(num_channels[-1] // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # TCN expects (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        # Pass through temporal blocks
        y = self.network(x)  # (batch_size, num_channels[-1], sequence_length)
        
        # Attention mechanism
        if self.use_attention:
            # Transpose for attention
            y_attn = y.transpose(1, 2)  # (batch, seq, channels)
            y_attn, _ = self.attention(y_attn, y_attn, y_attn)
            y = y_attn.transpose(1, 2)  # Back to (batch, channels, seq)
        
        # Global average pooling
        y = self.adaptive_pool(y)  # (batch_size, num_channels[-1], 1)
        y = y.squeeze(-1)  # (batch_size, num_channels[-1])
        
        # MLP head
        y = F.relu(self.fc1(self.dropout(y)))
        output = self.fc2(self.dropout(y))
        
        return output


class AdvancedTemporalBlock(nn.Module):
    """
    Enhanced temporal block with squeeze-and-excitation
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(AdvancedTemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Squeeze-and-Excitation block
        self.se_block = SqueezeExcitation(n_outputs)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize convolutional weights"""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Apply squeeze-and-excitation
        out = self.se_block(out)
        
        # Residual connection with sequence length matching
        res = x if self.downsample is None else self.downsample(x)
        
        # Ensure same sequence length
        if out.size(-1) != res.size(-1):
            min_length = min(out.size(-1), res.size(-1))
            out = out[:, :, :min_length]
            res = res[:, :, :min_length]
        
        return self.relu(out + res)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    """
    
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models with learnable weights
    """
    
    def __init__(self, models: List[nn.Module], output_size: int = 3):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.output_size = output_size
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
        # Optional meta-learner
        self.use_meta_learner = True
        if self.use_meta_learner:
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_models * output_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_size)
            )
    
    def forward(self, x):
        """Forward pass through ensemble"""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=1)  # (batch, num_models, output_size)
        
        if self.use_meta_learner:
            # Use meta-learner
            flattened = stacked_preds.view(stacked_preds.size(0), -1)  # (batch, num_models * output_size)
            output = self.meta_learner(flattened)
        else:
            # Weighted average
            weights = F.softmax(self.ensemble_weights, dim=0)
            output = torch.sum(stacked_preds * weights.view(1, -1, 1), dim=1)
        
        return output


def create_advanced_models(input_size: int = 11, output_size: int = 3) -> Dict[str, nn.Module]:
    """
    Create all advanced model architectures
    """
    models = {}
    
    # Advanced LSTM variants
    models['BiLSTM_Medium'] = BidirectionalLSTM(
        input_size=input_size, 
        hidden_sizes=[128, 64], 
        output_size=output_size
    )
    
    models['DeepLSTM_Large'] = DeepLSTM(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        output_size=output_size,
        layer_norm=True
    )
    
    models['AttentionLSTM'] = AttentionLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        output_size=output_size
    )
    
    # Advanced GRU (bidirectional)
    models['BiGRU_Medium'] = BidirectionalLSTM(  # Can reuse with GRU cells
        input_size=input_size,
        hidden_sizes=[128, 64],
        output_size=output_size
    )
    
    # Advanced TCN
    models['AdvancedTCN'] = AdvancedTCN(
        input_size=input_size,
        num_channels=[64, 128, 256, 128, 64],
        output_size=output_size,
        use_attention=True
    )
    
    models['DeepTCN'] = AdvancedTCN(
        input_size=input_size,
        num_channels=[32, 64, 128, 256, 256, 128, 64],
        output_size=output_size,
        use_attention=False
    )
    
    return models


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Advanced Neural Network Architectures")
    print("=" * 60)
    
    # Create all models
    models = create_advanced_models()
    
    # Test each model
    dummy_input = torch.randn(4, 36, 11)  # (batch, seq_len, features)
    
    for name, model in models.items():
        try:
            with torch.no_grad():
                output = model(dummy_input)
            
            params = count_parameters(model)
            print(f"{name:20s}: {params:,} parameters, output: {output.shape}")
            
        except Exception as e:
            print(f"{name:20s}: ERROR - {e}")
    
    print("\nAdvanced models ready for Week 4 optimization!")