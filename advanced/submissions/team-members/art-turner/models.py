"""
Week 3: Neural Network Models for Power Consumption Forecasting
LSTM, GRU, and TCN architectures with proper initialization and regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class LSTMBaseline(nn.Module):
    """
    Multi-layer LSTM baseline model with dropout and proper initialization
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [128, 64],
                 output_size: int = 3,
                 dropout_rate: float = 0.2,
                 bidirectional: bool = False):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden dimensions for each LSTM layer
            output_size: Number of output targets (3 zones)
            dropout_rate: Dropout probability between LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMBaseline, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.num_layers = len(hidden_sizes)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[0],
                batch_first=True,
                dropout=dropout_rate if len(hidden_sizes) > 1 else 0,
                bidirectional=bidirectional
            )
        )
        
        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            lstm_input_size = hidden_sizes[i-1] * (2 if bidirectional else 1)
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=hidden_sizes[i],
                    batch_first=True,
                    dropout=dropout_rate if i < len(hidden_sizes) - 1 else 0,
                    bidirectional=bidirectional
                )
            )
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        final_hidden_size = hidden_sizes[-1] * (2 if bidirectional else 1)
        self.fc = nn.Linear(final_hidden_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases
                nn.init.constant_(param.data, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        # Initialize linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            output: Predictions of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Pass through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            # Apply dropout between layers (except after last layer)
            if i < len(self.lstm_layers) - 1:
                x = self.dropout(x)
        
        # Take the last time step output
        x = x[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply final dropout
        x = self.dropout(x)
        
        # Linear output layer
        output = self.fc(x)
        
        return output


class GRUAlternative(nn.Module):
    """
    GRU-based alternative architecture with similar structure to LSTM
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = [128, 64],
                 output_size: int = 3,
                 dropout_rate: float = 0.2,
                 bidirectional: bool = False):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden dimensions for each GRU layer
            output_size: Number of output targets (3 zones)
            dropout_rate: Dropout probability between GRU layers
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUAlternative, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.num_layers = len(hidden_sizes)
        
        # GRU layers
        self.gru_layers = nn.ModuleList()
        
        # First GRU layer
        self.gru_layers.append(
            nn.GRU(
                input_size=input_size,
                hidden_size=hidden_sizes[0],
                batch_first=True,
                dropout=dropout_rate if len(hidden_sizes) > 1 else 0,
                bidirectional=bidirectional
            )
        )
        
        # Additional GRU layers
        for i in range(1, len(hidden_sizes)):
            gru_input_size = hidden_sizes[i-1] * (2 if bidirectional else 1)
            self.gru_layers.append(
                nn.GRU(
                    input_size=gru_input_size,
                    hidden_size=hidden_sizes[i],
                    batch_first=True,
                    dropout=dropout_rate if i < len(hidden_sizes) - 1 else 0,
                    bidirectional=bidirectional
                )
            )
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        final_hidden_size = hidden_sizes[-1] * (2 if bidirectional else 1)
        self.fc = nn.Linear(final_hidden_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        # Initialize linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            output: Predictions of shape (batch_size, output_size)
        """
        # Pass through GRU layers
        for i, gru in enumerate(self.gru_layers):
            x, _ = gru(x)
            # Apply dropout between layers (except after last layer)
            if i < len(self.gru_layers) - 1:
                x = self.dropout(x)
        
        # Take the last time step output
        x = x[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply final dropout
        x = self.dropout(x)
        
        # Linear output layer
        output = self.fc(x)
        
        return output


class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with dilated convolutions and residual connections
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Weight initialization
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
        
        # Residual connection - ensure same sequence length
        res = x if self.downsample is None else self.downsample(x)
        
        # Crop to match shorter sequence if needed
        if out.size(-1) != res.size(-1):
            min_length = min(out.size(-1), res.size(-1))
            out = out[:, :, :min_length]
            res = res[:, :, :min_length]
        
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling
    """
    
    def __init__(self, 
                 input_size: int,
                 num_channels: List[int] = [64, 128, 64],
                 output_size: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        """
        Args:
            input_size: Number of input features
            num_channels: List of channel sizes for each temporal block
            output_size: Number of output targets (3 zones)
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.output_size = output_size
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding to maintain sequence length
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=padding, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
        # Global average pooling and output layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            output: Predictions of shape (batch_size, output_size)
        """
        # TCN expects (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        
        # Pass through temporal blocks
        y = self.network(x)
        
        # Global average pooling
        y = self.adaptive_pool(y)  # (batch_size, num_channels[-1], 1)
        y = y.squeeze(-1)  # (batch_size, num_channels[-1])
        
        # Output layer
        output = self.fc(y)
        
        return output


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, input_shape=(1, 36, 11)):
    """Get a summary of model architecture and parameter count"""
    total_params = count_parameters(model)
    
    # Create dummy input to test forward pass
    dummy_input = torch.randn(input_shape)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            output_shape = tuple(output.shape)
    except Exception as e:
        output_shape = f"Error: {e}"
    
    return {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'trainable_params': f"{total_params:,}"
    }


def create_models(input_size: int = 11, output_size: int = 3) -> Dict[str, nn.Module]:
    """
    Create all three model architectures with different configurations
    
    Args:
        input_size: Number of input features
        output_size: Number of output targets
    
    Returns:
        Dictionary containing all model instances
    """
    models = {}
    
    # LSTM configurations
    lstm_configs = {
        'LSTM_Small': {'hidden_sizes': [64, 32], 'dropout_rate': 0.2},
        'LSTM_Medium': {'hidden_sizes': [128, 64], 'dropout_rate': 0.2},
        'LSTM_Large': {'hidden_sizes': [256, 128, 64], 'dropout_rate': 0.3}
    }
    
    for name, config in lstm_configs.items():
        models[name] = LSTMBaseline(
            input_size=input_size,
            output_size=output_size,
            **config
        )
    
    # GRU configurations
    gru_configs = {
        'GRU_Small': {'hidden_sizes': [64, 32], 'dropout_rate': 0.2},
        'GRU_Medium': {'hidden_sizes': [128, 64], 'dropout_rate': 0.2},
        'GRU_Large': {'hidden_sizes': [256, 128, 64], 'dropout_rate': 0.3}
    }
    
    for name, config in gru_configs.items():
        models[name] = GRUAlternative(
            input_size=input_size,
            output_size=output_size,
            **config
        )
    
    # TCN configurations
    tcn_configs = {
        'TCN_Small': {'num_channels': [32, 64, 32], 'dropout': 0.2},
        'TCN_Medium': {'num_channels': [64, 128, 64], 'dropout': 0.2},
        'TCN_Large': {'num_channels': [128, 256, 128, 64], 'dropout': 0.3}
    }
    
    for name, config in tcn_configs.items():
        models[name] = TemporalConvNet(
            input_size=input_size,
            output_size=output_size,
            **config
        )
    
    return models


if __name__ == "__main__":
    print("Testing Neural Network Architectures")
    print("=" * 50)
    
    # Create all models
    models = create_models()
    
    # Test each model
    for name, model in models.items():
        print(f"\n{name}:")
        summary = get_model_summary(model)
        print(f"  Parameters: {summary['trainable_params']}")
        print(f"  Input shape: {summary['input_shape']}")
        print(f"  Output shape: {summary['output_shape']}")
    
    print("\nAll models created successfully!")