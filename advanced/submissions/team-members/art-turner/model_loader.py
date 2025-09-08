"""
Model loading utilities for analysis scripts
Handles checkpoint format and model reconstruction
"""

import torch
from advanced_models import AttentionLSTM, BidirectionalLSTM, DeepLSTM, AdvancedTCN
from models import LSTMBaseline, GRUAlternative, TemporalConvNet

def load_model_from_checkpoint(model_path, input_size=11, output_size=3, device='cpu'):
    """
    Load model from checkpoint format saved by training_fixed_refactor.py
    
    Args:
        model_path: Path to the .pth file
        input_size: Number of input features (default 11)
        output_size: Number of output targets (default 3)
        device: Device to load model on
    
    Returns:
        model: Loaded PyTorch model ready for inference
        checkpoint: Original checkpoint data (for metadata)
    """
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle both direct model saves and checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # It's our checkpoint format - reconstruct model
        model_class_name = checkpoint.get('model_class', 'Unknown')
        print(f"   Reconstructing {model_class_name} from checkpoint...")
        
        # Get model config (if saved) or infer from state dict
        model_config = checkpoint.get('model_config', {})
        if not model_config:
            print("   No model config found, inferring from state dict...")
            model_config = infer_model_config_from_state_dict(
                checkpoint['model_state_dict'], model_class_name
            )
        
        # Create model based on class name and inferred parameters
        model = create_model_from_checkpoint(
            model_class_name=model_class_name,
            checkpoint=checkpoint,
            model_config=model_config,
            input_size=input_size,
            output_size=output_size
        )
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, checkpoint
        
    else:
        # It's a direct model save
        model = checkpoint.to(device)
        model.eval()
        return model, None

def create_model_from_checkpoint(model_class_name, checkpoint, model_config, input_size, output_size):
    """
    Create model instance based on class name and checkpoint data
    """
    
    # model_config is now passed as parameter
    
    # Model class mapping
    model_classes = {
        'AttentionLSTM': AttentionLSTM,
        'BidirectionalLSTM': BidirectionalLSTM, 
        'DeepLSTM': DeepLSTM,
        'AdvancedTCN': AdvancedTCN,
        'LSTMBaseline': LSTMBaseline,
        'GRUAlternative': GRUAlternative,
        'TemporalConvNet': TemporalConvNet
    }
    
    if model_class_name not in model_classes:
        raise ValueError(f"Unknown model class: {model_class_name}")
    
    model_class = model_classes[model_class_name]
    
    # Create model with appropriate parameters based on class
    if model_class_name == 'AttentionLSTM':
        # AttentionLSTM parameters: hidden_size, num_layers, dropout_rate
        model = model_class(
            input_size=input_size,
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2), 
            output_size=output_size,
            dropout_rate=model_config.get('dropout_rate', 0.2)
        )
        
    elif model_class_name in ['BidirectionalLSTM', 'LSTMBaseline', 'GRUAlternative']:
        # These use hidden_sizes list
        model = model_class(
            input_size=input_size,
            hidden_sizes=model_config.get('hidden_sizes', [128, 64]),
            output_size=output_size,
            dropout_rate=model_config.get('dropout_rate', 0.2)
        )
        
    elif model_class_name == 'DeepLSTM':
        # DeepLSTM parameters
        model = model_class(
            input_size=input_size,
            hidden_sizes=model_config.get('hidden_sizes', [256, 128, 64, 32]),
            output_size=output_size,
            dropout_rate=model_config.get('dropout_rate', 0.3),
            layer_norm=model_config.get('layer_norm', True)
        )
        
    elif model_class_name == 'TemporalConvNet':
        # TCN parameters
        model = model_class(
            input_size=input_size,
            num_channels=model_config.get('num_channels', [64, 128, 64]),
            output_size=output_size,
            dropout=model_config.get('dropout', 0.2)
        )
        
    elif model_class_name == 'AdvancedTCN':
        # AdvancedTCN parameters  
        model = model_class(
            input_size=input_size,
            num_channels=model_config.get('num_channels', [64, 128, 256, 128, 64]),
            output_size=output_size,
            dropout=model_config.get('dropout', 0.2),
            use_attention=model_config.get('use_attention', True)
        )
        
    else:
        # Default fallback - try with basic parameters
        try:
            model = model_class(
                input_size=input_size,
                output_size=output_size
            )
        except:
            raise ValueError(f"Could not create model {model_class_name} - unknown parameter structure")
    
    print(f"   Created {model_class_name} with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model

def infer_model_config_from_state_dict(state_dict, model_class_name):
    """
    Try to infer model configuration from state dictionary
    This is a fallback when model_config is not saved
    """
    config = {}
    
    try:
        if model_class_name == 'AttentionLSTM':
            # Look for LSTM hidden size from weight shapes
            if 'lstm.weight_ih_l0' in state_dict:
                # LSTM weight_ih shape is [4*hidden_size, input_size]
                hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
                config['hidden_size'] = hidden_size
                
                # Count LSTM layers
                num_layers = 0
                layer_idx = 0
                while f'lstm.weight_ih_l{layer_idx}' in state_dict:
                    num_layers += 1
                    layer_idx += 1
                config['num_layers'] = num_layers
                
                # Infer dropout rate from attention layer if possible
                config['dropout_rate'] = 0.2  # Default
                
                print(f"   Inferred: hidden_size={hidden_size}, num_layers={num_layers}")
                    
        elif model_class_name in ['BidirectionalLSTM', 'LSTMBaseline', 'GRUAlternative']:
            # Infer hidden sizes from layer weights
            hidden_sizes = []
            layer_idx = 0
            
            # Different architectures have different naming patterns
            if model_class_name == 'BidirectionalLSTM':
                weight_pattern = 'lstm_layers.{}.weight_ih_l0'
                gate_count = 4
            elif model_class_name == 'LSTMBaseline':
                weight_pattern = 'lstm_layers.{}.weight_ih_l0'
                gate_count = 4
            elif model_class_name == 'GRUAlternative':
                weight_pattern = 'gru_layers.{}.weight_ih_l0'
                gate_count = 3
                
            while True:
                weight_name = weight_pattern.format(layer_idx)
                if weight_name in state_dict:
                    hidden_size = state_dict[weight_name].shape[0] // gate_count
                    hidden_sizes.append(hidden_size)
                    layer_idx += 1
                else:
                    break
                    
            if hidden_sizes:
                config['hidden_sizes'] = hidden_sizes
                print(f"   Inferred: hidden_sizes={hidden_sizes}")
                
        elif model_class_name in ['TemporalConvNet', 'AdvancedTCN']:
            # For TCN models, look at conv layer weights
            num_channels = []
            layer_idx = 0
            
            while True:
                weight_name = f'network.{layer_idx}.conv1.weight'
                if weight_name in state_dict:
                    out_channels = state_dict[weight_name].shape[0]
                    num_channels.append(out_channels)
                    layer_idx += 1
                else:
                    break
                    
            if num_channels:
                config['num_channels'] = num_channels
                print(f"   Inferred: num_channels={num_channels}")
                
    except Exception as e:
        print(f"   Warning: Could not infer config from state dict: {e}")
        
    return config