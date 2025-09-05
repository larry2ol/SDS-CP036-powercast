"""
Diagnostic script to understand why LSTM/GRU failed so badly
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from models import LSTMBaseline, GRUAlternative, TemporalConvNet
from week2_feature_engineering_final import PowerConsumptionDataset
from torch.utils.data import DataLoader

def load_data_and_metadata():
    """Load preprocessed data and metadata"""
    print("Loading data...")
    
    # Load sequences and targets
    train_sequences = np.load('train_sequences.npy')
    train_targets = np.load('train_targets.npy')
    val_sequences = np.load('val_sequences.npy') 
    val_targets = np.load('val_targets.npy')
    
    with open('dataset_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create small subset for testing
    subset_size = 1000
    train_dataset = PowerConsumptionDataset(
        train_sequences[:subset_size], 
        train_targets[:subset_size]
    )
    val_dataset = PowerConsumptionDataset(
        val_sequences[:subset_size//4], 
        val_targets[:subset_size//4]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, metadata

def analyze_data_statistics(train_loader):
    """Analyze input and target statistics"""
    print("\n" + "="*50)
    print("DATA STATISTICS ANALYSIS")
    print("="*50)
    
    all_sequences = []
    all_targets = []
    
    for sequences, targets in train_loader:
        all_sequences.append(sequences)
        all_targets.append(targets)
    
    sequences = torch.cat(all_sequences, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    print(f"Input sequences shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    print()
    
    # Sequence statistics
    seq_mean = sequences.mean(dim=(0,1))  # Mean across batch and time
    seq_std = sequences.std(dim=(0,1))
    seq_min = sequences.min()
    seq_max = sequences.max()
    
    print("INPUT SEQUENCES:")
    print(f"  Overall range: [{seq_min:.4f}, {seq_max:.4f}]")
    print(f"  Per-feature means: {seq_mean}")
    print(f"  Per-feature stds: {seq_std}")
    print()
    
    # Target statistics
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0)
    target_min = targets.min()
    target_max = targets.max()
    
    print("TARGETS:")
    print(f"  Range: [{target_min:.2f}, {target_max:.2f}]")
    print(f"  Zone means: {target_mean}")
    print(f"  Zone stds: {target_std}")
    print(f"  Overall target mean: {targets.mean():.2f}")
    print(f"  Overall target std: {targets.std():.2f}")
    
    return sequences, targets

def test_model_forward_passes(sequences, targets, input_size=11, output_size=3):
    """Test forward passes and gradients for each model type"""
    print("\n" + "="*50)
    print("MODEL FORWARD PASS ANALYSIS")
    print("="*50)
    
    # Take a small batch
    batch_seq = sequences[:8]  
    batch_targets = targets[:8]
    
    models = {
        'LSTM_Medium': LSTMBaseline(input_size=input_size, hidden_sizes=[128, 64], output_size=output_size),
        'GRU_Medium': GRUAlternative(input_size=input_size, hidden_sizes=[128, 64], output_size=output_size),
        'TCN_Medium': TemporalConvNet(input_size=input_size, num_channels=[64, 128, 64], output_size=output_size)
    }
    
    criterion = nn.L1Loss()
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        model.train()
        
        # Forward pass
        with torch.no_grad():
            predictions = model(batch_seq)
        
        # Statistics
        pred_mean = predictions.mean().item()
        pred_std = predictions.std().item()
        pred_min = predictions.min().item()
        pred_max = predictions.max().item()
        
        print(f"  Predictions range: [{pred_min:.2f}, {pred_max:.2f}]")
        print(f"  Predictions mean: {pred_mean:.2f}, std: {pred_std:.2f}")
        print(f"  Target mean: {batch_targets.mean():.2f}")
        
        # Loss calculation
        loss = criterion(predictions, batch_targets)
        print(f"  Initial loss: {loss.item():.2f}")
        
        # Gradient analysis
        model.zero_grad()
        loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0
        param_count = 0
        for name_param, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
        
        avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
        print(f"  Average gradient norm: {avg_grad_norm:.6f}")
        
        results[name] = {
            'initial_loss': loss.item(),
            'pred_range': [pred_min, pred_max],
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'grad_norm': avg_grad_norm
        }
    
    return results

def test_different_learning_rates(sequences, targets, input_size=11, output_size=3):
    """Test how different learning rates affect initial training steps"""
    print("\n" + "="*50)
    print("LEARNING RATE SENSITIVITY ANALYSIS")
    print("="*50)
    
    batch_seq = sequences[:32]
    batch_targets = targets[:32]
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    for lr in learning_rates:
        print(f"\nLearning Rate: {lr}")
        print("-" * 30)
        
        # Test LSTM
        lstm = LSTMBaseline(input_size=input_size, hidden_sizes=[128, 64], output_size=output_size)
        optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=lr)
        criterion = nn.L1Loss()
        
        # Test TCN for comparison
        tcn = TemporalConvNet(input_size=input_size, num_channels=[64, 128, 64], output_size=output_size)
        optimizer_tcn = torch.optim.Adam(tcn.parameters(), lr=lr)
        
        # 5 training steps
        lstm_losses = []
        tcn_losses = []
        
        for step in range(5):
            # LSTM step
            optimizer_lstm.zero_grad()
            pred_lstm = lstm(batch_seq)
            loss_lstm = criterion(pred_lstm, batch_targets)
            loss_lstm.backward()
            optimizer_lstm.step()
            lstm_losses.append(loss_lstm.item())
            
            # TCN step  
            optimizer_tcn.zero_grad()
            pred_tcn = tcn(batch_seq)
            loss_tcn = criterion(pred_tcn, batch_targets)
            loss_tcn.backward()
            optimizer_tcn.step()
            tcn_losses.append(loss_tcn.item())
        
        print(f"  LSTM losses: {[f'{l:.1f}' for l in lstm_losses]}")
        print(f"  TCN losses:  {[f'{l:.1f}' for l in tcn_losses]}")

def test_simpler_rnn_architectures(sequences, targets, input_size=11, output_size=3):
    """Test if simpler RNN architectures work better"""
    print("\n" + "="*50) 
    print("SIMPLE RNN ARCHITECTURE TEST")
    print("="*50)
    
    batch_seq = sequences[:32]
    batch_targets = targets[:32]
    criterion = nn.L1Loss()
    
    # Test very simple architectures
    configs = [
        ('LSTM_Simple', {'hidden_sizes': [64], 'dropout_rate': 0.0}),
        ('LSTM_NoDropout', {'hidden_sizes': [128, 64], 'dropout_rate': 0.0}),
        ('LSTM_SmallLR', {'hidden_sizes': [128, 64], 'dropout_rate': 0.2})
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        model = LSTMBaseline(input_size=input_size, output_size=output_size, **config)
        
        # Use smaller learning rate for SmallLR variant
        lr = 0.0001 if 'SmallLR' in name else 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Test 5 steps
        for step in range(5):
            optimizer.zero_grad()
            pred = model(batch_seq)
            loss = criterion(pred, batch_targets)
            loss.backward()
            optimizer.step()
            
            if step == 0:
                print(f"  Initial loss: {loss.item():.2f}")
            elif step == 4:
                print(f"  After 5 steps: {loss.item():.2f}")
                improvement = (loss.item() / configs[0][1].get('initial_loss', loss.item())) if step == 0 else 0
                print(f"  Learning rate: {lr}")

def main():
    """Main diagnostic function"""
    print("RNN FAILURE DIAGNOSTIC ANALYSIS")
    print("="*60)
    
    # Load data
    train_loader, val_loader, metadata = load_data_and_metadata()
    
    # Analyze data statistics
    sequences, targets = analyze_data_statistics(train_loader)
    
    # Test model forward passes
    forward_results = test_model_forward_passes(sequences, targets)
    
    # Test learning rate sensitivity
    test_different_learning_rates(sequences, targets)
    
    # Test simpler architectures
    test_simpler_rnn_architectures(sequences, targets)
    
    # Summary recommendations
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    print("\nKey Findings:")
    print("1. Check if target values are properly scaled")
    print("2. RNN models may need lower learning rates")
    print("3. Architecture complexity might be an issue")
    print("4. Gradient flow problems in RNN training")
    
    print("\nRecommended fixes:")
    print("- Try learning rates: 0.0001, 0.00001")
    print("- Remove or reduce dropout") 
    print("- Test single-layer LSTM/GRU")
    print("- Check data preprocessing differences")
    print("- Consider different loss functions (MSE vs MAE)")
    
    return forward_results

if __name__ == "__main__":
    results = main()