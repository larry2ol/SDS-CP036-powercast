"""
Quick test of RNN models with fixed normalized targets
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from models import LSTMBaseline, GRUAlternative, TemporalConvNet
from week2_feature_engineering_fixed import PowerConsumptionDataset
from torch.utils.data import DataLoader

def test_fixed_models():
    print("TESTING RNN MODELS WITH FIXED NORMALIZED TARGETS")
    print("="*60)
    
    # Load fixed data
    print("Loading fixed normalized data...")
    train_sequences = np.load('train_sequences_fixed.npy')
    train_targets = np.load('train_targets_fixed.npy')
    
    # Create small dataset for quick test
    subset_size = 1000
    train_dataset = PowerConsumptionDataset(
        train_sequences[:subset_size], 
        train_targets[:subset_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Load target scaler for denormalization
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    print(f"Data shapes: sequences={train_sequences.shape}, targets={train_targets.shape}")
    print(f"Target range: [{train_targets.min():.3f}, {train_targets.max():.3f}]")
    print()
    
    # Test models
    models = {
        'LSTM_Medium': LSTMBaseline(input_size=11, hidden_sizes=[128, 64], output_size=3),
        'GRU_Medium': GRUAlternative(input_size=11, hidden_sizes=[128, 64], output_size=3), 
        'TCN_Medium': TemporalConvNet(input_size=11, num_channels=[64, 128, 64], output_size=3)
    }
    
    criterion = nn.L1Loss()
    
    print("QUICK TRAINING TEST (10 steps each):")
    print("-" * 40)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        
        losses = []
        
        # Test 10 training steps
        batch_iter = iter(train_loader)
        for step in range(10):
            try:
                batch_seq, batch_targets = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch_seq, batch_targets = next(batch_iter)
            
            optimizer.zero_grad()
            predictions = model(batch_seq)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if step == 0:
                print(f"  Initial loss: {loss.item():.4f}")
            elif step == 9:
                print(f"  After 10 steps: {loss.item():.4f}")
                improvement = (losses[0] - losses[-1]) / losses[0] * 100
                print(f"  Improvement: {improvement:.1f}%")
                
                # Test denormalization
                pred_denorm = target_scaler.inverse_transform(predictions[:5].detach().numpy())
                target_denorm = target_scaler.inverse_transform(batch_targets[:5].numpy())
                print(f"  Sample predictions (denorm): {pred_denorm[0]}")
                print(f"  Sample targets (denorm): {target_denorm[0]}")
    
    print("\n" + "="*60)
    print("RESULTS ANALYSIS:")
    print("="*60)
    print("If RNN models now show:")
    print("- Initial losses around 0.1-0.5 (not 15,000+)")
    print("- Clear improvement over 10 steps")
    print("- Reasonable denormalized predictions")
    print("Then the target normalization fix is working!")

if __name__ == "__main__":
    test_fixed_models()