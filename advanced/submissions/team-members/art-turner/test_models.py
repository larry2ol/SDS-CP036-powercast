"""
Test script for Week 3 neural network models
Verify model architectures and basic functionality
"""

import torch
import torch.nn as nn
import numpy as np
from models import LSTMBaseline, GRUAlternative, TemporalConvNet, create_models, get_model_summary
import json


def test_model_creation():
    """Test that all models can be created without errors"""
    print("Testing Model Creation...")
    print("-" * 40)
    
    input_size = 11
    output_size = 3
    
    try:
        # Test individual model creation
        lstm = LSTMBaseline(input_size=input_size, output_size=output_size)
        gru = GRUAlternative(input_size=input_size, output_size=output_size)
        tcn = TemporalConvNet(input_size=input_size, output_size=output_size)
        
        print("[[OK]] Individual models created successfully")
        
        # Test batch model creation
        models = create_models(input_size=input_size, output_size=output_size)
        print(f"[OK] Created {len(models)} model variants")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error creating models: {e}")
        return False


def test_forward_pass():
    """Test forward pass through all models"""
    print("\nTesting Forward Pass...")
    print("-" * 40)
    
    # Create dummy input
    batch_size = 4
    sequence_length = 36
    input_size = 11
    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    
    models = create_models(input_size=input_size)
    
    success_count = 0
    for model_name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output shape
            expected_shape = (batch_size, 3)
            if output.shape == expected_shape:
                print(f"[OK] {model_name}: Output shape {output.shape}")
                success_count += 1
            else:
                print(f"[FAIL] {model_name}: Wrong output shape {output.shape}, expected {expected_shape}")
        
        except Exception as e:
            print(f"[FAIL] {model_name}: Forward pass failed - {e}")
    
    print(f"\nSuccessful forward passes: {success_count}/{len(models)}")
    return success_count == len(models)


def test_gradient_flow():
    """Test that gradients can flow through the models"""
    print("\nTesting Gradient Flow...")
    print("-" * 40)
    
    # Create dummy data
    batch_size = 2
    sequence_length = 36
    input_size = 11
    output_size = 3
    
    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    dummy_target = torch.randn(batch_size, output_size)
    
    # Test with one model from each architecture
    test_models = {
        'LSTM': LSTMBaseline(input_size=input_size, output_size=output_size),
        'GRU': GRUAlternative(input_size=input_size, output_size=output_size),
        'TCN': TemporalConvNet(input_size=input_size, output_size=output_size)
    }
    
    success_count = 0
    for model_name, model in test_models.items():
        try:
            # Forward pass
            output = model(dummy_input)
            
            # Compute loss
            criterion = nn.MSELoss()
            loss = criterion(output, dummy_target)
            
            # Backward pass
            loss.backward()
            
            # Check if gradients exist
            has_gradients = any(param.grad is not None for param in model.parameters())
            
            if has_gradients:
                print(f"[OK] {model_name}: Gradients computed successfully")
                success_count += 1
            else:
                print(f"[FAIL] {model_name}: No gradients found")
            
            # Clear gradients
            model.zero_grad()
            
        except Exception as e:
            print(f"[FAIL] {model_name}: Gradient computation failed - {e}")
    
    print(f"\nSuccessful gradient computations: {success_count}/{len(test_models)}")
    return success_count == len(test_models)


def test_model_summaries():
    """Test model summary generation"""
    print("\nTesting Model Summaries...")
    print("-" * 40)
    
    models = create_models()
    
    for model_name, model in models.items():
        try:
            summary = get_model_summary(model)
            print(f"[OK] {model_name}: {summary['trainable_params']} parameters")
        except Exception as e:
            print(f"[FAIL] {model_name}: Summary generation failed - {e}")
    
    return True


def test_different_configurations():
    """Test models with different hyperparameter configurations"""
    print("\nTesting Different Configurations...")
    print("-" * 40)
    
    input_size = 11
    output_size = 3
    
    configurations = [
        # LSTM configurations
        {'model_class': LSTMBaseline, 'name': 'LSTM_Small', 
         'kwargs': {'hidden_sizes': [32], 'dropout_rate': 0.1}},
        {'model_class': LSTMBaseline, 'name': 'LSTM_Large', 
         'kwargs': {'hidden_sizes': [256, 128, 64], 'dropout_rate': 0.3}},
        {'model_class': LSTMBaseline, 'name': 'LSTM_Bidirectional', 
         'kwargs': {'hidden_sizes': [64], 'bidirectional': True}},
        
        # GRU configurations
        {'model_class': GRUAlternative, 'name': 'GRU_Small', 
         'kwargs': {'hidden_sizes': [32], 'dropout_rate': 0.1}},
        {'model_class': GRUAlternative, 'name': 'GRU_Large', 
         'kwargs': {'hidden_sizes': [256, 128, 64], 'dropout_rate': 0.3}},
        
        # TCN configurations
        {'model_class': TemporalConvNet, 'name': 'TCN_Small', 
         'kwargs': {'num_channels': [32, 32], 'dropout': 0.1}},
        {'model_class': TemporalConvNet, 'name': 'TCN_Large', 
         'kwargs': {'num_channels': [128, 256, 128, 64], 'dropout': 0.3}},
    ]
    
    success_count = 0
    dummy_input = torch.randn(2, 36, input_size)
    
    for config in configurations:
        try:
            model = config['model_class'](
                input_size=input_size,
                output_size=output_size,
                **config['kwargs']
            )
            
            # Test forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            if output.shape == (2, 3):
                print(f"[OK] {config['name']}: Configuration works")
                success_count += 1
            else:
                print(f"[FAIL] {config['name']}: Wrong output shape")
        
        except Exception as e:
            print(f"[FAIL] {config['name']}: Configuration failed - {e}")
    
    print(f"\nSuccessful configurations: {success_count}/{len(configurations)}")
    return success_count == len(configurations)


def performance_benchmark():
    """Basic performance benchmark of different architectures"""
    print("\nPerformance Benchmark...")
    print("-" * 40)
    
    import time
    
    batch_size = 32
    sequence_length = 36
    input_size = 11
    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    
    test_models = {
        'LSTM_Medium': LSTMBaseline(input_size=input_size, hidden_sizes=[128, 64], output_size=3),
        'GRU_Medium': GRUAlternative(input_size=input_size, hidden_sizes=[128, 64], output_size=3),
        'TCN_Medium': TemporalConvNet(input_size=input_size, num_channels=[64, 128, 64], output_size=3)
    }
    
    for model_name, model in test_models.items():
        model.eval()
        
        # Warm up
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Time forward passes
        n_runs = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy_input)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / n_runs * 1000  # Convert to milliseconds
        
        # Get parameter count
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"{model_name}: {avg_time:.2f}ms per forward pass, {param_count:,} parameters")


def run_all_tests():
    """Run all tests and provide summary"""
    print("NEURAL NETWORK MODEL TESTING")
    print("="*60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Gradient Flow", test_gradient_flow),
        ("Model Summaries", test_model_summaries),
        ("Different Configurations", test_different_configurations),
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"[FAIL] {test_name}: Test failed with exception - {e}")
    
    # Run performance benchmark
    performance_benchmark()
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed_tests}/{len(tests)} tests passed")
    print(f"{'='*60}")
    
    if passed_tests == len(tests):
        print("All tests passed! Models are ready for training.")
        return True
    else:
        print("Some tests failed. Please check the implementations.")
        return False


if __name__ == "__main__":
    run_all_tests()