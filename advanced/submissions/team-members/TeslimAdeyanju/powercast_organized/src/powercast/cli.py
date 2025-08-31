#!/usr/bin/env python3
"""
PowerCast CLI Module
Command-line interface for training and evaluating PowerCast models
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from powercast.week3_neural_networks import PowerCastModelBuilder, PowerCastTrainer, ModelEvaluator
from powercast.week2_feature_engineering import PreprocessingPipeline
from powercast.utils import load_and_validate_data

def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('powercast.log')
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path="config/config.yml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default settings.")
        return get_default_config()

def get_default_config():
    """Return default configuration"""
    return {
        'data': {
            'file_path': 'data/data.csv',
            'target_columns': ['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']
        },
        'training': {
            'epochs': 50,
            'batch_size': 32
        },
        'models': {
            'lstm': {'units': [64, 32]},
            'gru': {'units': [64, 32]},
            'dense': {'units': [128, 64, 32]}
        }
    }

def train_models():
    """Train PowerCast models"""
    parser = argparse.ArgumentParser(description='Train PowerCast models')
    parser.add_argument('--config', '-c', default='config/config.yml',
                       help='Path to configuration file')
    parser.add_argument('--data', '-d', default='data/data.csv',
                       help='Path to data file')
    parser.add_argument('--models', '-m', nargs='+', 
                       choices=['lstm', 'gru', 'dense'], 
                       default=['lstm', 'gru', 'dense'],
                       help='Models to train')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--output-dir', '-o', default='models/',
                       help='Output directory for trained models')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    logger.info("Starting PowerCast model training...")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load and validate data
        logger.info(f"Loading data from {args.data}")
        df = load_and_validate_data(args.data)
        
        # Setup preprocessing
        preprocessor = PreprocessingPipeline()
        X_train, X_test, y_train, y_test = preprocessor.prepare_sequences(df)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train each requested model
        for model_type in args.models:
            logger.info(f"Training {model_type.upper()} model...")
            
            # Build model
            builder = PowerCastModelBuilder()
            if model_type == 'lstm':
                model = builder.build_lstm_model(X_train.shape)
            elif model_type == 'gru':
                model = builder.build_gru_model(X_train.shape)
            else:  # dense
                model = builder.build_dense_model(X_train.shape)
            
            # Train model
            trainer = PowerCastTrainer(model)
            history = trainer.train_model(
                X_train, y_train, X_test, y_test,
                epochs=args.epochs,
                batch_size=config.get('training', {}).get('batch_size', 32)
            )
            
            # Save model
            model_path = os.path.join(args.output_dir, f'powercast_{model_type}_model.h5')
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

def evaluate_models():
    """Evaluate trained PowerCast models"""
    parser = argparse.ArgumentParser(description='Evaluate PowerCast models')
    parser.add_argument('--config', '-c', default='config/config.yml',
                       help='Path to configuration file')
    parser.add_argument('--data', '-d', default='data/data.csv',
                       help='Path to data file')
    parser.add_argument('--models-dir', '-m', default='models/',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', '-o', default='results/',
                       help='Output directory for evaluation results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    logger.info("Starting PowerCast model evaluation...")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load and validate data
        logger.info(f"Loading data from {args.data}")
        df = load_and_validate_data(args.data)
        
        # Setup preprocessing
        preprocessor = PreprocessingPipeline()
        X_train, X_test, y_train, y_test = preprocessor.prepare_sequences(df)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find and evaluate models
        model_files = list(Path(args.models_dir).glob('*.h5'))
        
        if not model_files:
            logger.warning(f"No model files found in {args.models_dir}")
            return
        
        evaluator = ModelEvaluator()
        
        for model_file in model_files:
            logger.info(f"Evaluating model: {model_file.name}")
            
            # Load model
            from tensorflow.keras.models import load_model
            model = load_model(str(model_file))
            
            # Evaluate model
            results = evaluator.evaluate_model(model, X_test, y_test)
            
            # Save results
            results_file = os.path.join(args.output_dir, f'{model_file.stem}_results.txt')
            with open(results_file, 'w') as f:
                f.write(f"Evaluation Results for {model_file.name}\n")
                f.write("=" * 50 + "\n")
                for metric, value in results.items():
                    f.write(f"{metric}: {value:.6f}\n")
            
            logger.info(f"Results saved to {results_file}")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='PowerCast CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add subcommands
    subparsers.add_parser('train', help='Train PowerCast models')
    subparsers.add_parser('evaluate', help='Evaluate PowerCast models')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_models()
    elif args.command == 'evaluate':
        evaluate_models()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
