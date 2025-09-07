# 🔋 PowerCast: Deep Learning for Time-Series Power Consumption Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Advanced time-series forecasting for power consumption in Tetouan City using deep learning neural networks

## 📁 Project Structure

```
powercast_project/
├── README.md                 # Project overview and setup instructions
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation setup
├── config/                  # Configuration files
│   ├── model_config.yaml   # Model hyperparameters
│   └── data_config.yaml    # Data processing settings
├── data/                    # Dataset files
│   ├── data.csv            # Main Tetouan power consumption dataset
│   └── processed/          # Processed data files
├── src/                     # Source code modules
│   └── modules/            # Core PowerCast modules
│       ├── __init__.py     # Package initialization
│       ├── utils.py        # Common utilities
│       ├── week1_eda.py    # Exploratory Data Analysis
│       ├── week2_feature_engineering.py  # Feature engineering
│       └── week3_neural_networks.py      # Neural network models
├── notebooks/               # Jupyter notebooks
│   ├── powercast_analysis.ipynb  # Main analysis notebook
│   └── experiments/        # Additional experiment notebooks
├── models/                  # Trained model files
│   ├── best_lstm_powercast_model.h5
│   ├── best_gru_powercast_model.h5
│   └── best_dense_powercast_model.h5
└── docs/                    # Documentation
    ├── README.md           # Original project documentation
    ├── REPORT.md           # Detailed project report
    └── QA.md               # Questions and answers
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd powercast_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install PowerCast Package

```bash
# Install in development mode
pip install -e .
```

### 3. Run the Analysis

```bash
# Start Jupyter Lab
jupyter lab

# Open the main notebook
# notebooks/powercast_analysis.ipynb
```

## 📊 Features

- **🔍 Comprehensive EDA**: Multi-dimensional exploratory data analysis
- **⚡ Feature Engineering**: Advanced time-series preprocessing pipeline
- **🧠 Neural Networks**: LSTM, GRU, and Dense model implementations
- **📈 Model Evaluation**: Comprehensive performance assessment
- **🔧 Modular Design**: Clean, reusable code architecture
- **📚 Documentation**: Extensive Q&A and reporting

## 🏗️ Architecture

### Core Modules

- **`week1_eda.py`**: Data exploration, visualization, and pattern analysis
- **`week2_feature_engineering.py`**: Sequence generation, scaling, and preprocessing
- **`week3_neural_networks.py`**: Model building, training, and evaluation
- **`utils.py`**: Common utilities and helper functions

### Model Performance

| Model | Parameters | MAE | RMSE | R² |
|-------|------------|-----|------|-----|
| LSTM  | 19,976     | ★★★★★ | ★★★★★ | 0.85+ |
| GRU   | 15,401     | ★★★★☆ | ★★★★☆ | 0.82+ |
| Dense | 161,921    | ★★★☆☆ | ★★★☆☆ | 0.75+ |

## 🔬 Usage Examples

### Basic Usage

```python
# Import PowerCast modules
from src.modules.week1_eda import DataExplorer, TimeSeriesAnalyzer
from src.modules.week2_feature_engineering import PreprocessingPipeline
from src.modules.week3_neural_networks import PowerCastModelBuilder, PowerCastTrainer

# Load and explore data
explorer = DataExplorer(data)
eda_report = explorer.generate_comprehensive_report()

# Preprocess data
pipeline = PreprocessingPipeline()
X_train, y_train = pipeline.create_sequences(data, sequence_length=144)

# Build and train models
builder = PowerCastModelBuilder(input_shape=(144, 7))
lstm_model = builder.build_lstm_model()
trainer = PowerCastTrainer()
history = trainer.train_model(lstm_model, X_train, y_train)
```

### Advanced Configuration

```python
# Custom model configuration
model_config = {
    'lstm_units': [64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

# Build custom model
custom_model = builder.build_lstm_model(**model_config)
```

## 📝 Documentation

- **[Project Report](docs/REPORT.md)**: Comprehensive analysis and findings
- **[Q&A Documentation](docs/QA.md)**: Detailed answers to project questions
- **[API Reference](docs/api/)**: Module and function documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Teslim Uthman Adeyanju**
- 📧 Email: [info@adeyanjuteslim.co.uk](mailto:info@adeyanjuteslim.co.uk)
- 🔗 LinkedIn: [linkedin.com/in/adeyanjuteslimuthman](https://www.linkedin.com/in/adeyanjuteslimuthman)
- 🌐 Website: [adeyanjuteslim.co.uk](https://adeyanjuteslim.co.uk)

## 🙏 Acknowledgments

- Tetouan City power grid data providers
- TensorFlow and Keras development teams
- Open source community contributors

---

⭐ **Star this repository if it helped you!**
