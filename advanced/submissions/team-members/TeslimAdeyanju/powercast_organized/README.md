# 🔋 PowerCast: Deep Learning for Time-Series Power Consumption Forecasting

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A comprehensive deep learning solution for time-series power consumption forecasting in Tetouan City, Morocco. This project implements modular neural network architectures (LSTM, GRU, Dense) for accurate power consumption prediction.

## 📋 Table of Contents

- [🔋 PowerCast: Deep Learning for Time-Series Power Consumption Forecasting](#-powercast-deep-learning-for-time-series-power-consumption-forecasting)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🏗️ Project Structure](#️-project-structure)
  - [🚀 Quick Start](#-quick-start)
  - [📊 Usage](#-usage)
  - [🔧 Configuration](#-configuration)
  - [📈 Model Architectures](#-model-architectures)
  - [🎯 Results](#-results)
  - [📚 Documentation](#-documentation)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [👤 Author](#-author)

## ✨ Features

- **🧠 Multiple Neural Network Architectures**: LSTM, GRU, and Dense models
- **📊 Comprehensive EDA**: Automated exploratory data analysis tools
- **⚙️ Modular Design**: Clean, reusable, and extensible codebase
- **📈 Performance Evaluation**: Multi-metric model assessment
- **🔄 Preprocessing Pipeline**: Automated feature engineering and scaling
- **📝 Interactive Notebooks**: Detailed analysis and visualization
- **🚀 Production Ready**: Organized structure for deployment

## 🏗️ Project Structure

```
powercast_organized/
├── 📁 src/powercast/          # Core source code
│   ├── __init__.py           # Package initialization
│   ├── utils.py              # Common utilities
│   ├── week1_eda.py          # Exploratory Data Analysis
│   ├── week2_feature_engineering.py  # Feature engineering
│   └── week3_neural_networks.py      # Neural network models
├── 📁 notebooks/             # Jupyter notebooks
│   └── powercast_analysis.ipynb     # Main analysis notebook
├── 📁 data/                  # Dataset files
│   └── data.csv              # Tetouan City power consumption data
├── 📁 models/                # Trained model files
│   ├── best_lstm_powercast_model.h5
│   ├── best_gru_powercast_model.h5
│   └── best_dense_powercast_model.h5
├── 📁 docs/                  # Documentation
│   ├── README.md             # Original project README
│   ├── REPORT.md             # Detailed project report
│   └── QA.md                 # Questions and answers
├── 📁 tests/                 # Unit tests
│   └── test_modules.py       # Module tests
├── 📁 config/                # Configuration files
│   └── model_config.yml      # Model parameters
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── Makefile                  # Build automation
└── .gitignore               # Git ignore patterns
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd powercast_organized
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

### Quick Test

```bash
python -c "from src.powercast import week1_eda; print('✅ PowerCast modules imported successfully!')"
```

## 📊 Usage

### 1. Interactive Analysis

Open the main analysis notebook:
```bash
jupyter lab notebooks/powercast_analysis.ipynb
```

### 2. Using the Modules

```python
# Import PowerCast modules
from src.powercast.week1_eda import DataExplorer, TimeSeriesAnalyzer
from src.powercast.week2_feature_engineering import PreprocessingPipeline
from src.powercast.week3_neural_networks import PowerCastModelBuilder, PowerCastTrainer

# Load and explore data
explorer = DataExplorer(data)
explorer.basic_info()
explorer.plot_distributions()

# Feature engineering
pipeline = PreprocessingPipeline()
X_train, y_train = pipeline.create_sequences(data, sequence_length=144)

# Build and train models
builder = PowerCastModelBuilder(input_shape=(144, 7))
lstm_model = builder.build_lstm_model()

trainer = PowerCastTrainer()
trainer.train_model(lstm_model, X_train, y_train)
```

### 3. Model Training Pipeline

```bash
# Using the Makefile
make train          # Train all models
make evaluate       # Evaluate model performance
make test          # Run unit tests
```

## 🔧 Configuration

Model parameters can be configured in `config/model_config.yml`:

```yaml
models:
  lstm:
    units: [64, 32]
    dropout: 0.2
    learning_rate: 0.001
  gru:
    units: [64, 32]
    dropout: 0.2
    learning_rate: 0.001
```

## 📈 Model Architectures

### LSTM Model
- **Parameters**: 19,976
- **Architecture**: 2-layer LSTM (64, 32 units)
- **Best for**: Long-term dependencies, complex patterns

### GRU Model
- **Parameters**: 15,401
- **Architecture**: 2-layer GRU (64, 32 units)
- **Best for**: Efficiency, real-time applications

### Dense Model
- **Parameters**: 161,921
- **Architecture**: 3-layer Dense (256, 128, 64 units)
- **Best for**: Baseline comparison, feature interactions

## 🎯 Results

| Model | MAE | RMSE | R² | Training Time |
|-------|-----|------|----|--------------| 
| LSTM  | 0.2847 | 0.3361 | 0.847 | ~30 epochs |
| GRU   | 0.2952 | 0.3547 | 0.832 | ~25 epochs |
| Dense | 0.3164 | 0.3892 | 0.798 | ~20 epochs |

## 📚 Documentation

- [📖 Main Documentation](docs/README.md) - Comprehensive project documentation
- [📊 Project Report](docs/REPORT.md) - Detailed analysis and results
- [❓ Q&A Section](docs/QA.md) - Common questions and answers
- [📓 Analysis Notebook](notebooks/powercast_analysis.ipynb) - Interactive analysis

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

---

⭐ **Star this repository if you found it helpful!**

*PowerCast - Powering the future with intelligent forecasting* ⚡
