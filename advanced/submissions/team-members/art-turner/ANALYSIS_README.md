# Model Analysis Suite

This suite provides comprehensive interpretability and error analysis tools for the power forecasting models.

## Quick Start

### Run Complete Analysis
```bash
python run_full_analysis.py --model_path best_model.pth
```

This will generate:
- Interpretability analysis (feature importance, temporal patterns, saliency maps)
- Error pattern analysis (residual distributions, temporal errors, clustering)
- Comprehensive markdown report with insights and recommendations

### Individual Analysis Components

#### 1. Interpretability Analysis
```python
from interpretability_analysis import run_interpretability_analysis

results = run_interpretability_analysis(
    model_path='best_model.pth',
    output_dir='interpretability_results'
)
```

**Generates:**
- `temporal_importance.png` - Which timesteps matter most for each zone
- `feature_importance.png` - Which input features are most influential
- `saliency_heatmap.png` - Feature × timestep importance heatmap
- `interpretability_results.json` - Numerical results

**Key Insights:**
- Which past timesteps are most predictive
- Most important environmental vs cyclical features
- Zone-specific feature dependencies

#### 2. Error Pattern Analysis
```python
from error_pattern_analysis import run_comprehensive_error_analysis

results = run_comprehensive_error_analysis(
    model_path='best_model.pth',
    output_dir='error_analysis_results'
)
```

**Generates:**
- `temporal_error_patterns.png` - How errors vary over time
- `magnitude_error_analysis.png` - Error vs prediction magnitude
- `residual_distributions.png` - Statistical properties of errors
- `comprehensive_error_analysis.json` - Detailed statistics

**Key Insights:**
- Model bias detection (systematic over/under-prediction)
- Heteroscedasticity (errors varying with magnitude)
- Non-normal residuals indicating model limitations
- Temporal patterns in prediction quality

## Analysis Components Explained

### Interpretability Methods

#### Gradient Saliency
- Computes gradients of outputs w.r.t. inputs
- Shows which features/timesteps most influence predictions
- Separate analysis for each output zone

#### Temporal Importance
- Identifies critical timesteps in the lookback window
- Reveals model's temporal dependencies
- Helps optimize sequence length

#### Feature Importance
- Rankings input features by predictive power
- Separate rankings for each zone
- Guides feature engineering efforts

#### SHAP Analysis (if available)
- Model-agnostic explanation method
- Requires `pip install shap`
- Provides consistent feature attributions

### Error Analysis Methods

#### Residual Distribution Analysis
- Tests for normality (Shapiro-Wilk, Jarque-Bera)
- Detects skewness and heavy tails
- Identifies outliers using IQR method

#### Temporal Error Patterns
- Rolling window error statistics
- Identifies periods of high/low accuracy
- Reveals seasonal or systematic patterns

#### Magnitude-Based Errors
- Analyzes errors vs prediction magnitude
- Detects heteroscedasticity
- Guides robust modeling approaches

#### Conditional Error Analysis
- Errors conditioned on input features
- Identifies problematic operating conditions
- Reveals model blind spots

#### Error Clustering
- Groups samples by error patterns
- Uses PCA + K-means clustering
- Identifies systematic failure modes

## Usage Examples

### After Week 4 Optimization
```bash
# Analyze best model from optimization
python run_full_analysis.py --model_path "optimization_results/best_ensemble_model.pth"
```

### Quick Error Check
```bash
# Skip interpretability for faster analysis
python run_full_analysis.py --model_path model.pth --skip_interpretability
```

### Comparing Models
```bash
# Run for multiple models
for model in lstm_best.pth gru_best.pth tcn_best.pth; do
    python run_full_analysis.py --model_path $model --output_dir "analysis_$model"
done
```

### In Jupyter Notebook
```python
import torch
from interpretability_analysis import ModelInterpreter
from error_pattern_analysis import AdvancedErrorAnalyzer

# Load your model
model = torch.load('best_model.pth')

# Create interpreter
interpreter = ModelInterpreter(model, feature_scaler, target_scaler, 
                             feature_names, target_names)

# Quick feature importance
feature_importance = interpreter.feature_importance_analysis(test_sequences)
interpreter.plot_feature_importance(feature_importance)
```

## Output Structure

```
analysis_results/
├── comprehensive_analysis_report.md     # Main insights and recommendations
├── interpretability_results/
│   ├── temporal_importance.png
│   ├── feature_importance.png
│   ├── saliency_heatmap.png
│   └── interpretability_results.json
└── error_analysis_results/
    ├── temporal_error_patterns.png
    ├── magnitude_error_analysis.png
    ├── residual_distributions.png
    └── comprehensive_error_analysis.json
```

## Requirements

### Core Requirements
- torch
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

### Optional Requirements
- `shap` - for SHAP analysis (`pip install shap`)

## Interpreting Results

### Good Model Indicators
- **Low bias**: Mean residuals near 0
- **Normal residuals**: Shapiro p-value > 0.05
- **Homoscedastic**: Constant error variance across magnitudes
- **Reasonable feature importance**: Environmental features dominant

### Warning Signs
- **High bias**: Systematic over/under-prediction
- **Non-normal residuals**: Heavy tails, skewness
- **Heteroscedasticity**: Error variance increases with magnitude
- **Temporal patterns**: Systematic errors at certain times

### Common Issues & Solutions

#### High Bias
- **Problem**: Mean residual >> 0
- **Solution**: Add bias correction, retrain with balanced sampling

#### Non-normal Residuals
- **Problem**: Heavy tails, outliers
- **Solution**: Robust loss functions (Huber, quantile), outlier detection

#### Heteroscedasticity
- **Problem**: Error variance increases with magnitude
- **Solution**: Logarithmic targets, robust regression

#### Poor Feature Importance
- **Problem**: Cyclical features dominating environmental ones
- **Solution**: Feature scaling review, domain knowledge integration

## Integration with MLflow

Results can be logged to MLflow for experiment tracking:

```python
import mlflow

with mlflow.start_run():
    # Run analysis
    results = run_interpretability_analysis(model_path)
    
    # Log key metrics
    for zone, importance in results['feature_importance'].items():
        top_feature = max(importance.items(), key=lambda x: x[1])
        mlflow.log_metric(f"{zone}_top_feature_importance", top_feature[1])
    
    # Log plots as artifacts
    mlflow.log_artifacts("interpretability_results")
```

This analysis suite provides the foundation for understanding model behavior, diagnosing issues, and guiding improvements in your power forecasting pipeline.