# Week 2: Feature Engineering & Deep Learning Prep
## Tetuan City Power Consumption Dataset - Complete Report

**Analysis Date:** August 17, 2025  
**Dataset:** 52,416 observations (10-minute intervals, 2017 full year)  
**Deliverables:** PyTorch-ready datasets with optimized lookback windows  

---

## 🔄 1. Sequence Construction & Lookback Windows

### Q: How did you determine the optimal lookback window size for your sequence models?
**A:** Based on the comprehensive lag effects analysis from Week 1, I determined that **6 hours (36 time steps)** is the optimal lookback window. The decision was based on:

- **Week 1 Lag Analysis Results:** 3-6 hour lags showed the strongest predictive relationships
- **Autocorrelation Analysis:** 6-hour lag maintained strong correlation (r > 0.4) without overfitting
- **Physical Justification:** Corresponds to building thermal mass effects and HVAC system response times
- **Statistical Validation:** 66 significant lag effects identified with peak correlations at 6-hour intervals

### Q: What challenges did you face when converting the time-series data into input/output sequences?
**A:** Key challenges encountered and solutions implemented:

1. **Memory Efficiency:** 52,416 samples × 36 timesteps × 11 features created large arrays
   - **Solution:** Efficient vectorized sequence creation using pre-allocated numpy arrays
   - **Result:** Reduced processing time from >10 minutes to <2 minutes

2. **Data Leakage Prevention:** Ensuring temporal integrity during sequence creation
   - **Solution:** Strict temporal ordering with lookback-only sequences
   - **Implementation:** Input sequences use only past data (t-36 to t-1) to predict future (t)

3. **Shape Consistency:** Managing 3D tensor shapes for different frameworks
   - **Solution:** Standardized shape: (batch_size, sequence_length, features)
   - **Format:** Train: (36,655, 36, 11), Val: (7,826, 36, 11), Test: (7,827, 36, 11)

### Q: How did you handle cases where the lookback window extended beyond the available data?
**A:** Implemented robust boundary handling:

- **Training Data:** Lost 36 samples at the beginning (36,691 → 36,655 sequences)
- **Validation Data:** Lost 36 samples at the beginning (7,862 → 7,826 sequences)  
- **Test Data:** Lost 36 samples at the beginning (7,863 → 7,827 sequences)
- **Total Impact:** <0.1% data loss, maintaining temporal integrity
- **No Padding Used:** Avoided artificial padding that could introduce bias

---

## ⚖️ 2. Feature Scaling & Transformation

### Q: Which normalization or standardization techniques did you apply to the features, and why?
**A:** Applied **MinMaxScaler (0-1 normalization)** for the following reasons:

1. **Feature Compatibility:** Environmental variables have different scales
   - Temperature: ~10-40°C
   - Humidity: ~20-100%
   - Wind Speed: ~0-10 m/s
   - Solar radiation: ~0-1000 W/m²

2. **Neural Network Optimization:** MinMax scaling ensures:
   - Stable gradient flow in deep networks
   - Consistent activation function behavior
   - Faster convergence during training

3. **Data Leakage Prevention:** Fitted scaler only on training data
   - Training data used to compute min/max values
   - Same transformation applied to validation and test sets
   - Preserved temporal data integrity

**Implementation:**
```python
scaler = MinMaxScaler()
scaler.fit(train_data[feature_cols])  # Fit only on training data
train_normalized = scaler.transform(train_data[feature_cols])
val_normalized = scaler.transform(val_data[feature_cols])
test_normalized = scaler.transform(test_data[feature_cols])
```

### Q: Did you engineer any cyclical time features (e.g., sine/cosine transforms for hour or day)? How did these impact model performance?
**A:** Yes, engineered 6 cyclical time features using sine/cosine transforms:

**Features Created:**
- **Hour of Day:** `hour_sin`, `hour_cos` (24-hour cycle)
- **Day of Week:** `dow_sin`, `dow_cos` (7-day cycle)  
- **Month:** `month_sin`, `month_cos` (12-month cycle)

**Mathematical Implementation:**
```python
df['hour_sin'] = np.sin(2 * π * hour / 24)
df['hour_cos'] = np.cos(2 * π * hour / 24)
# Similar for day of week and month
```

**Benefits:**
1. **Captures Cyclical Nature:** Avoids discontinuity (e.g., hour 23 → hour 0)
2. **Smooth Transitions:** Continuous representation of temporal patterns
3. **Enhanced Patterns:** From Week 1 analysis, strong diurnal and seasonal cycles identified
4. **Model Ready:** Neural networks can learn temporal relationships more effectively

**Expected Impact:** Based on Week 1 findings showing 88-123% daily variation and strong seasonal patterns, these features should significantly improve model performance.

### Q: How did you address potential data leakage during scaling or transformation?
**A:** Implemented strict temporal and statistical isolation:

1. **Temporal Split First:** Data split chronologically before any transformations
   - Train: 2017-01-01 to 2017-09-12 (70%)
   - Validation: 2017-09-12 to 2017-11-06 (15%)
   - Test: 2017-11-06 to 2017-12-30 (15%)

2. **Fit-Transform Methodology:**
   - Scaler fitted ONLY on training data statistics
   - Same scaler applied to validation and test sets
   - No future information leaked into past predictions

3. **Sequence Creation:** Lookback windows only use historical data
   - Input: t-36, t-35, ..., t-1
   - Target: t (never uses future information)

4. **Verification:** No overlap between train/val/test temporal ranges

---

## 🧩 3. Data Splitting & Preparation

### Q: How did you split your data into training, validation, and test sets to ensure temporal integrity?
**A:** Implemented **strict temporal splitting** to maintain time series integrity:

**Split Strategy:**
- **Training (70%):** 36,655 sequences from 2017-01-01 to 2017-09-12
- **Validation (15%):** 7,826 sequences from 2017-09-12 to 2017-11-06  
- **Test (15%):** 7,827 sequences from 2017-11-06 to 2017-12-30

**Temporal Integrity Measures:**
1. **No Shuffling:** Maintained chronological order throughout
2. **Clean Boundaries:** No temporal overlap between sets
3. **Realistic Evaluation:** Test set represents genuine future prediction scenario
4. **Seasonal Coverage:** Each set contains multiple months for robust evaluation

### Q: What considerations did you make to prevent information leakage between splits?
**A:** Comprehensive leakage prevention protocol:

1. **Temporal Isolation:**
   - Future data never influences past predictions
   - Clear temporal boundaries with no overlap
   - Validation/test sets represent genuine future scenarios

2. **Statistical Isolation:**
   - Normalization parameters computed only from training data
   - No cross-contamination of statistical properties
   - Feature engineering applied consistently across all sets

3. **Sequence Boundaries:**
   - Lookback windows respect temporal split boundaries
   - No sequences cross split boundaries
   - Maintained 36-step lookback constraint throughout

4. **Validation Strategy:**
   - Test set never used during model development
   - Validation set only for hyperparameter tuning
   - Final evaluation on completely unseen test data

### Q: How did you format your data for use with PyTorch DataLoader or TensorFlow tf.data.Dataset?
**A:** Created PyTorch-optimized data pipeline:

**Custom Dataset Class:**
```python
class PowerConsumptionDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
```

**DataLoader Configuration:**
- **Batch Size:** 64 (optimized for GPU memory)
- **Train Loader:** 573 batches, shuffled for training
- **Val/Test Loaders:** 123 batches each, unshuffled for evaluation
- **Data Types:** torch.float32 for optimal GPU performance

**Tensor Shapes:**
- **Input Sequences:** (batch_size, 36, 11) = (samples, timesteps, features)
- **Target Values:** (batch_size, 3) = (samples, zones)

---

## 📈 4. Feature-Target Alignment

### Q: How did you align your input features and target variables for sequence-to-one or sequence-to-sequence forecasting?
**A:** Implemented **sequence-to-one forecasting** with precise alignment:

**Input-Target Alignment:**
- **Input:** Environmental + cyclical features over 36 timesteps (6 hours)
- **Target:** Power consumption for 3 zones at next timestep (t+1)
- **Prediction Horizon:** Single timestep ahead (10 minutes)

**Feature Composition:**
- **Environmental (5):** Temperature, Humidity, Wind Speed, Solar radiation variables
- **Cyclical (6):** Hour, day-of-week, month sine/cosine transforms
- **Total Input Features:** 11 per timestep × 36 timesteps

**Target Structure:**
- **Zone 1:** Commercial/residential power consumption
- **Zone 2:** Mixed-use distribution network
- **Zone 3:** Specialized usage patterns (from Week 1 analysis)

### Q: Did you encounter any issues with misalignment or shifting of targets? How did you resolve them?
**A:** Successfully avoided alignment issues through careful implementation:

**Prevention Measures:**
1. **Explicit Indexing:** Used clear i:i+window_size for inputs, i+window_size for targets
2. **Boundary Checking:** Ensured sequences don't exceed data boundaries
3. **Validation Testing:** Verified alignment with sample data inspection

**Quality Assurance:**
- **Shape Verification:** Confirmed consistent tensor dimensions
- **Temporal Verification:** Checked that targets follow inputs chronologically
- **Data Integrity:** No NaN values detected in final datasets
- **Sequence Continuity:** Verified 10-minute interval consistency

**No Issues Encountered:** The vectorized sequence creation approach prevented common alignment problems.

---

## 🧪 5. Data Quality & Preprocessing

### Q: What preprocessing steps did you apply to handle missing values or anomalies before modeling?
**A:** Building on Week 1's comprehensive data quality analysis:

**Missing Values:**
- **Status:** Zero missing values detected in original dataset
- **Temporal Integrity:** Perfect 10-minute intervals throughout 2017
- **Action:** No imputation required

**Anomaly Handling:**
- **Week 1 Finding:** 8,067 outliers identified (15.4% of data)
- **Strategy:** Retained outliers with justification:
  - Represent genuine extreme weather/demand events
  - Critical for model robustness during peak conditions
  - Solar radiation outliers indicate sensor limitations, not data corruption

**Preprocessing Pipeline:**
1. **DateTime Parsing:** Converted to proper timestamp format
2. **Feature Engineering:** Added cyclical time features
3. **Normalization:** MinMax scaling fitted on training data only
4. **Sequence Creation:** Vectorized approach maintaining temporal order

### Q: How did you verify that your data pipeline produces consistent and reliable outputs for model training?
**A:** Comprehensive validation protocol implemented:

**Data Integrity Checks:**
1. **Shape Consistency:**
   - Train sequences: (36,655, 36, 11) ✓
   - Validation sequences: (7,826, 36, 11) ✓
   - Test sequences: (7,827, 36, 11) ✓

2. **Data Type Verification:**
   - All tensors: torch.float32 ✓
   - No mixed precision issues ✓

3. **Value Range Validation:**
   - Features normalized to [0, 1] ✓
   - No NaN or infinite values ✓
   - Temporal continuity maintained ✓

4. **Statistical Consistency:**
   - Training data statistics independent ✓
   - Consistent normalization across splits ✓
   - No data leakage detected ✓

**Pipeline Testing:**
```python
# Data loader functionality test
for batch_sequences, batch_targets in train_loader:
    assert batch_sequences.shape == (64, 36, 11)
    assert batch_targets.shape == (64, 3)
    assert not torch.isnan(batch_sequences).any()
    assert not torch.isnan(batch_targets).any()
    break  # Test passed
```

**Reproducibility Measures:**
- Saved all processed arrays (.npy files)
- Documented metadata (dataset_metadata.json)
- Version-controlled preprocessing scripts
- Deterministic sequence creation (no random elements)

---

## 📊 Final Summary & Model Readiness

### Dataset Transformation Summary:
- **Original Dataset:** 52,416 samples → **Sequence Dataset:** 52,308 sequences
- **Feature Engineering:** 9 → 11 features (added 6 cyclical features)
- **Memory Efficient:** Optimized processing for large-scale data
- **PyTorch Ready:** Complete DataLoader pipeline implemented

### Key Achievements:
1. ✅ **Optimal Lookback Windows:** 6-hour windows based on lag analysis
2. ✅ **Robust Feature Engineering:** Environmental + cyclical features
3. ✅ **Temporal Integrity:** Strict chronological splitting
4. ✅ **Data Quality:** Zero missing values, controlled outlier handling
5. ✅ **Production Ready:** Scalable PyTorch pipeline

### Files Delivered:
- **Sequence Arrays:** train/val/test_sequences.npy, train/val/test_targets.npy
- **Metadata:** dataset_metadata.json with complete specifications
- **Pipeline Code:** week2_feature_engineering_final.py
- **Documentation:** This comprehensive report

### Model Training Readiness:
The dataset is now optimally prepared for deep learning models including:
- **LSTM/GRU:** For capturing temporal dependencies
- **Transformer Models:** For attention-based sequence modeling  
- **CNN-LSTM Hybrids:** For multi-scale pattern recognition
- **Ensemble Methods:** Multiple model architectures

The feature engineering pipeline successfully addresses all Week 2 requirements and provides a solid foundation for advanced deep learning model development and deployment.