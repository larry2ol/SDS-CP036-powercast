# 🔴 PowerCast – Nima - Advanced Track

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

---

### 🧭 1. Time Consistency & Structure

Q: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?
A: No, there are no missing or irregular timestamps.
To verify this, I treated the DateTime column as a time series and checked for gaps. Then, I asked the program to find any instances where a 10-minute timestamp was expected but not present. The analysis found zero missing timestamps across all 52,416 records.

Q: What is the sampling frequency and are all records spaced consistently?
A: The sampling frequency is 10 minutes, and all records are spaced consistently.

Q: Did you encounter any duplicates or inconsistent `DateTime` entries?
A: No, there were no duplicate or inconsistent DateTime entries found.
To check for this, I performed two checks:
Duplicate Timestamps: I searched for any DateTime values that appeared more than once in the dataset. The result was zero, meaning every timestamp is unique.

---

### 📊 2. Temporal Trends & Seasonality

Q: What daily or weekly patterns are observable in power consumption across the three zones?
A: In all three zones, consumption typically follows a similar daily pattern. It begins to increase at 7:00 in the morning, reaching a plateau from midday to early evening. Consumption then hits a peak around 20:00 before decreasing throughout the night, reaching its lowest point by 7:00 the next morning. There is a discernible weekly pattern where consumption is typically higher during weekdays compared to weekends (Saturday and Sunday).

Q: Are there seasonal or time-of-day peaks and dips in energy usage?
A: Yes, there are clear time-of-day peaks and dips. Power consumption generally increases in the morning, peaks in the afternoon and evening, and drops significantly overnight. This pattern is consistent across all three zones.

Q: Which visualizations helped you uncover these patterns?
A: Boxplots were used to effectively visualize these patterns, showing the distribution of power consumption for each hour of the day and each day of the week.

---

### 🌦️ 3. Environmental Feature Relationships

Q: Which environmental variables (temperature, humidity, wind speed, solar radiation) correlate most with energy usage?
A: The single most influential environmental variable is Temperature. It has a strong positive correlation with energy usage across all three zones. This means that as the temperature increases, power consumption also increases significantly.

Q: Are any variables inversely correlated with demand in specific zones?
A: Yes. The most significant inversely correlated variable is Humidity. In all three zones, as humidity decreases, power consumption tends to increase.

Q: Did your analysis differ across zones? Why might that be?
A: The analysis shows that the correlation patterns are remarkably similar across all three zones. Temperature is the dominant driver in every zone, and humidity is the primary inverse variable in all of them. The strength of these correlations varies only slightly from one zone to another.

---

### 🌀 4. Lag Effects & Time Dependency

Q: Did you observe any lagged effects where past weather conditions predict current power usage?
A: Yes, significant lagged effects were observed for multiple weather features, not just temperature.

Q: How did you analyze lag (e.g., shifting features, plotting lag correlation)?
A: The analysis was done by programmatically shifting each environmental feature ('Temperature', 'Humidity', 'general_diffuse_flows') by different time intervals (from 1 to 24 hours) and calculating the correlation coefficient with 'Zone_1_Power_Consumption' at each step.


Q: What lag intervals appeared most relevant and why?
A: Temperature & Humidity: The correlation remains high for the first few hours and then gradually decreases. This makes physical sense, as buildings retain heat and humidity. A 24-hour lag is also highly relevant for both, indicating a strong daily cycle.
General Diffuse Flows (Solar Radiation): This feature shows a different pattern. The correlation peaks a few hours after the event, which strongly supports the idea of thermal mass. The sun heats buildings during the day, and this leads to increased power usage for cooling a few hours later. The 24-hour lag is also prominent here.

---

### ⚠️ 5. Data Quality & Sensor Anomalies

Q: Did you detect any outliers in the weather or consumption readings?
A: Yes, outliers were detected, but they appear to be legitimate extreme values rather than data errors. The most notable outliers are the power consumption readings during the hottest summer days, which are significantly higher than the annual average. These are not sensor malfunctions but represent real, critical peak demand events.

Q: How did you identify and treat these anomalies?
A: The outliers were identified visually and statistically, and the recommended treatment is to retain them. The most effective way to identify these outliers is with a Box Plot. 

Q: What might be the impact of retaining or removing them in your model?
A: The decision to retain or remove these outliers has a significant impact on the model's performance and reliability. Retaining them can make some models (like simple linear regression) less accurate on average because the extreme values can skew the results. However, more advanced models (like Gradient Boosting or Random Forest) handle these outliers effectively. By removing them, the model would be probably naive and unreliable. It would be completely blind to peak demand scenarios. When a real heatwave occurs, the model would severely underestimate the required power, which could lead to resource shortfalls and blackouts.

---

## 🛠️ Week 2: Feature Engineering & Deep Learning Preparation

### 🔄 1. Sequence Construction & Lookback Windows

Q: How did you determine the optimal lookback window size for your sequence models?  
The optimal lookback window size is a critical hyperparameter that often requires experimentation. For this initial setup, a lookback window of 144 steps was chosen. Since the data is sampled every 10 minutes, 144 steps correspond to exactly 24 hours. This is a logical starting point because the EDA in previous weeks revealed strong daily seasonality; the power consumption at a certain time is highly correlated with its value at the same time the previous day.

Q: What challenges did you face when converting the time-series data into input/output sequences?  
The main challenge is conceptual: ensuring that each input sequence X correctly maps to its corresponding target y. This requires careful indexing. The provided create_sequences function iterates through the dataset and, for each position i, it creates an input sequence from i to i + lookback_window and defines the target as the value at position i + lookback_window. This ensures perfect alignment for a sequence-to-one forecasting task.

Q: How did you handle cases where the lookback window extended beyond the available data?  
This was handled implicitly by the loop in the create_sequences function. The loop runs from i = 0 up to len(input_data) - lookback_window. This automatically stops the sequence creation process at the last possible point where a full lookback window and its corresponding target can be formed, thus preventing any IndexError.

---

### ⚖️ 2. Feature Scaling & Transformation

Q: Which normalization or standardization techniques did you apply to the features, and why?  
Min-Max Scaling (MinMaxScaler) was applied to scale all features to a range of [0, 1]. This technique was chosen because neural networks, particularly RNNs, are sensitive to the scale of input data. Normalization ensures that all features contribute equally to the model's learning process and helps the gradient descent algorithm to converge faster and more stably.

Q: Did you engineer any cyclical time features (e.g., sine/cosine transforms for hour or day)? How did these impact model performance?  
Yes, cyclical features were engineered for hour and dayofweek using sine and cosine transformations. This is crucial for deep learning models. A feature like hour (ranging from 0 to 23) doesn't naturally convey that hour 23 is "close" to hour 0. By converting it into sin(hour) and cos(hour), we represent the time on a 2D circle, so the model can understand the cyclical proximity of these time points. This typically leads to a significant improvement in model performance as it allows the network to properly learn daily and weekly patterns.

Q: How did you address potential data leakage during scaling or transformation?  
Data leakage was prevented by adhering to a strict rule: the scaler was fit only on the training data. The parameters (min and max values for each feature) learned from the training data were then used to transform the validation and test sets. This simulates a real-world scenario where the model has no knowledge of the distribution of future (validation/test) data.

---

### 🧩 3. Data Splitting & Preparation

Q: How did you split your data into training, validation, and test sets to ensure temporal integrity?  
 A chronological split was performed. The data, being ordered by time, was split into three contiguous blocks: the first 70% for training, the next 15% for validation, and the final 15% for testing. This is the only correct way to split time-series data to ensure the model is always trained on the past and validated/tested on the future.

Q: What considerations did you make to prevent information leakage between splits?  
The primary consideration was the chronological split itself. Additionally, all preprocessing steps (scaling, sequence creation) were performed independently on each set after the split (with the scaler being fit only on the training set). This guarantees that no information from the validation or test sets leaks into the training process.

Q: How did you format your data for use with PyTorch DataLoader or TensorFlow tf.data.Dataset?  
The data was formatted for PyTorch DataLoaders. The process was as follows:

The NumPy arrays (X_train, y_train, etc.) from the sequence creation step were converted into torch.Tensor objects.

A TensorDataset was created for each split, pairing the input feature tensors with their corresponding target tensors.

Finally, DataLoader objects were created from these datasets. The DataLoader for training was configured to shuffle the data (shuffle=True) to improve model generalization, while the validation and test loaders have shuffling disabled (shuffle=False) to evaluate performance on the sequences in their natural, chronological order.

---

### 📈 4. Feature-Target Alignment

Q: How did you align your input features and target variables for sequence-to-one or sequence-to-sequence forecasting?  
For this sequence-to-one setup, the alignment is handled inside the create_sequences function. For each input sequence of length lookback_window (e.g., from time t-144 to t-1), the target variable is defined as the power consumption value at the immediately following timestep (t). This ensures a direct and correct mapping between a history of features and the value to be predicted.

Q: Did you encounter any issues with misalignment or shifting of targets? How did you resolve them?  
No issues were encountered because the process was designed to prevent them. By creating the sequences and their corresponding targets within the same loop and using careful indexing, misalignment is avoided by design.

---

### 🧪 5. Data Quality & Preprocessing

Q: What preprocessing steps did you apply to handle missing values or anomalies before modeling?  
The initial dataset was complete with no missing values. The only NaNs that could have appeared would be from feature engineering steps like rolling windows (which were not used in this deep learning prep). Outliers identified in Week 1 were retained, as scaling to a [0, 1] range helps to mitigate their influence on the model's weight updates.

Q: How did you verify that your data pipeline produces consistent and reliable outputs for model training?  
The pipeline's reliability was verified by:

Checking Shapes: Printing the shapes of the arrays and tensors at each step (X_train.shape, y_train.shape, etc.) confirms that the dimensions are correct for a recurrent neural network.

Batch Inspection: The final step in the script iterates through one batch of the train_loader and prints its shape. This serves as a final check that the DataLoader is correctly yielding batches of data in the expected format, ready to be fed into a model.

---

## ✅ Week 3: Neural Network Design & Baseline Training

---

### 🧠 1. Model Architecture & Design

Q: Which neural network architecture(s) did you choose for baseline forecasting (e.g., LSTM, GRU, TCN), and what motivated your selection?  
A Long Short-Term Memory (LSTM) network was chosen as the baseline architecture. LSTMs are a type of Recurrent Neural Network (RNN) specifically designed to address the vanishing gradient problem, making them exceptionally good at learning long-range dependencies in sequential data. Given the strong daily and weekly patterns in power consumption, an LSTM is a natural choice to capture these temporal relationships. A GRU (Gated Recurrent Unit) would be another excellent choice, often providing similar performance with slightly less complexity.

Q: How did you structure your input sequences and targets for the chosen model(s)?  
The data was structured for a sequence-to-one forecasting task.

Input Sequence: Each input sample X has a shape of (144, 8), representing a lookback window of 144 time steps (24 hours) and 8 features.

Target: Each input sequence maps to a single target value y, which is the Zone_1_Power_Consumption at the time step immediately following the end of the input sequence. The model's goal is to predict this single future value based on the 24-hour history.

Q: What considerations did you make regarding the depth, number of units, and activation functions in your network?  
For a baseline model, a relatively simple architecture was chosen to avoid overfitting and establish a performance benchmark:

Depth: The model uses two stacked LSTM layers (num_layers=2). Using more than one layer allows the network to learn higher-level temporal representations.

Number of Units: Each LSTM layer has 64 hidden units (hidden_size=64). This number is a common starting point and offers a good balance between model capacity and computational cost.

Activation Functions: LSTMs internally use sigmoid and tanh activation functions within their gating mechanisms. A final Linear layer is used to map the LSTM's output to the single predicted value, without a final activation function, which is standard for regression tasks.

---

### 🏋️ 2. Training & Experimentation

Q: Which loss function and optimizer did you use for training, and why are they suitable for this task?  

Loss Function: Mean Squared Error (MSELoss) was used. It is the most common loss function for regression problems, as it calculates the average of the squared differences between the predicted and actual values. It has the benefit of penalizing larger errors more heavily.

Optimizer: The Adam optimizer was chosen. Adam is an adaptive learning rate optimization algorithm that is computationally efficient, requires little memory, and is well-suited for a wide range of problems, making it an excellent default choice.

Q: How did you incorporate regularization techniques such as Dropout or Batch Normalization, and what impact did they have?  
Dropout was incorporated directly into the LSTM layers (dropout=0.2). Dropout randomly sets a fraction of input units to zero at each update during training time, which helps prevent co-adaptation of neurons and reduces overfitting. A rate of 20% was chosen as a sensible starting point. This makes the model more robust and improves its ability to generalize to unseen data.

Q: What challenges did you encounter during training (e.g., overfitting, vanishing gradients), and how did you address them?  
The main anticipated challenge is overfitting, where the model learns the training data too well and performs poorly on the validation set. This was addressed proactively by:

Using Dropout as a regularization method.

Keeping the model architecture relatively simple (2 layers, 64 units).

Monitoring both training and validation loss after each epoch. A large divergence between the two would be a clear sign of overfitting, at which point techniques like early stopping could be implemented.

---

### 📊 3. Evaluation & Metrics

Q: Which metrics did you use to evaluate your model’s performance, and why are they appropriate for time-series forecasting?  
Mean Absolute Error (MAE): This metric gives the average absolute difference between the predicted and actual values in the original units (kW). It's easy to interpret and provides a clear sense of the average prediction error.

Root Mean Squared Error (RMSE): This metric also provides the error in the original units but penalizes larger errors more significantly. It is useful for understanding the impact of significant prediction misses.

Q: How did you use MLflow (or another tool) to track your training experiments and results?  
While MLflow was not explicitly implemented in this script for simplicity, a professional workflow would involve wrapping the training loop with it. The process would be:

Start an MLflow run: mlflow.start_run().

Log hyperparameters: Log lookback_window, hidden_size, num_layers, learning_rate, etc., using mlflow.log_param().

Log metrics: Inside the training loop, log the train_loss and val_loss for each epoch using mlflow.log_metric().

Log final metrics: After evaluation, log the final test MAE and RMSE.

Log the model: Save the trained model using mlflow.pytorch.log_model().
This creates a reproducible record of all experiments, making it easy to compare different architectures and hyperparameters.

Q: What insights did you gain from visualizing forecasted vs. actual power consumption for each zone?  
The visualization shows that the baseline LSTM model does a very good job of capturing the overall trend and daily seasonality of the power consumption. The predicted curve follows the actual curve closely, successfully predicting the daily peaks and overnight troughs. This confirms that the model has learned the fundamental patterns in the data.

---

### 🔍 4. Model Interpretation & Insights

Q: How did you interpret the learned patterns or feature importance in your neural network?  
Interpreting "feature importance" in LSTMs is more complex than with tree-based models. Instead of a direct importance score, interpretation comes from analyzing the model's behavior. The strong performance confirms the model learned the importance of the cyclical time features and the recent power consumption values provided in the lookback window. Advanced techniques like SHAP (SHapley Additive exPlanations) could be applied to get more granular insights into how the model makes predictions at specific time steps.

Q: Did you observe any systematic errors or biases in your model predictions? How did you investigate and address them?  
This was investigated by analyzing the residuals (prediction errors).

Residuals Over Time: The plot of residuals over time shows that the errors are mostly centered around zero, without long periods of consistent over- or under-prediction. This suggests there is no major systematic bias.

Distribution of Residuals: The histogram shows that the errors are roughly normally distributed around a mean of zero, which is a sign of a well-behaved model. There might be a slight tendency for a few larger negative errors, indicating the model sometimes over-predicts consumption.

Q: What trade-offs did you consider when selecting your final baseline model architecture?
The main trade-off was between model complexity and performance/training time.

A simpler model (fewer layers/units) would train faster but might not have the capacity to learn all the nuances of the data.

A more complex model might achieve higher accuracy but would take longer to train and be more prone to overfitting.
The chosen 2-layer, 64-unit LSTM architecture represents a balanced starting point, offering sufficient capacity to learn the patterns without being excessively complex for a baseline experiment.

Week 4: Model Optimization & Interpretability
🏗️ 1. Architecture Tuning & Experimentation
Q: Which architectural changes (e.g., depth, number of units, bidirectionality, dilation) did you experiment with, and why?
We experimented with a Bidirectional LSTM (BiLSTM). A standard LSTM processes the time series in chronological order (from past to present). A BiLSTM adds a second LSTM layer that processes the sequence in reverse order (from present to past). The outputs from both are then combined. This was chosen because it allows the model to capture dependencies from both past and future contexts within the lookback window, which can lead to a richer understanding of the sequence's patterns.

Q: How did you decide on the final architecture for your deep learning model?
The decision to use the BiLSTM architecture was based on its common application in time-series forecasting and its theoretical advantages. After implementing it, we observed a slight but consistent improvement in evaluation metrics (lower MAE/RMSE, higher R²) across all three zones compared to the baseline unidirectional LSTM. This performance gain, coupled with the more robust training process (see next section), validated its selection as the final architecture.

Q: What impact did these changes have on model performance and training stability?
The BiLSTM architecture resulted in a modest performance improvement. More importantly, when combined with the training strategies below, it led to a more stable and efficient training process. The model converged faster and was less prone to overfitting, as we could stop the training process automatically once performance on the validation set stopped improving.

⏸️ 2. Training Strategies & Regularization
Q: How did you apply early stopping or learning rate scheduling during training?
Both were implemented in the training loop:

Early Stopping: A custom EarlyStopping class was created to monitor the validation loss at the end of each epoch. If the validation loss did not improve for a patience of 5 consecutive epochs, the training process was halted. This prevents the model from continuing to train when it's no longer learning, saving time and preventing overfitting.

Learning Rate Scheduling: We used PyTorch's ReduceLROnPlateau scheduler. This scheduler monitors the validation loss and automatically reduces the optimizer's learning rate (by a factor of 0.5) if the loss plateaus for a patience of 3 epochs. This helps the model to fine-tune its weights and settle into a better minimum.

Q: What regularization techniques (e.g., Dropout, Batch Normalization) did you use, and how did they affect results?
We continued to use Dropout with a rate of 20% in the BiLSTM layers. As before, this is a crucial technique to prevent overfitting by randomly deactivating neurons during training, forcing the network to learn more robust features. It is a key component for achieving good generalization on the test set.

Q: How did you monitor and address overfitting or underfitting during optimization?
Overfitting was monitored and addressed through a combination of methods:

Monitoring Validation Loss: The primary method was to compare the training loss and validation loss after each epoch. A large gap where training loss continues to decrease while validation loss stagnates or increases is a clear sign of overfitting.

Early Stopping: This was the direct intervention to address overfitting. By stopping training when validation loss stops improving, we select the model at its point of peak generalization.

Dropout: This was the proactive regularization technique used to make overfitting less likely from the start.

🧠 3. Model Interpretability
Q: Which interpretability methods (e.g., SHAP, saliency maps, attention plots) did you use to understand your model’s predictions?
I used SHAP (SHapley Additive exPlanations). Specifically, the DeepExplainer from the shap library, which is designed for deep learning models. SHAP is a game-theoretic approach that assigns an "importance" value to each feature for each individual prediction, indicating how much that feature contributed to pushing the prediction higher or lower.

Q: What insights did you gain about feature importance or temporal dependencies from these methods?
The SHAP summary plots provided several key insights:

Dominant Features: Across all zones, the power consumption of the zone itself (Zone_X_Power_Consumption) was by far the most important feature. This is expected, as recent power usage is a very strong predictor of future usage (autocorrelation).

Environmental Drivers: After self-consumption, Temperature and the cyclical time features (hour_sin/hour_cos) were consistently the next most important drivers, confirming the findings from our initial EDA.

Zone-Specific Differences: While the overall patterns were similar, SHAP can reveal subtle differences. For example, Wind_Speed might show slightly higher importance in one zone compared to another, aligning with our earlier correlation analysis.

Q: How did interpretability findings influence your modeling or feature engineering decisions?
The SHAP results strongly validated our feature engineering decisions. The high importance of the cyclical time features (hour_sin/cos) confirmed that this transformation was effective and necessary. It also reinforced the decision to include all the power consumption zones as features for each model, as they clearly contain valuable predictive information.

📊 4. Error Analysis & Residuals
Q: How did you analyze residuals and error distributions across different zones?
The analysis was performed using the same two plots for each zone as in the baseline model:

Residuals Over Time: This plot shows the prediction error (Actual - Predicted) for each point in the test set. We look for patterns, such as the model consistently over- or under-predicting during certain times of the day.

Distribution of Residuals: A histogram of the errors. We look for a roughly normal distribution centered at zero, which indicates that the model's errors are random and not systematically biased.

Q: Did you identify any systematic errors or biases in your model predictions? How did you address them?
The residual plots showed that the optimized model's errors were largely centered around zero and normally distributed, indicating no major systematic bias. There were occasional larger errors, often corresponding to sudden, sharp peaks or troughs in power consumption that are inherently difficult to predict. Further improvements could involve engineering features specifically designed to capture these sudden changes, such as rolling standard deviation.

Q: What steps did you take to ensure robust evaluation and fair comparison of model performance across different configurations?
Fixed Data Splits: The training, validation, and test sets were split chronologically and remained fixed for all experiments. This ensures that we are always comparing models on the same data.

Consistent Metrics: The same set of metrics (MAE, RMSE, R²) was used to evaluate all models.

Separate Models per Zone: By training a dedicated model for each zone, we allow each model to learn the specific patterns of its target, leading to a fairer and likely more accurate evaluation than a single model trying to predict all three zones at once.

Week 5: Deployment
🟢 Easy: Streamlit Cloud
This final phase of the project involves deploying the trained forecasting models into an interactive web application using the Streamlit library.

What the Code Does: The Final_web_app_W5_Nima.py script creates a user-friendly web interface. It loads the three pre-trained Bidirectional LSTM models and the data scaler that were saved by the training script. It allows a user to upload the historical dataset, select a specific date and time, choose a zone, and receive an on-demand forecast for the power consumption in the next 10 minutes.

Prerequisites to Run:
Python and the necessary libraries must be installed (streamlit, pandas, torch, scikit-learn).
The script must be run from a terminal that is correctly configured for your Python environment (e.g., Anaconda Prompt).

How to Run the Application:
Open a terminal (like Anaconda Prompt).
Navigate to the folder containing your project files.
Run the command: "streamlit run Final_web_app_W5_Nima.py" or "python -m streamlit run Final_web_app_W5_Nima.py".
A new tab will open in your web browser with the live application.

Required Files for Operation:
The application script (Final_web_app_W5_Nima.py).
The three trained model files: model_Zone_1.pth, model_Zone_2.pth, model_Zone_3.pth.
The saved scaler file: scaler.pkl.
The user must provide the Tetuan City power consumption.csv file through the application's upload interface.