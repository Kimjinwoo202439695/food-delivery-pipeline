food-delivery-pipeline

This project introduces an automated machine learning pipeline (AutoML Framework) designed to predict delivery durations (Regression) and proactively classify delivery delay tiers (Classification) utilizing food delivery platform datasets.

Departing from rigid, single-use scripts tied to a specific dataset, this pipeline is engineered as a highly reusable and scalable open-source library. By utilizing a parameter-driven architecture, users can seamlessly inject entirely new datasets and feature lists, allowing the internal algorithms to dynamically parse, preprocess, and optimize model performance without modifying the core source code.

🛠 Key Architectural Features (Contributions)

1. Parameter-Driven Universal Interface

Eliminates hard-coded column dependencies. The top-level single function (run_delivery_full_pipeline) dynamically accepts file paths and feature specifications via functional arguments.

Ensures seamless adaptability to alternative logistics or courier datasets (e.g., parcel delivery, drone logistics) without modifying the internal source code.

2. Dynamic Imputation & Categorical Variable Processing

Automatically inspects user-specified feature lists to perform dynamic data imputation: categorical variables are filled via mode (most frequent value), while numerical features are imputed using the column mean.

Automatically handles variable dimensions in text categories, dynamically expanding via One-Hot Encoding (dummy variables) to prepare data for robust ML training.

3. Business-Logic Driven Target Engineering (Classification)

Instead of using a fixed, arbitrary delay threshold, the pipeline dynamically categorizes distances into safe operational intervals (0-20km) based on the project guidelines.

Computes the median delivery time for each distance group to establish a baseline expected duration, calculates the mathematical residual against the actual time, and automatically engineer a 3-tier target variable: Normal (0) / Mild Delay (1) / Severe Delay (2) for Random Forest training.

4. Statistical Significance-Based Feature Selection (Regression)

Features a mathematically formulated p-value calculation engine driven by linear algebra operations and probability density functions (via SciPy), completely bypassing heavy external statistical packages.

Systematically screens and drops non-significant noise variables where p-value >= 0.05. This autonomous feature selection alleviates multicollinearity, ensuring that the final Multiple Linear Regression model is trained strictly on statistically significant predictors to maximize explainability ($R^2$) and model stability.

5. Multi-Loop AutoML Optimization Engine

Performs an exhaustive grid search across a combinatorial space of 3 distinct scalers (Standard, MinMax, Robust) and Random Forest hyperparameters (max_depth: [5, 10, 15]), evaluating a total of 9 pipeline configurations.

Utilizes Stratified 5-Fold Cross Validation to enforce rigorous evaluation benchmarks. Upon execution, it ranks all configurations and outputs a clean, comprehensive report showcasing the Top 5 Pipelines along with the Absolute Best Combination directly to the console.

🚀 Usage

Built entirely on Scikit-learn and SciPy execution engines, this pipeline guarantees 100% stable execution across all Python environments (including Google Colab) free of package version conflicts. To deploy this pipeline on an alternative dataset, simply modify the following 5 parameter configuration lines inside the if __name__ == "__main__": block at the bottom of delivery_pipeline.py:

if __name__ == "__main__":
    # Adjust these 5 parameters to instantly plug in and analyze a completely new dataset:
    TARGET_FILE = 'Food_Delivery_Times.csv'         # Path to your dataset file
    TIME_COL = 'Delivery_Time_min'                  # Regression target variable name
    DIST_COL = 'Distance_km'                        # Distance baseline variable for residuals
    CAT_FEATURES = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'] # List of categorical features
    NUM_FEATURES = ['Courier_Experience_yrs']       # List of numerical features


📊 Evaluation Metrics

Classification: Stratified 5-Fold Cross Validation (Accuracy)

Regression: OLS t-statistic p-value Screening, $R^2$ Score, MAE, MAPE
