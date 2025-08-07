import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import warnings
import random
warnings.filterwarnings('ignore')

# Load and prepare data (assuming your existing code above)
data_path = "./Energy/"
train_csv = pd.read_csv(data_path + "train.csv", index_col=0)
test_csv = pd.read_csv(data_path + "test.csv", index_col=0)
building_csv = pd.read_csv(data_path + "building_info.csv")

SEED = 42
print(f"[Current SEED : {SEED}]")
random.seed(SEED)
np.random.seed(SEED)

# Clean building data
building_col = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for i in building_col:
    building_csv[i] = building_csv[i].replace('-', np.nan).astype(float)
building_csv = building_csv.fillna(0)

# Merge datasets
train = pd.merge(train_csv, building_csv, on='건물번호', how='left')
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

# Buildings with zero solar radiation values
zero_bnos = [9, 10, 24, 46, 77, 80, 87, 93, 94, 95, 98]

def create_time_features(df):
    """Create time-based features from datetime column"""
    df = df.copy()
    df['일시'] = pd.to_datetime(df['일시'])
    
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['quarter'] = df['일시'].dt.quarter
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

def impute_solar_radiation_knn_lgb(train_df, zero_building_numbers):
    """
    Impute solar radiation values using KNN + LightGBM approach
    
    Args:
        train_df: Training dataframe
        zero_building_numbers: List of building numbers with zero solar radiation
    
    Returns:
        DataFrame with imputed solar radiation values
    """
    
    # Create a copy to work with
    df = train_df.copy()
    
    # Add time features
    df = create_time_features(df)
    
    # Separate data into buildings with valid solar radiation and those needing imputation
    valid_solar_mask = ~df['건물번호'].isin(zero_building_numbers)
    valid_solar_data = df[valid_solar_mask].copy()
    zero_solar_data = df[~valid_solar_mask].copy()
    
    print(f"Buildings with valid solar radiation: {len(valid_solar_data)}")
    print(f"Buildings needing imputation: {len(zero_solar_data)}")
    
    # Features for prediction (excluding target and identifier columns)
    feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
                   '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
                   'month', 'day', 'hour', 'dayofweek', 'quarter',
                   'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 
                   'dayofweek_sin', 'dayofweek_cos']
    
    # Add building type encoding
    building_type_encoded = pd.get_dummies(df['건물유형'], prefix='building_type')
    df = pd.concat([df, building_type_encoded], axis=1)
    feature_cols.extend(building_type_encoded.columns.tolist())
    
    # Update valid and zero solar data with encoded features
    valid_solar_data = df[valid_solar_mask].copy()
    zero_solar_data = df[~valid_solar_mask].copy()
    
    # Step 1: KNN Imputation
    print("Step 1: KNN Imputation...")
    
    # Prepare data for KNN
    X_valid = valid_solar_data[feature_cols].fillna(0)
    y_valid = valid_solar_data['일사(MJ/m2)']
    X_zero = zero_solar_data[feature_cols].fillna(0)
    
    # Scale features for KNN
    scaler = StandardScaler()
    X_valid_scaled = scaler.fit_transform(X_valid)
    X_zero_scaled = scaler.transform(X_zero)
    
    # Apply KNN
    knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
    knn.fit(X_valid_scaled, y_valid)
    knn_predictions = knn.predict(X_zero_scaled)
    
    # Step 2: LightGBM Refinement
    print("Step 2: LightGBM Refinement...")
    
    # Create refined dataset: valid data + KNN imputed data
    zero_solar_data_knn = zero_solar_data.copy()
    zero_solar_data_knn['일사(MJ/m2)'] = knn_predictions
    
    # Combine valid and KNN-imputed data for LightGBM training
    combined_data = pd.concat([valid_solar_data, zero_solar_data_knn], axis=0)
    
    # Prepare LightGBM data
    X_lgb = combined_data[feature_cols].fillna(0)
    y_lgb = combined_data['일사(MJ/m2)']
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_lgb, y_lgb, test_size=0.2, random_state=SEED, stratify=combined_data['건물번호']
    )
    
    # LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': SEED
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    # Step 3: Final predictions for zero solar buildings
    print("Step 3: Generating final predictions...")
    
    X_zero_final = zero_solar_data[feature_cols].fillna(0)
    final_predictions = model.predict(X_zero_final, num_iteration=model.best_iteration)
    
    # Ensure non-negative predictions (clip negative values to 0)
    final_predictions = np.maximum(final_predictions, 0)
    
    # Set solar radiation to 0 based on sunshine hours (일조)
    # Only allow solar radiation when there's sunshine ± 1 hour
    zero_solar_data_processed = zero_solar_data.copy().reset_index(drop=True)
    
    # For each row, check if sunshine hours > 0 within ±1 hour window
    datetime_col = pd.to_datetime(zero_solar_data_processed['일시'])
    
    # Create mask for allowing solar radiation
    solar_allowed_mask = np.zeros(len(zero_solar_data_processed), dtype=bool)
    
    # Group by building and date to check sunshine patterns
    zero_solar_data_processed['date'] = datetime_col.dt.date
    zero_solar_data_processed['hour'] = datetime_col.dt.hour
    
    for building_no in zero_solar_data_processed['건물번호'].unique():
        building_mask = zero_solar_data_processed['건물번호'] == building_no
        building_data = zero_solar_data_processed[building_mask].copy()
        
        for date in building_data['date'].unique():
            date_mask = building_data['date'] == date
            daily_data = building_data[date_mask].copy().sort_values('hour')
            
            # Find hours with sunshine > 0 from the original train data
            # Get corresponding data from original train for this building and date
            train_building_date = train_df[
                (train_df['건물번호'] == building_no) & 
                (pd.to_datetime(train_df['일시']).dt.date == date)
            ].copy()
            
            if len(train_building_date) > 0:
                train_building_date['hour'] = pd.to_datetime(train_building_date['일시']).dt.hour
                sunshine_hours = train_building_date[train_building_date['일조(hr)'] > 0]['hour'].values
                
                if len(sunshine_hours) > 0:
                    # Allow solar radiation for sunshine hours ± 1 hour
                    min_hour = max(0, sunshine_hours.min() - 2)
                    max_hour = min(23, sunshine_hours.max() + 2)
                    
                    # Update mask for this building and date using iloc positions
                    daily_positions = np.where(building_mask & (zero_solar_data_processed['date'] == date))[0]
                    hour_condition = (zero_solar_data_processed.loc[daily_positions, 'hour'] >= min_hour) & \
                                   (zero_solar_data_processed.loc[daily_positions, 'hour'] <= max_hour)
                    
                    # Get the positions where hour_condition is True
                    valid_positions = daily_positions[hour_condition.values]
                    solar_allowed_mask[valid_positions] = True
    
    # Apply the sunshine-based mask
    final_predictions[~solar_allowed_mask] = 0
    
    print(f"Set {(~solar_allowed_mask).sum()} records to 0 (outside sunshine ±1 hour window)")
    print(f"Kept {solar_allowed_mask.sum()} records with potential solar radiation")
    
    # Clean up temporary columns
    if 'date' in zero_solar_data_processed.columns:
        zero_solar_data_processed = zero_solar_data_processed.drop(['date', 'hour'], axis=1)
    
    # Update the original dataframe
    result_df = train_df.copy()
    zero_mask = result_df['건물번호'].isin(zero_building_numbers)
    result_df.loc[zero_mask, '일사(MJ/m2)'] = final_predictions
    
    # Print statistics
    print(f"\nImputation Results:")
    print(f"Original zero values: {len(zero_solar_data)}")
    print(f"KNN RMSE on validation: {np.sqrt(np.mean((knn.predict(scaler.transform(X_val[X_val.index.isin(valid_solar_data.index)])) - y_val[y_val.index.isin(valid_solar_data.index)])**2)):.4f}")
    print(f"LightGBM best score: {model.best_score['valid_0']['rmse']:.4f}")
    print(f"Final predictions - Min: {final_predictions.min():.4f}, Max: {final_predictions.max():.4f}, Mean: {final_predictions.mean():.4f}")
    print(f"Non-zero predictions: {(final_predictions > 0).sum()}/{len(final_predictions)}")
    
    return result_df

# Apply the imputation
print("Starting solar radiation imputation...")
train_imputed = impute_solar_radiation_knn_lgb(train, zero_bnos)

# Verify the imputation
print(f"\nVerification:")
print(f"Original train shape: {train.shape}")
print(f"Imputed train shape: {train_imputed.shape}")

# Check if zero values are replaced
for bno in zero_bnos:
    original_zero_count = (train[train['건물번호'] == bno]['일사(MJ/m2)'] == 0).sum()
    imputed_zero_count = (train_imputed[train_imputed['건물번호'] == bno]['일사(MJ/m2)'] == 0).sum()
    print(f"Building {bno}: Original zeros: {original_zero_count}, After imputation: {imputed_zero_count}")

# Save the imputed training data
train_imputed.to_csv(f"./Energy/_final/train_test/{SEED}train_imputed.csv")
print(f"\nImputed training data saved to './Energy/_final/train_test/{SEED}train_imputed.csv'")