import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import warnings
import random
warnings.filterwarnings('ignore')

# Load and prepare data
data_path = "./Energy/"
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

# Merge test data with building info
test = pd.merge(test_csv, building_csv, on='건물번호', how='left')

# Load imputed train data
train = pd.read_csv("./Energy/_final/train_test/42train_imputed.csv", index_col=0)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

def create_time_features(df):
    """Create time-based features from datetime column"""
    df = df.copy()
    df['일시'] = pd.to_datetime(df['일시'])
    
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['quarter'] = df['일시'].dt.quarter
    df['day_of_year'] = df['일시'].dt.dayofyear
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

def predict_sunshine_hours(train_df, test_df):
    """Step 1: Predict sunshine hours for test data"""
    print("="*50)
    print("STEP 1: PREDICTING SUNSHINE HOURS")
    print("="*50)
    
    # Create copies and add time features
    train_data = create_time_features(train_df.copy())
    test_data = create_time_features(test_df.copy())
    
    # Features for sunshine prediction (excluding sunshine and solar radiation)
    sunshine_feature_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
                            '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
                            'month', 'day', 'hour', 'dayofweek', 'quarter', 'day_of_year',
                            'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 
                            'dayofweek_sin', 'dayofweek_cos']
    
    # Building type encoding
    combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    building_type_encoded = pd.get_dummies(combined_data['건물유형'], prefix='building_type')
    combined_data = pd.concat([combined_data, building_type_encoded], axis=1)
    sunshine_feature_cols.extend(building_type_encoded.columns.tolist())
    
    # Split back
    train_encoded = combined_data.iloc[:len(train_data)].copy()
    test_encoded = combined_data.iloc[len(train_data):].copy().reset_index(drop=True)
    
    # Prepare data for sunshine prediction
    X_train_sunshine = train_encoded[sunshine_feature_cols].fillna(0)
    y_train_sunshine = train_encoded['일조(hr)']
    X_test_sunshine = test_encoded[sunshine_feature_cols].fillna(0)
    
    # Train/validation split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_sunshine, y_train_sunshine, test_size=0.2, random_state=SEED
    )
    
    print(f"Training sunshine prediction model...")
    print(f"Train samples: {len(X_train_split)}, Validation samples: {len(X_val_split)}")
    
    # KNN for sunshine prediction
    scaler_sunshine = StandardScaler()
    X_train_scaled = scaler_sunshine.fit_transform(X_train_split)
    X_val_scaled = scaler_sunshine.transform(X_val_split)
    X_test_scaled = scaler_sunshine.transform(X_test_sunshine)
    
    knn_sunshine = KNeighborsRegressor(n_neighbors=15, weights='distance')
    knn_sunshine.fit(X_train_scaled, y_train_split)
    knn_sunshine_pred = knn_sunshine.predict(X_test_scaled)
    
    # LightGBM for sunshine prediction
    lgb_params_sunshine = {
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
    
    train_data_sunshine = lgb.Dataset(X_train_split, label=y_train_split)
    val_data_sunshine = lgb.Dataset(X_val_split, label=y_val_split, reference=train_data_sunshine)
    
    lgb_sunshine = lgb.train(
        lgb_params_sunshine,
        train_data_sunshine,
        valid_sets=[val_data_sunshine],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    lgb_sunshine_pred = lgb_sunshine.predict(X_test_sunshine, num_iteration=lgb_sunshine.best_iteration)
    
    # Ensemble sunshine predictions
    sunshine_predictions = 0.6 * lgb_sunshine_pred + 0.4 * knn_sunshine_pred
    
    # Apply constraints
    sunshine_predictions = np.maximum(sunshine_predictions, 0)  # Non-negative
    sunshine_predictions = np.minimum(sunshine_predictions, 1)  # Max 1 hour per hour
    
    # Apply nighttime constraints
    test_hours = test_encoded['hour'].values
    night_mask = (test_hours <= 5) | (test_hours >= 21)
    sunshine_predictions[night_mask] = 0
    
    print(f"Sunshine prediction results:")
    print(f"Min: {sunshine_predictions.min():.4f}, Max: {sunshine_predictions.max():.4f}")
    print(f"Mean: {sunshine_predictions.mean():.4f}, Std: {sunshine_predictions.std():.4f}")
    print(f"Non-zero predictions: {(sunshine_predictions > 0).sum()}/{len(sunshine_predictions)}")
    print(f"LightGBM validation RMSE: {lgb_sunshine.best_score['valid_0']['rmse']:.4f}")
    
    return sunshine_predictions, lgb_sunshine, knn_sunshine, scaler_sunshine, test_encoded

def predict_solar_radiation_improved(train_df, test_df_with_sunshine, predicted_sunshine):
    """Step 2: Improved solar radiation prediction with physical constraints"""
    print("="*50)
    print("STEP 2: IMPROVED SOLAR RADIATION PREDICTION")
    print("="*50)
    
    # Add predicted sunshine to test data
    test_data = test_df_with_sunshine.copy()
    test_data['일조(hr)'] = predicted_sunshine
    
    train_data = create_time_features(train_df.copy())
    
    # Get building_type columns from test_data
    building_type_cols = [col for col in test_data.columns if col.startswith('building_type_')]
    
    # Add building_type columns to train_data
    if building_type_cols:
        train_building_types = pd.get_dummies(train_data['건물유형'], prefix='building_type')
        for col in building_type_cols:
            if col not in train_building_types.columns:
                train_building_types[col] = 0
        train_building_types = train_building_types[building_type_cols]
        train_data = pd.concat([train_data, train_building_types], axis=1)
    
    # Enhanced features for better physical relationship
    train_data['sunshine_temp_interaction'] = train_data['일조(hr)'] * train_data['기온(°C)']
    train_data['sunshine_humidity_interaction'] = train_data['일조(hr)'] * (100 - train_data['습도(%)']) / 100
    train_data['clear_sky_indicator'] = ((train_data['강수량(mm)'] == 0) & (train_data['습도(%)'] < 70)).astype(int)
    
    test_data['sunshine_temp_interaction'] = test_data['일조(hr)'] * test_data['기온(°C)']
    test_data['sunshine_humidity_interaction'] = test_data['일조(hr)'] * (100 - test_data['습도(%)']) / 100
    test_data['clear_sky_indicator'] = ((test_data['강수량(mm)'] == 0) & (test_data['습도(%)'] < 70)).astype(int)
    
    # Enhanced feature set
    solar_feature_cols = [
        # Basic weather
        '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)',
        # Building info
        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
        # Time features
        'month', 'day', 'hour', 'dayofweek', 'quarter', 'day_of_year',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        # Interaction features
        'sunshine_temp_interaction', 'sunshine_humidity_interaction', 'clear_sky_indicator'
    ]
    solar_feature_cols.extend(building_type_cols)
    
    # Prepare data
    X_train_solar = train_data[solar_feature_cols].fillna(0)
    y_train_solar = train_data['일사(MJ/m2)']
    X_test_solar = test_data[solar_feature_cols].fillna(0)
    
    # Filter training data to improve physical consistency
    sunshine_solar_ratio = y_train_solar / (train_data['일조(hr)'] + 1e-6)
    reasonable_ratio_mask = (sunshine_solar_ratio >= 0.5) & (sunshine_solar_ratio <= 8.0)
    
    print(f"Filtering training data for physical consistency...")
    print(f"Original samples: {len(X_train_solar)}")
    print(f"Filtered samples: {reasonable_ratio_mask.sum()}")
    
    X_train_solar = X_train_solar[reasonable_ratio_mask]
    y_train_solar = y_train_solar[reasonable_ratio_mask]
    
    # Train/validation split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_solar, y_train_solar, test_size=0.2, random_state=SEED
    )
    
    # KNN with adjusted parameters
    scaler_solar = StandardScaler()
    X_train_scaled = scaler_solar.fit_transform(X_train_split)
    X_val_scaled = scaler_solar.transform(X_val_split)
    X_test_scaled = scaler_solar.transform(X_test_solar)
    
    knn_solar = KNeighborsRegressor(n_neighbors=20, weights='distance')
    knn_solar.fit(X_train_scaled, y_train_split)
    knn_solar_pred = knn_solar.predict(X_test_scaled)
    
    # LightGBM with adjusted parameters
    lgb_params_solar = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 20,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': SEED
    }
    
    train_data_solar = lgb.Dataset(X_train_split, label=y_train_split)
    val_data_solar = lgb.Dataset(X_val_split, label=y_val_split, reference=train_data_solar)
    
    lgb_solar = lgb.train(
        lgb_params_solar,
        train_data_solar,
        valid_sets=[val_data_solar],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )
    
    lgb_solar_pred = lgb_solar.predict(X_test_solar, num_iteration=lgb_solar.best_iteration)
    
    # Conservative ensemble
    solar_predictions = 0.6 * knn_solar_pred + 0.4 * lgb_solar_pred
    
    # Apply strict physical constraints
    solar_predictions = np.maximum(solar_predictions, 0)
    
    # Strong sunshine-based constraints
    for i in range(len(solar_predictions)):
        sunshine_val = predicted_sunshine[i]
        hour_val = test_data.iloc[i]['hour']
        
        # Nighttime: Force to 0
        if hour_val <= 5 or hour_val >= 21:
            solar_predictions[i] = 0
        # Dawn/dusk: Limit based on sunshine
        elif hour_val in [6, 7, 18, 19, 20]:
            max_allowed = sunshine_val * 2.0
            solar_predictions[i] = min(solar_predictions[i], max_allowed)
        # Daytime: Apply sunshine-based limits
        else:
            if sunshine_val <= 0.1:
                solar_predictions[i] = min(solar_predictions[i], 0.3)
            elif sunshine_val <= 0.3:
                solar_predictions[i] = min(solar_predictions[i], sunshine_val * 2.5)
            elif sunshine_val <= 0.7:
                solar_predictions[i] = min(solar_predictions[i], sunshine_val * 3.5)
            else:
                solar_predictions[i] = min(solar_predictions[i], sunshine_val * 4.0)
    
    # Final sanity check
    solar_predictions = np.minimum(solar_predictions, 4.0)
    
    print(f"Solar radiation prediction results:")
    print(f"Min: {solar_predictions.min():.4f}, Max: {solar_predictions.max():.4f}")
    print(f"Mean: {solar_predictions.mean():.4f}, Std: {solar_predictions.std():.4f}")
    print(f"Non-zero predictions: {(solar_predictions > 0).sum()}/{len(solar_predictions)}")
    print(f"LightGBM validation RMSE: {lgb_solar.best_score['valid_0']['rmse']:.4f}")
    
    return solar_predictions, lgb_solar, knn_solar, scaler_solar

def predict_test_sunshine_and_solar(train_df, test_df):
    """Complete prediction pipeline"""
    print("Starting prediction pipeline for test data...")
    
    # Step 1: Predict sunshine hours
    sunshine_pred, lgb_sunshine_model, knn_sunshine_model, scaler_sunshine, test_with_features = predict_sunshine_hours(train_df, test_df)
    
    # Step 2: Predict solar radiation using predicted sunshine
    solar_pred, lgb_solar_model, knn_solar_model, scaler_solar = predict_solar_radiation_improved(train_df, test_with_features, sunshine_pred)
    
    # Create final result dataframe
    result_df = test_df.copy()
    result_df['일조(hr)'] = sunshine_pred
    result_df['일사(MJ/m2)'] = solar_pred
    
    return result_df, {
        'sunshine_models': (lgb_sunshine_model, knn_sunshine_model, scaler_sunshine),
        'solar_models': (lgb_solar_model, knn_solar_model, scaler_solar)
    }

# Execute prediction pipeline
print("Starting sunshine and solar radiation prediction for test data...")
test_with_predictions, models = predict_test_sunshine_and_solar(train, test)

# Verification
print(f"\n" + "="*60)
print("FINAL VERIFICATION")
print("="*60)
print(f"Original test shape: {test.shape}")
print(f"Test with predictions shape: {test_with_predictions.shape}")

# Check predicted values by building (sample)
print(f"\nPredicted values by building (first 5 buildings):")
for bno in test_with_predictions['건물번호'].unique()[:5]:
    building_data = test_with_predictions[test_with_predictions['건물번호'] == bno]
    sunshine_stats = building_data['일조(hr)']
    solar_stats = building_data['일사(MJ/m2)']
    
    print(f"Building {bno}:")
    print(f"  Sunshine - Non-zero: {(sunshine_stats > 0).sum()}/{len(sunshine_stats)}, Mean: {sunshine_stats.mean():.4f}")
    print(f"  Solar    - Non-zero: {(solar_stats > 0).sum()}/{len(solar_stats)}, Mean: {solar_stats.mean():.4f}")

print(f"\n✅ Prediction completed successfully!")
print(f"Final predictions shape: {test_with_predictions.shape}")