import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class GroundwaterPredictor:
    """
    Machine Learning model for predicting groundwater levels
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.state_encoder = LabelEncoder()
        self.district_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_column = 'groundwater_level'
        self.is_trained = False
        
    def load_data(self, data_path: str = "processed_groundwater_data.csv") -> pd.DataFrame:
        """Load the processed groundwater data"""
        try:
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training"""
        
        # Create a copy to avoid modifying original data
        df_prep = df.copy()
        
        # Encode categorical variables
        df_prep['state_encoded'] = self.state_encoder.fit_transform(df_prep['state'].astype(str))
        df_prep['district_encoded'] = self.district_encoder.fit_transform(df_prep['district'].astype(str))
        
        # Extract additional time features
        df_prep['quarter'] = df_prep['date'].dt.quarter
        df_prep['day_of_year'] = df_prep['date'].dt.dayofyear
        df_prep['is_monsoon'] = ((df_prep['month_num'] >= 6) & (df_prep['month_num'] <= 9)).astype(int)
        df_prep['is_winter'] = ((df_prep['month_num'] >= 11) | (df_prep['month_num'] <= 2)).astype(int)
        
        # Define feature columns
        self.feature_columns = [
            'year', 'month_num', 'quarter', 'day_of_year', 'is_monsoon', 'is_winter',
            'state_encoded', 'district_encoded',
            'rainfall_mm', 'rainfall_nc', 'rainfall_pq', 'total_rainfall',
            'geographical_area_ha', 'recharge_area_ha', 'total_area_ha',
            'groundwater_recharge_ham', 'recharge_nc', 'total_recharge'
        ]
        
        # Add lag features if they exist
        lag_columns = [col for col in df_prep.columns if col.endswith(('_lag1', '_lag3', '_lag12'))]
        self.feature_columns.extend(lag_columns)
        
        # Handle missing values
        for col in self.feature_columns:
            if col in df_prep.columns:
                df_prep[col] = df_prep[col].fillna(0)
            else:
                df_prep[col] = 0
                
        print(f"Prepared {len(self.feature_columns)} features for training")
        return df_prep
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets chronologically"""
        
        # Sort by date to ensure chronological split
        df_sorted = df.sort_values('date')
        
        # Use 80% for training (2020-2024) and 20% for testing (2025)
        split_date = pd.Timestamp('2025-01-01')
        
        train_df = df_sorted[df_sorted['date'] < split_date]
        test_df = df_sorted[df_sorted['date'] >= split_date]
        
        print(f"Train set: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"Test set: {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")
        
        # Prepare features and target
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.target_column]
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "random_forest"):
        """Train the machine learning model"""
        
        print(f"Training {model_type} model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if model_type == "random_forest":
            # Random Forest with hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            self.model = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            
        elif model_type == "gradient_boosting":
            # Gradient Boosting with hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10],
                'learning_rate': [0.1, 0.05],
                'subsample': [0.8, 1.0]
            }
            
            gb = GradientBoostingRegressor(random_state=42)
            self.model = GridSearchCV(gb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {self.model.best_params_}")
        print(f"Best CV score: {-self.model.best_score_:.4f}")
        
        self.is_trained = True
        
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the trained model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print("Model Evaluation Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return metrics, y_pred
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature importance
        if hasattr(self.model.best_estimator_, 'feature_importances_'):
            importance = self.model.best_estimator_.feature_importances_
            
            feature_imp = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print(feature_imp.head(10))
            
            return feature_imp
        else:
            print("Feature importance not available for this model type")
            return None
    
    def predict_future(self, state: str, district: str, year: int, month: int, 
                      additional_features: Optional[Dict] = None) -> float:
        """Predict groundwater level for a specific location and time"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create feature vector
        features = np.zeros(len(self.feature_columns))
        feature_dict = dict(zip(self.feature_columns, features))
        
        # Set basic time features
        feature_dict['year'] = year
        feature_dict['month_num'] = month
        feature_dict['quarter'] = (month - 1) // 3 + 1
        
        # Create date for day_of_year calculation
        try:
            date = pd.Timestamp(year=year, month=month, day=1)
            feature_dict['day_of_year'] = date.dayofyear
        except:
            feature_dict['day_of_year'] = month * 30  # Approximate
        
        # Set seasonal features
        feature_dict['is_monsoon'] = 1 if 6 <= month <= 9 else 0
        feature_dict['is_winter'] = 1 if month >= 11 or month <= 2 else 0
        
        # Encode state and district
        try:
            if state in self.state_encoder.classes_:
                feature_dict['state_encoded'] = self.state_encoder.transform([state])[0]
            else:
                feature_dict['state_encoded'] = 0  # Unknown state
                
            if district in self.district_encoder.classes_:
                feature_dict['district_encoded'] = self.district_encoder.transform([district])[0]
            else:
                feature_dict['district_encoded'] = 0  # Unknown district
        except:
            feature_dict['state_encoded'] = 0
            feature_dict['district_encoded'] = 0
        
        # Set additional features if provided
        if additional_features:
            for key, value in additional_features.items():
                if key in feature_dict:
                    feature_dict[key] = value
        
        # Convert to array and scale
        feature_array = np.array([list(feature_dict.values())])
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.model.predict(feature_array_scaled)[0]
        
        return max(0.1, prediction)  # Ensure minimum positive value
    
    def save_model(self, model_path: str = "groundwater_model.joblib"):
        """Save the trained model and preprocessing objects"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'state_encoder': self.state_encoder,
            'district_encoder': self.district_encoder,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = "groundwater_model.joblib"):
        """Load a pre-trained model"""
        
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.state_encoder = model_data['state_encoder']
            self.district_encoder = model_data['district_encoder']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            
            self.is_trained = True
            print(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main training pipeline"""
    
    # Initialize predictor
    predictor = GroundwaterPredictor()
    
    # Load data
    df = predictor.load_data()
    if df is None:
        print("Failed to load data")
        return None
    
    # Prepare features
    df_prepared = predictor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(df_prepared)
    
    # Train model
    predictor.train_model(X_train, y_train, model_type="random_forest")
    
    # Evaluate model
    metrics, predictions = predictor.evaluate_model(X_test, y_test)
    
    # Show feature importance
    feature_imp = predictor.feature_importance()
    
    # Save model
    predictor.save_model()
    
    # Test prediction
    print("\nTesting prediction for Karnataka, Bangalore Urban, 2026, March:")
    prediction = predictor.predict_future("KARNATAKA", "BANGALORE URBAN", 2026, 3)
    print(f"Predicted groundwater level: {prediction:.2f}")
    
    return predictor, metrics

if __name__ == "__main__":
    model, metrics = main()