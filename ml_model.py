import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pickle
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Optional seaborn import
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Seaborn not available - some plotting features may be limited")
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated libraries (optional imports)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("XGBoost available for GPU acceleration")
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("CatBoost available for GPU acceleration")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print("LightGBM available for GPU acceleration")
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

class GroundwaterPredictor:
    """
    Machine Learning model for predicting groundwater levels with GPU acceleration support
    """
    
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.scaler = StandardScaler()
        self.state_encoder = LabelEncoder()
        self.district_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_column = 'groundwater_level'
        self.is_trained = False
        self.use_gpu = use_gpu
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> Dict[str, bool]:
        """Check GPU availability for different libraries"""
        gpu_status = {
            'cuda_available': False,
            'xgboost_gpu': False,
            'catboost_gpu': False,
            'lightgbm_gpu': False
        }
        
        # Check CUDA availability
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_status['cuda_available'] = True
                print("NVIDIA GPU detected!")
        except:
            pass
        
        # Check XGBoost GPU support
        if XGB_AVAILABLE and gpu_status['cuda_available']:
            try:
                # Test if XGBoost can use GPU
                test_data = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
                test_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
                test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                gpu_status['xgboost_gpu'] = True
                print("XGBoost GPU support confirmed")
            except Exception as e:
                print(f"XGBoost GPU not available: {e}")
        
        # Check CatBoost GPU support
        if CATBOOST_AVAILABLE and gpu_status['cuda_available']:
            try:
                test_model = cb.CatBoostRegressor(iterations=1, task_type="GPU", devices='0', verbose=False)
                test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                gpu_status['catboost_gpu'] = True
                print("CatBoost GPU support confirmed")
            except Exception as e:
                print(f"CatBoost GPU not available: {e}")
        
        # Check LightGBM GPU support  
        if LGB_AVAILABLE and gpu_status['cuda_available']:
            try:
                test_model = lgb.LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0, n_estimators=1)
                test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                gpu_status['lightgbm_gpu'] = True
                print("LightGBM GPU support confirmed")
            except Exception as e:
                print(f"LightGBM GPU not available: {e}")
        
        if not any([gpu_status['xgboost_gpu'], gpu_status['catboost_gpu'], gpu_status['lightgbm_gpu']]):
            print("No GPU acceleration available. Using CPU-based models.")
        
        return gpu_status
        
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
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    model_type: str = "auto", use_gpu: Optional[bool] = None):
        """
        Train the machine learning model with optional GPU acceleration
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model ('auto', 'xgboost', 'catboost', 'lightgbm', 'random_forest', 'gradient_boosting')
            use_gpu: Override GPU usage (None uses class setting)
        """
        
        # Determine GPU usage
        gpu_enabled = use_gpu if use_gpu is not None else (self.use_gpu and any(self.gpu_available.values()))
        
        # Auto-select best available model
        if model_type == "auto":
            if gpu_enabled and self.gpu_available['xgboost_gpu']:
                model_type = "xgboost"
            elif gpu_enabled and self.gpu_available['catboost_gpu']:
                model_type = "catboost"
            elif gpu_enabled and self.gpu_available['lightgbm_gpu']:
                model_type = "lightgbm"
            else:
                model_type = "random_forest"
        
        print(f"Training {model_type} model{'with GPU acceleration' if gpu_enabled else ' (CPU)'}")
        
        # Scale features for sklearn models only
        if model_type in ["random_forest", "gradient_boosting"]:
            X_train_processed = self.scaler.fit_transform(X_train)
        else:
            # Tree-based models like XGBoost, CatBoost, LightGBM don't require scaling
            X_train_processed = X_train.values
            # Still fit scaler for consistency in prediction pipeline
            self.scaler.fit(X_train)
        
        if model_type == "xgboost" and XGB_AVAILABLE:
            # XGBoost with GPU support
            if gpu_enabled and self.gpu_available['xgboost_gpu']:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.8, 0.9, 1.0],
                    'tree_method': ['gpu_hist'],
                    'gpu_id': [0]
                }
                base_model = xgb.XGBRegressor(random_state=42, tree_method='gpu_hist', gpu_id=0)
            else:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.8, 0.9, 1.0]
                }
                base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
            self.model = GridSearchCV(base_model, param_grid, cv=3, 
                                    scoring='neg_mean_squared_error', n_jobs=1 if gpu_enabled else -1)
        
        elif model_type == "catboost" and CATBOOST_AVAILABLE:
            # CatBoost with GPU support
            if gpu_enabled and self.gpu_available['catboost_gpu']:
                param_grid = {
                    'iterations': [100, 200, 300],
                    'depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'task_type': ['GPU'],
                    'devices': ['0']
                }
                base_model = cb.CatBoostRegressor(random_state=42, task_type="GPU", 
                                                devices='0', verbose=False)
            else:
                param_grid = {
                    'iterations': [100, 200, 300],
                    'depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.05, 0.01]
                }
                base_model = cb.CatBoostRegressor(random_state=42, verbose=False, thread_count=-1)
            
            self.model = GridSearchCV(base_model, param_grid, cv=3, 
                                    scoring='neg_mean_squared_error', n_jobs=1)
        
        elif model_type == "lightgbm" and LGB_AVAILABLE:
            # LightGBM with GPU support
            if gpu_enabled and self.gpu_available['lightgbm_gpu']:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.8, 0.9, 1.0],
                    'device': ['gpu'],
                    'gpu_platform_id': [0],
                    'gpu_device_id': [0]
                }
                base_model = lgb.LGBMRegressor(random_state=42, device='gpu', 
                                             gpu_platform_id=0, gpu_device_id=0)
            else:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.8, 0.9, 1.0]
                }
                base_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
            
            self.model = GridSearchCV(base_model, param_grid, cv=3, 
                                    scoring='neg_mean_squared_error', n_jobs=1 if gpu_enabled else -1)
        
        elif model_type == "random_forest":
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
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        print("Starting model training...")
        start_time = datetime.now()
        
        self.model.fit(X_train_processed, y_train)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best parameters: {self.model.best_params_}")
        print(f"Best CV score: {-self.model.best_score_:.4f}")
        
        self.is_trained = True
        
        # Store training info
        self.training_info = {
            'model_type': model_type,
            'gpu_used': gpu_enabled,
            'training_time_seconds': training_time,
            'cv_score': -self.model.best_score_,
            'best_params': self.model.best_params_
        }
        
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
        
        # Convert to native Python float to avoid JSON serialization issues
        return float(max(0.1, prediction))  # Ensure minimum positive value
    
    def save_model(self, model_path: str = "groundwater_model", 
                   format_type: str = "joblib", 
                   metrics: Optional[Dict] = None,
                   include_metadata: bool = True):
        """
        Save the trained model and preprocessing objects
        
        Args:
            model_path: Base path for the model file (extension will be added automatically)
            format_type: Format to save the model ('joblib', 'pickle')
            metrics: Training/evaluation metrics to include in metadata
            include_metadata: Whether to save additional metadata
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'state_encoder': self.state_encoder,
            'district_encoder': self.district_encoder,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'format_version': '2.0',
            'saved_timestamp': datetime.now().isoformat(),
            'sklearn_version': joblib.__version__
        }
        
        # Add metrics if provided
        if metrics:
            model_data['training_metrics'] = metrics
        
        # Determine file extension and save method
        if format_type.lower() == "pickle":
            file_path = f"{model_path}.pkl"
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Model saved to {file_path} (pickle format)")
            except Exception as e:
                print(f"Error saving model in pickle format: {e}")
                return False
                
        elif format_type.lower() == "joblib":
            file_path = f"{model_path}.joblib"
            try:
                joblib.dump(model_data, file_path, compress=3)  # Add compression
                print(f"Model saved to {file_path} (joblib format)")
            except Exception as e:
                print(f"Error saving model in joblib format: {e}")
                return False
        else:
            raise ValueError("format_type must be 'joblib' or 'pickle'")
        
        # Save metadata separately if requested
        if include_metadata:
            metadata = {
                'model_path': file_path,
                'format_type': format_type,
                'feature_count': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'saved_timestamp': model_data['saved_timestamp'],
                'format_version': model_data['format_version']
            }
            
            if metrics:
                metadata['metrics'] = metrics
            
            metadata_path = f"{model_path}_metadata.json"
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Metadata saved to {metadata_path}")
            except Exception as e:
                print(f"Warning: Could not save metadata: {e}")
        
        return True
    
    def load_model(self, model_path: str = "groundwater_model.joblib", 
                   auto_detect_format: bool = True):
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the model file
            auto_detect_format: Whether to auto-detect the format from file extension
        """
        
        try:
            # Auto-detect format if requested
            if auto_detect_format:
                if model_path.endswith('.pkl'):
                    format_type = 'pickle'
                elif model_path.endswith('.joblib'):
                    format_type = 'joblib'
                else:
                    # Try to detect by attempting to load
                    if os.path.exists(f"{model_path}.joblib"):
                        model_path = f"{model_path}.joblib"
                        format_type = 'joblib'
                    elif os.path.exists(f"{model_path}.pkl"):
                        model_path = f"{model_path}.pkl"
                        format_type = 'pickle'
                    else:
                        format_type = 'joblib'  # Default
            
            # Load based on format
            if model_path.endswith('.pkl') or format_type == 'pickle':
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                print(f"Model loaded from {model_path} (pickle format)")
            else:
                model_data = joblib.load(model_path)
                print(f"Model loaded from {model_path} (joblib format)")
            
            # Extract model components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.state_encoder = model_data['state_encoder']
            self.district_encoder = model_data['district_encoder']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            
            # Print metadata if available
            if 'saved_timestamp' in model_data:
                print(f"Model was saved on: {model_data['saved_timestamp']}")
            if 'training_metrics' in model_data:
                print("Training metrics:")
                for metric, value in model_data['training_metrics'].items():
                    print(f"  {metric}: {value:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self, model_path: str) -> Optional[Dict]:
        """Get information about a saved model without loading it"""
        
        metadata_path = f"{model_path}_metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading metadata: {e}")
        
        # Try to extract basic info from the model file itself
        try:
            if os.path.exists(f"{model_path}.joblib"):
                model_data = joblib.load(f"{model_path}.joblib")
            elif os.path.exists(f"{model_path}.pkl"):
                with open(f"{model_path}.pkl", 'rb') as f:
                    model_data = pickle.load(f)
            else:
                return None
            
            return {
                'feature_count': len(model_data.get('feature_columns', [])),
                'target_column': model_data.get('target_column', 'unknown'),
                'saved_timestamp': model_data.get('saved_timestamp', 'unknown'),
                'has_metrics': 'training_metrics' in model_data
            }
        except Exception as e:
            print(f"Error reading model info: {e}")
            return None
    
    @staticmethod
    def compare_models(model_paths: List[str]) -> Optional[Dict]:
        """
        Compare multiple saved models and return the best one based on metrics
        
        Args:
            model_paths: List of model file paths (without extensions)
            
        Returns:
            Dictionary with comparison results and best model recommendation
        """
        
        model_comparisons = []
        
        for model_path in model_paths:
            try:
                # Try to load model info
                temp_predictor = GroundwaterPredictor()
                model_info = temp_predictor.get_model_info(model_path)
                
                if model_info is None:
                    continue
                
                # Try to load the actual model to get metrics
                loaded = False
                metrics = None
                
                # Try different formats
                for ext in ['.joblib', '.pkl']:
                    full_path = f"{model_path}{ext}"
                    if os.path.exists(full_path):
                        if temp_predictor.load_model(full_path):
                            loaded = True
                            # Extract metrics from model data
                            try:
                                if ext == '.joblib':
                                    model_data = joblib.load(full_path)
                                else:
                                    with open(full_path, 'rb') as f:
                                        model_data = pickle.load(f)
                                
                                metrics = model_data.get('training_metrics', {})
                            except:
                                pass
                            break
                
                if loaded:
                    comparison_entry = {
                        'model_path': model_path,
                        'model_info': model_info,
                        'metrics': metrics,
                        'r2_score': metrics.get('r2', 0) if metrics else 0,
                        'rmse': metrics.get('rmse', float('inf')) if metrics else float('inf'),
                        'saved_timestamp': model_info.get('saved_timestamp', ''),
                        'feature_count': model_info.get('feature_count', 0)
                    }
                    model_comparisons.append(comparison_entry)
                    
            except Exception as e:
                print(f"Error comparing model {model_path}: {e}")
                continue
        
        if not model_comparisons:
            print("No valid models found for comparison")
            return None
        
        # Find best model based on R² score (higher is better)
        best_model = max(model_comparisons, key=lambda x: x['r2_score'])
        
        # Sort all models by R² score (descending)
        model_comparisons.sort(key=lambda x: x['r2_score'], reverse=True)
        
        comparison_result = {
            'best_model': best_model,
            'all_models': model_comparisons,
            'comparison_summary': {
                'total_models': len(model_comparisons),
                'best_r2_score': best_model['r2_score'],
                'best_rmse': best_model['rmse'],
                'best_model_path': best_model['model_path']
            }
        }
        
        # Print comparison results
        print("\n" + "=" * 60)
        print("🏆 MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(f"{'Model Path':<30} {'R² Score':<10} {'RMSE':<10} {'Date Saved':<20}")
        print("-" * 60)
        
        for model in model_comparisons:
            timestamp = model['saved_timestamp']
            if timestamp and timestamp != 'unknown':
                try:
                    # Format timestamp for display
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = timestamp[:16] if len(timestamp) > 16 else timestamp
            else:
                date_str = 'Unknown'
            
            print(f"{model['model_path']:<30} {model['r2_score']:<10.4f} "
                  f"{model['rmse']:<10.4f} {date_str:<20}")
        
        print(f"\n🥇 Best Model: {best_model['model_path']} (R² = {best_model['r2_score']:.4f})")
        
        return comparison_result
    
    @staticmethod 
    def load_best_model(model_paths: List[str]) -> Optional['GroundwaterPredictor']:
        """
        Compare models and automatically load the best one
        
        Args:
            model_paths: List of model file paths to compare
            
        Returns:
            GroundwaterPredictor instance with the best model loaded, or None if failed
        """
        
        comparison_result = GroundwaterPredictor.compare_models(model_paths)
        
        if comparison_result is None:
            print("❌ No models available for comparison")
            return None
        
        best_model_path = comparison_result['best_model']['model_path']
        
        # Load the best model
        predictor = GroundwaterPredictor()
        
        # Try different formats for the best model
        for ext in ['.joblib', '.pkl']:
            full_path = f"{best_model_path}{ext}"
            if os.path.exists(full_path):
                if predictor.load_model(full_path):
                    print(f"✅ Loaded best model: {best_model_path}")
                    return predictor
        
        print(f"❌ Failed to load best model: {best_model_path}")
        return None

def main(use_gpu: bool = True, model_type: str = "auto"):
    """Main training pipeline with GPU support"""
    
    # Initialize predictor with GPU support
    predictor = GroundwaterPredictor(use_gpu=use_gpu)
    
    # Load data
    df = predictor.load_data()
    if df is None:
        print("Failed to load data")
        return None
    
    # Prepare features
    df_prepared = predictor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(df_prepared)
    
    # Train model with GPU acceleration
    predictor.train_model(X_train, y_train, model_type=model_type)
    
    # Evaluate model
    metrics, predictions = predictor.evaluate_model(X_test, y_test)
    
    # Show feature importance
    feature_imp = predictor.feature_importance()
    
    # Save model with training metrics and metadata
    predictor.save_model(
        model_path="groundwater_model", 
        format_type="joblib",
        metrics=metrics,
        include_metadata=True
    )
    
    # Also save in pickle format as backup
    predictor.save_model(
        model_path="groundwater_model_backup", 
        format_type="pickle",
        metrics=metrics,
        include_metadata=True
    )
    
    # Test prediction
    print("\nTesting prediction for Karnataka, Bangalore Urban, 2026, March:")
    prediction = predictor.predict_future("KARNATAKA", "BANGALORE URBAN", 2026, 3)
    print(f"Predicted groundwater level: {prediction:.2f}")
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"Model Type: {predictor.training_info['model_type']}")
    print(f"GPU Used: {predictor.training_info['gpu_used']}")
    print(f"Training Time: {predictor.training_info['training_time_seconds']:.2f} seconds")
    print(f"Cross-validation Score: {predictor.training_info['cv_score']:.4f}")
    
    return predictor, metrics

if __name__ == "__main__":
    model, metrics = main()