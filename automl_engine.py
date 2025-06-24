import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
from typing import List, Dict, Tuple, Any

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    A comprehensive data preprocessing class that handles:
    - Column type detection
    - Missing value imputation
    - Feature scaling
    - Categorical encoding
    - Constant column removal
    """
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names_out = None
        self.column_types = {}
        self.dropped_columns = []
        self.preprocessing_summary = {}
        self.original_columns = []
        
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect and classify columns as numerical, categorical, datetime, or constant.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with column types as keys and column lists as values
        """
        logger.info("Starting column type detection...")
        
        column_types = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'constant': []
        }
        
        self.original_columns = df.columns.tolist()
        
        for col in df.columns:
            # Check if column is constant (all values are the same)
            if df[col].nunique() <= 1:
                column_types['constant'].append(col)
                logger.debug(f"Column '{col}' identified as constant")
                continue
            
            # Check for datetime columns
            if df[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types['datetime'].append(col)
                logger.debug(f"Column '{col}' identified as datetime")
                continue
            
            # Try to convert to datetime if it looks like a date
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head(100), errors='raise')
                    column_types['datetime'].append(col)
                    logger.debug(f"Column '{col}' identified as datetime (converted)")
                    continue
                except:
                    pass
            
            # Check for numerical columns
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types['numerical'].append(col)
                logger.debug(f"Column '{col}' identified as numerical")
            else:
                # Everything else is categorical
                column_types['categorical'].append(col)
                logger.debug(f"Column '{col}' identified as categorical")
        
        self.column_types = column_types
        logger.info(f"Column type detection complete: {len(column_types['numerical'])} numerical, "
                   f"{len(column_types['categorical'])} categorical, "
                   f"{len(column_types['datetime'])} datetime, "
                   f"{len(column_types['constant'])} constant")
        
        return column_types
    
    def drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop constant columns from the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with constant columns removed
        """
        constant_cols = self.column_types.get('constant', [])
        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)
            self.dropped_columns.extend(constant_cols)
        
        return df
    
    def handle_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle datetime columns by extracting useful features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with datetime columns processed
        """
        datetime_cols = self.column_types.get('datetime', [])
        
        for col in datetime_cols:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Extract datetime features
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                
                # Add new columns to numerical type
                new_cols = [f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek']
                self.column_types['numerical'].extend(new_cols)
                
                # Remove original datetime column
                df = df.drop(columns=[col])
                self.dropped_columns.append(col)
                
                logger.info(f"Processed datetime column '{col}' into features: {new_cols}")
                
            except Exception as e:
                logger.warning(f"Failed to process datetime column '{col}': {e}")
                # Treat as categorical if datetime processing fails
                self.column_types['categorical'].append(col)
                if col in self.column_types['datetime']:
                    self.column_types['datetime'].remove(col)
        
        return df
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create a scikit-learn preprocessing pipeline.
        
        Returns:
            ColumnTransformer with appropriate preprocessing steps
        """
        logger.info("Creating preprocessing pipeline...")
        
        transformers = []
        
        # Numerical pipeline: impute with mean, then scale
        if self.column_types['numerical']:
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('numerical', numerical_pipeline, self.column_types['numerical']))
            logger.debug(f"Added numerical pipeline for columns: {self.column_types['numerical']}")
        
        # Categorical pipeline: impute with most frequent, then one-hot encode
        if self.column_types['categorical']:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('categorical', categorical_pipeline, self.column_types['categorical']))
            logger.debug(f"Added categorical pipeline for columns: {self.column_types['categorical']}")
        
        if not transformers:
            raise ValueError("No valid columns found for preprocessing")
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        logger.info("Preprocessing pipeline created successfully")
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the preprocessing pipeline and transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (transformed_data, feature_names)
        """
        logger.info("Starting fit_transform process...")
        
        # Detect column types
        self.detect_column_types(df)
        
        # Drop constant columns
        df_processed = self.drop_constant_columns(df)
        
        # Handle datetime columns
        df_processed = self.handle_datetime_columns(df_processed)
        
        # Update column types after datetime processing
        remaining_cols = df_processed.columns.tolist()
        for col_type in ['numerical', 'categorical']:
            self.column_types[col_type] = [col for col in self.column_types[col_type] 
                                         if col in remaining_cols]
        
        # Create and fit the preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline()
        
        # Fit and transform the data
        X_transformed = preprocessor.fit_transform(df_processed)
        
        # Get feature names
        feature_names = self._get_feature_names()
        self.feature_names_out = feature_names
        
        # Create preprocessing summary
        self.preprocessing_summary = {
            'total_features_before': len(self.original_columns),
            'total_features_after': len(feature_names),
            'numerical_features': len(self.column_types['numerical']),
            'categorical_features': len(self.column_types['categorical']),
            'features_dropped': len(self.dropped_columns),
            'dropped_features': self.dropped_columns
        }
        
        logger.info(f"Preprocessing complete. Shape: {X_transformed.shape}")
        return X_transformed, feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed data array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        
        # Apply the same preprocessing steps as during fitting
        df_processed = df.drop(columns=[col for col in self.dropped_columns if col in df.columns])
        
        # Handle datetime columns the same way
        for col in self.column_types.get('datetime', []):
            if col in df_processed.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    
                    df_processed[f'{col}_year'] = df_processed[col].dt.year
                    df_processed[f'{col}_month'] = df_processed[col].dt.month
                    df_processed[f'{col}_day'] = df_processed[col].dt.day
                    df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
                    
                    df_processed = df_processed.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"Failed to process datetime column '{col}' during transform: {e}")
        
        X_transformed = self.preprocessor.transform(df_processed)
        logger.info(f"Transform complete. Shape: {X_transformed.shape}")
        return X_transformed
    
    def _get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Add numerical feature names
        if self.column_types['numerical']:
            feature_names.extend(self.column_types['numerical'])
        
        # Add categorical feature names
        if self.column_types['categorical']:
            try:
                # Get the categorical transformer
                cat_transformer = None
                for name, transformer, _ in self.preprocessor.transformers_:
                    if name == 'categorical':
                        cat_transformer = transformer
                        break
                
                if cat_transformer is not None:
                    encoder = cat_transformer.named_steps['encoder']
                    cat_features = encoder.get_feature_names_out(self.column_types['categorical'])
                    feature_names.extend(cat_features)
                else:
                    # Fallback if transformer not found
                    feature_names.extend(self.column_types['categorical'])
            except Exception as e:
                logger.warning(f"Could not get categorical feature names: {e}")
                feature_names.extend(self.column_types['categorical'])
        
        return feature_names
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the preprocessing steps performed.
        
        Returns:
            Dictionary containing preprocessing summary
        """
        return {
            'column_types': self.column_types,
            'dropped_columns': self.dropped_columns,
            'feature_names_out': self.feature_names_out,
            'preprocessing_summary': self.preprocessing_summary
        }
