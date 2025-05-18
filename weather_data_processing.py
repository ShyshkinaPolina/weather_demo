import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any

def drop_na_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop rows with NA values in the specified columns.

    Args:
        df (pd.DataFrame): The raw dataframe.
        columns (list): List of columns to check for NA values.

    Returns:
        pd.DataFrame: DataFrame with NA values dropped.
    """
    return df.dropna(subset=columns)

def split_data_by_year(df: pd.DataFrame, year_col: str) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training, validation, and test sets based on the year.

    Args:
        df (pd.DataFrame): The raw dataframe.
        year_col (str): The column containing year information.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train, validation, and test dataframes.
    """
    year = pd.to_datetime(df[year_col]).dt.year
    train_df = df[year < 2015]
    val_df = df[year == 2015]
    test_df = df[year > 2015]
    return {'train': train_df, 'val': val_df, 'test': test_df}

def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training, validation, and test sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train, validation, and test dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train, val, and test sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data

def impute_missing_values(data: Dict[str, Any], imputer, numeric_cols: list) -> None:
    """
    Impute missing numerical values using the mean strategy.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        numeric_cols (list): List of numerical columns.
    """
    data[numeric_cols] = imputer.transform(data[numeric_cols])

def scale_numeric_features(data: Dict[str, Any], scaler, numeric_cols: list) -> None:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        numeric_cols (list): List of numerical columns.
    """
    data[numeric_cols] = scaler.transform(data[numeric_cols])

def encode_categorical_features(data: Dict[str, Any], encoder, categorical_cols: list) -> None:
    """
    One-hot encode categorical features.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        categorical_cols (list): List of categorical columns.
    """
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded = encoder.transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=data.index)
    inputs = data.drop(columns=categorical_cols)
    return pd.concat([inputs, encoded_df], axis=1)


def preprocess_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.

    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets for train, val, and test sets.
    """
    raw_df = drop_na_values(raw_df, ['RainToday', 'RainTomorrow'])
    split_dfs = split_data_by_year(raw_df, 'Date')
    input_cols = list(raw_df.columns)[1:-1]
    target_col = 'RainTomorrow'
    data = create_inputs_targets(split_dfs, input_cols, target_col)
    
    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()

    impute_missing_values(data, numeric_cols)
    scale_numeric_features(data, numeric_cols)
    encode_categorical_features(data, categorical_cols)

    # Extract X_train, X_val, X_test
    X_train = data['train_inputs'][numeric_cols + data['encoded_cols']]
    X_val = data['val_inputs'][numeric_cols + data['encoded_cols']]
    X_test = data['test_inputs'][numeric_cols + data['encoded_cols']]

    return {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
        'test_X': X_test,
        'test_y': data['test_targets'],
    }

def preprocess_new_data(input_df: pd.DataFrame, input_cols: list, encoder, scaler, imputer, scale_numeric: bool = True) -> pd.DataFrame:
    """
    Preprocess new data using the provided scaler and encoder.

    Args:
        new_df (pd.DataFrame): The new dataframe.
        input_cols (list): List of input columns.
        scaler (MinMaxScaler): Fitted scaler.
        encoder (OneHotEncoder): Fitted encoder.
        scale_numeric (bool): Option to scale numerical features.

    Returns:
        pd.DataFrame: Processed inputs for the new data.
    """

    numeric_cols = input_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = input_df.select_dtypes('object').columns.tolist()
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    return X_input