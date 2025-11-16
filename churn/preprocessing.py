import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_feature_types(X: pd.DataFrame):
    num_features = X.select_dtypes(include=['int64','float64']).columns.to_list()
    cat_features = X.select_dtypes(include=['object', 'bool','category']).columns.to_list()
    return num_features, cat_features

def get_preprocessor(num_features, cat_features):
    preprocessor = ColumnTransformer(
        transformers= [
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ]
    )
    return preprocessor

if __name__ == '__main__':
    from cleaning import make_cleaned_dataset
    cleaned_data = make_cleaned_dataset()
    num_features, cat_features = get_feature_types(cleaned_data)
    preprocessor = get_preprocessor(num_features, cat_features)

