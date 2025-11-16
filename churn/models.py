from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from preprocessing import get_preprocessor
import typing

def get_models(num_features, cat_features, random_state=45) ->typing.Dict[str, Pipeline]:
    preprocessor = get_preprocessor(num_features, cat_features)

    models = {'decision_tree': Pipeline([('preprocessor', preprocessor),
                                         ('clf', DecisionTreeClassifier(random_state=random_state))]),
              'random_forest': Pipeline([('preprocessor', preprocessor),
                                         ('clf', RandomForestClassifier(random_state=random_state))]),
              'gradient_boosting': Pipeline([('preprocessor', preprocessor),
                                             ('clf', GradientBoostingClassifier(random_state=random_state))]),
              'xgboost': Pipeline([('preprocessor', preprocessor),
                                   ('clf', XGBClassifier(random_state=random_state))]),
              }

    return models


if __name__ == '__main__':
    from cleaning import make_cleaned_dataset
    from preprocessing import get_feature_types

    data = make_cleaned_dataset()
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    num_features, cat_features = get_feature_types(X)
    models = get_models(num_features, cat_features)
    m = models['decision_tree']
    models[0].fit(X, y)
