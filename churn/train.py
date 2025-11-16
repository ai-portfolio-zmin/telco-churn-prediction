import pandas as pd

from cleaning import make_cleaned_dataset
from preprocessing import get_feature_types
from models import get_models
from sklearn.model_selection import train_test_split
from metrics import eval_model

def train_all_models(plot = False):
    data = make_cleaned_dataset()
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    num_features, cat_features = get_feature_types(X)
    models = get_models(num_features, cat_features)

    results = []
    for name, model in models.items():
        model.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        y_predict_prob = model.predict_proba(X_test)[:,1]
        result = eval_model(y_test,y_predict_prob, y_predict, plot=plot)
        results.append(result)

    result_df = pd.DataFrame(results)
    print(result_df)


if __name__ == '__main__':

    train_all_models()



