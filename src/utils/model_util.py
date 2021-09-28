import joblib


class model_util:
    def save_model(model, name: str):
        joblib.dump(model, f'{name}.pkl')

    def test_model(model_path: str, X_test):
        model = joblib.load(model_path)
        print(model.predict(X_test))
