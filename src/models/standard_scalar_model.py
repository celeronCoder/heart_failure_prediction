from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils.globals import X_train, y_train


class standard_scalar_model:
    def get_model(self):
        lr_clf_pipline = make_pipeline(StandardScaler(), LogisticRegression())
        lr_clf_pipline.fit(X_train, y_train)

        return lr_clf_pipline

    def get_prediction(self, X_test):
        model = self.get_model()
        prediction = model.predict(X_test)

        return prediction
