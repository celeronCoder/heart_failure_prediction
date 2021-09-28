from utils.globals import X_train, y_train
from sklearn.ensemble import GradientBoostingClassifier


class gradient_boost_classifier_model:
    def get_model(self):
        gbdt = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0)
        gbdt.fit(X_train, y_train)

        return gbdt

    def get_prediction(self, X_test):
        gbdt = self.get_model()
        pred_gbdt = gbdt.predict(X_test)

        return pred_gbdt
