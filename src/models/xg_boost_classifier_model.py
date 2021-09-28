from utils.globals import X_test, X_train, y_test, y_train
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt


class xg_boost_classifier_model:
    def get_model(self):
        xgb1 = XGBClassifier(colsample_bytree=1.0, learning_rate=0.1,
                             max_depth=4, n_estimators=400, subsample=1.0)

        eval_set = [(X_test, y_test)]

        xgb1.fit(X_train, y_train, early_stopping_rounds=10,
                 eval_metric="logloss", eval_set=eval_set, verbose=False)

        # prediction = xgb1.predict(X_test)

        return xgb1

    def get_prediction(self, X_test):
        model = self.get_model()
        prediction = model.predict(X_test)

        return prediction

    def plot_feature_importance(xgb):
        plot_importance(xgb)
        plt.savefig("feature_importance.png")
