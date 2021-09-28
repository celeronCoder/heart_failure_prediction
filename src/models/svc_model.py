from utils.globals import X_train, y_train

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class svc_model:
    def define_param_range():
        """
        defines parameter range
        """
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}

        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
        grid.fit(X_train, y_train)
        print(grid.best_estimator_)

    def get_model(self):
        svc = SVC(C=10, gamma=0.0001)
        svc.fit(X_train, y_train)

        return svc

    def get_prediction(self, X_test):
        model = self.get_model()
        prediction = model.predict(X_test)

        return prediction
