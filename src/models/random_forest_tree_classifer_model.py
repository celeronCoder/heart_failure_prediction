from utils.globals import X_train, y_train
from utils.randomized_search import randomized_search
from sklearn.ensemble import RandomForestClassifier


class random_forest_tree_classifer_model:
    def run_random_search(self):
        randomized_search(params={
            'min_samples_leaf': [1, 2, 4, 6, 8, 10, 20, 30],
            'min_impurity_decrease': [0.0, 0.01, 0.05, 0.10, 0.15, 0.2],
            'max_features': ['auto', 0.8, 0.7, 0.6, 0.5, 0.4],
            'max_depth': [None, 2, 4, 6, 8, 10, 20],
        }, clf=RandomForestClassifier(random_state=2))

    def get_model(self):
        rf_clf = RandomForestClassifier(
            max_depth=2, max_features=0.5, min_impurity_decrease=0.01, min_samples_leaf=10, random_state=2)
        rf_clf.fit(X_train, y_train)

        return rf_clf

    def get_prediction(self, X_test):
        model = self.get_model()
        prediction = model.predict(X_test)

        return prediction
