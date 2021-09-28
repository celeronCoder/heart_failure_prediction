from utils.randomized_search import randomized_search
from utils.globals import X_train, y_train
from sklearn.tree import DecisionTreeClassifier


class decision_tree_classifier_model:
    def run_random_search(self):
        randomized_search(params={'criterion': ['entropy', 'gini'],
                                  'splitter': ['random', 'best'],
                                  #   deepcode ignore DuplicateKey: <please specify a reason of ignoring this>
                                  'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01],
                                  'min_samples_split': [2, 3, 4, 5, 6, 8, 10],
                                  'min_samples_leaf': [1, 0.01, 0.02, 0.03, 0.04],
                                  'min_impurity_decrease': [0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
                                  'max_leaf_nodes': [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
                                  'max_features': ['auto', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                                  'max_depth': [None, 2, 4, 6, 8],
                                  #   deepcode ignore DuplicateKey: <please specify a reason of ignoring this>
                                  'min_weight_fraction_leaf': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
                                  })

    def get_model(self):
        ds_clf = DecisionTreeClassifier(max_depth=8, max_features=0.9, max_leaf_nodes=30, min_impurity_decrease=0.05, min_samples_leaf=0.02,
                                        min_samples_split=10, min_weight_fraction_leaf=0.005,
                                        random_state=2, splitter='random')
        ds_clf.fit(X_train, y_train)

        return ds_clf

    def get_prediction(self, X_test):
        model = self.get_model()
        prediction = model.predict(X_test)

        return prediction
