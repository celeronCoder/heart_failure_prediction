from utils.globals import X_train, X_test, y_test, y_train
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV


def randomized_search(params, runs=20, clf=DecisionTreeClassifier(random_state=2)):
    rand_clf = RandomizedSearchCV(
        clf, params, n_iter=runs, cv=5, n_jobs=1, random_state=2)

    rand_clf.fit(X_train, y_train)

    best_model = rand_clf.best_estimator_
    best_score = rand_clf.best_score_

    print("Training Score: {:3f}".format(best_score))
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print('Test score: {:3f}'.format(accuracy))
