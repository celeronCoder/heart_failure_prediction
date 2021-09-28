from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def evaluating_model(y_test, y_pred):
    """
    Funtions for evaluating model
    """
    print("Accuracy Score:- ", accuracy_score(y_test, y_pred))
    print("Precision Score:- ", precision_score(y_test, y_pred))
    print("Recall Score:- ", recall_score(y_test, y_pred))
    print("Confusion Matrix:- \n", confusion_matrix(y_test, y_pred))
