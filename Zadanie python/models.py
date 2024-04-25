from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



def logistic_regression (X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression(random_state=1993)

    log_reg.fit(X_train, y_train)

    y_pred_log = log_reg.predict(X_test)

    accuracy = str(round(accuracy_score(y_test, y_pred_log)*100,2))+'%'
    recall= str(round(recall_score(y_test, y_pred_log)*100,0))+'%'
    print (f'Czułość modelu {recall}; Dokładność modelu {accuracy}')

    print(classification_report(y_test, y_pred_log))
    return y_pred_log


def conf_matrix (y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Klasa 0', 'Klasa 1'], yticklabels=['Klasa 0', 'Klasa 1'])
    plt.xlabel('Predykowane etykiety')
    plt.ylabel('Rzeczywiste etykiety')
    plt.title('Confusion Matrix')
    plt.show()


def random_forrest_class (X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100 ,random_state=1993)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = str(round(accuracy_score(y_test, y_pred_rf)*100,2))+'%'
    recall_rf= str(round(recall_score(y_test, y_pred_rf)*100,0))+'%'
    print (f'Czułość modelu {recall_rf}; Dokładność modelu {accuracy_rf}')
    print(classification_report(y_test, y_pred_rf))
    return y_pred_rf