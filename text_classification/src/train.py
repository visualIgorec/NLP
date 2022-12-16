from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def train(
    train_features,
    train_targets,
    ):

    ### training models for text classification ###


    # Decision Tree
    print('Decision Tree is started')
    model_dt = DecisionTreeClassifier()
    model_dt.fit(train_features, train_targets)

    # Random Forest
    print('Random Forest is started')
    model_rf = RandomForestClassifier()
    model_rf.fit(train_features, train_targets)

    # Logistic Regression
    print('Logistic Regression is started')
    model_lr = LogisticRegression(solver='newton-cg')
    model_lr.fit(train_features, train_targets)

    print('done!')
    return {
            'Decision_Tree': model_dt,
            'Randon_Forest': model_rf,
            'Logistic Regression': model_lr
            }



