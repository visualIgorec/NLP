from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

def metric(
    trained_model,
    test_features,
    test_targets,
    model_name
    ):

    ### Compute Confusion matrix, accuracy and F1 Score ###

    # accuracy
    print(f"Accuracy: Decision {model_name} is {round(trained_model.score(test_features, test_targets),3)}")
    y_predicted = trained_model.predict(test_features)

    # f1-score on the test data
    fscore = f1_score(test_targets, y_predicted, average='macro')
    print(f'f1-score: {model_name} is {round(fscore, 3)}')

    # Compute confusion matrix
    cm = confusion_matrix(test_targets, y_predicted, labels=trained_model.classes_)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = trained_model.classes_)
    disp.plot()
    plt.xticks(rotation=45)
    plt.title(f'Normalized confusion matrix for {model_name}')
    plt.show()
