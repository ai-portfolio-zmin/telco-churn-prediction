from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             precision_recall_curve,
                             f1_score,
                             average_precision_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             classification_report)
import matplotlib.pyplot as plt
import pandas as pd

def eval_model(y_test, y_prob, y_pred, plot =False):
    auc_score = roc_auc_score(y_test, y_pred)
    tpr, fnr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test,y_prob)
    average_pre_score = average_precision_score(y_test, y_prob)
    c_matrix = confusion_matrix(y_test, y_pred)
    f_score = f1_score(y_test,y_pred)
    class_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))

    if plot:
        plt.figure()
        plt.plot(fnr, tpr)
        plt.xlabel('False negative rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.show()

        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Recall precision curve')
        plt.show()

        ConfusionMatrixDisplay(c_matrix).plot()

    print(class_report)
    print(f'auc_score: {auc_score}')
    print(f'average precision scoree: {average_pre_score}')

    return {'auc_score':auc_score,
            'f1_score': f_score,
            'average_precision_score':average_pre_score,
            }


