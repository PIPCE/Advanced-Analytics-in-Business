import numpy as np
from sklearn.metrics import confusion_matrix

class PerformanceMetrics:

    def cost_with_algorithm(cost_matrix, predictions):

        cost_vector = np.where(predictions == 0, cost_matrix.iloc[:, 0], cost_matrix.iloc[:, 1])
        cost = np.sum(cost_vector)

        return cost

    def cost_without_alg(cost_matrix,clas):

        cost = np.sum(cost_matrix.iloc[:, 0].dot(clas))

        return cost

    def accuracy(predictions, clas):

        return 1-sum(abs(predictions - clas))/len(clas)

    def confusion_matrix(predictions, clas):

        CM = confusion_matrix(clas, predictions)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        if (TP+FP == 0) or (TP+FN == 0):
            F_measure = np.nan
        else:
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            F_measure = 2*precision*recall/(precision+recall)

        return F_measure

