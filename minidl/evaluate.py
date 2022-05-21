import numpy as np

def predict(model, x_test, y_test=None):
    model.network.forward(x_test.T)
    preds = np.argmax(model.network.preds,axis=0)
    if y_test is not None:
        acc = sum(preds==y_test)/len(y_test)
        return acc, preds
    return None, preds