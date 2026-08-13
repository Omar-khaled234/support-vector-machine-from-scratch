"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    # TODO: rescale each column of x to have mean 0 and std 1 (leave zero-std columns alone).
    mean = np.mean(x , axis=0)
    std = np.std(x, axis=0)
    std[std==0]=1
    z = (x-mean)/std
    return z

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    b = 0.0
    w = np.zeros(n_features)
    return {'w' : w , 'b' : b}

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    w= params["w"]
    b= params["b"]
    pred = x @ w + b
    return pred

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    pred = []
    for score in scores:
        if score >= 0:
            pred.append(1)
        else:
            pred.append(-1)
    return pred

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    hinge_loss = np.maximum(0 , (1-y*score))
    return hinge_loss

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    scores = compute_scores(x,params)
    loss = hinge_loss_example( scores, y)
    w = params['w']
    l2 = np.mean(loss) + reg_lambda * (w @ w)
    return l2

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # TODO: compute the gradient of the SVM objective wrt params['w'] and params['b'].
    scores = compute_scores(x, params)
    margin = hinge_loss_example(scores,y)
    m = len(y)
    w = params['w']
    dw = 2* reg_lambda * w
    db = 0
    for i in range(m):
        if margin[i]>0:
            dw+= -(y[i] * x[i])/m
            db += -y[i]/m
    return {'dw' : dw , 'db' : db}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    w = params['w'].copy()
    b = params['b']
    w -= learning_rate*grads['dw']
    b -= learning_rate*grads['db']
    return {'w' : w , 'b' : b}

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    # TODO: fit a linear SVM by repeatedly updating parameters over n_epochs passes.
    params = initialize_parameters(x.shape[1])
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    return params

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

