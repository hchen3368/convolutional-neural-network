from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import pandas as pd

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    # compute scores
    scores = np.matmul(X, W)
    scores = scores - np.max(scores, axis=1).reshape(-1,1)
    # compute the scores at the correct labels S_y
    # by first encoding y as dummy variables
    y_dummy = pd.get_dummies(pd.DataFrame(y),columns=[0]).values
    scores_y = np.sum(scores * y_dummy, axis=1)
    # cache exp of scores for later
    exp_scores = np.exp(scores)
    # compute softmax loss with L2 regularization
    loss = np.sum(np.log(np.sum(exp_scores,axis=1)) - scores_y) / num_train + 0.5 * reg * np.sum(W*W)

    # normalize the rows of exp_scores to get probability
    prob = exp_scores / np.sum(exp_scores,axis=1).reshape(-1,1)
    # compute gradient
    dW = np.matmul(X.T, prob - y_dummy) /num_train + reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    # compute scores
    scores = np.matmul(X, W)
    scores = scores - np.max(scores, axis=1).reshape(-1,1)
    # compute the scores at the correct labels S_y
    # by first encoding y as dummy variables
    y_dummy = pd.get_dummies(pd.DataFrame(y),columns=[0]).values
    scores_y = np.sum(scores * y_dummy, axis=1)
    # cache exp of scores for later
    exp_scores = np.exp(scores)
    # compute softmax loss with L2 regularization
    loss = np.sum(np.log(np.sum(exp_scores,axis=1)) - scores_y) / num_train + 0.5 * reg * np.sum(W*W)

    # normalize the rows of exp_scores to get probability
    prob = exp_scores / np.sum(exp_scores,axis=1).reshape(-1,1)
    # compute gradient
    dW = np.matmul(X.T, prob - y_dummy) /num_train + reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
