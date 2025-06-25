"""
     layers.py
     Ζήτημα 2.1: Αρθρωτή Οπισθοδιάδοση (Modular Backpropagation)
  """
import numpy as np


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array of shape (N, Din) giving input data
    - w: A numpy array of shape (Din, Dout) giving weights
    - b: A numpy array of shape (Dout, ) giving biases

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = x.dot(w) + b
    ###########################################################################

    cache = (x, w, b)
    return out, cache


def fc_backward(grad_out, cache):
    """
    Computes the backward pass for a fully-connected layer.

    Inputs:
    - grad_out: Numpy array of shape (N, Dout) giving upstream gradients
    - cache: Tuple of:
      - x: A numpy array of shape (N, Din) giving input data
      - w: A numpy array of shape (Din, Dout) giving weights
      - b: A numpy array of shape (Dout , ) giving biases

    Returns a tuple of downstream gradients:
    - grad_x: A numpy array of shape (N, Din) of gradient with respect to x
    - grad_w: A numpy array of shape (Din, Dout) of gradient with respect to w
    - grad_b: A numpy array of shape (Dout, ) of gradient with respect to b
    """
    x, w, b = cache
    grad_x = grad_out.dot(w.T)
    grad_w = x.T.dot(grad_out)
    grad_b = np.sum(grad_out, axis=0, keepdims=True)
    ###########################################################################

    return grad_x, grad_w, grad_b


def relu_forward(x):
    """
    Computes the forward pass for the Rectified Linear Unit (ReLU) nonlinearity

    Input:
    - x: A numpy array of inputs, of any shape

    Returns a tuple of:
    - out: A numpy array of outputs, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    ###########################################################################

    cache = x
    return out, cache


def relu_backward(grad_out, cache):
    """
    Computes the backward pass for a Rectified Linear Unit (ReLU) nonlinearity

    Input:
    - grad_out: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - grad_x: Gradient with respect to x
    """
    x = cache
    grad_x = grad_out * (x > 0)
    ###########################################################################

    return grad_x


def l2_loss(x, y):
    """
    Computes the loss and gradient of L2 loss.

    loss = 0.5 * sum_i (x_i - y_i)**2 / N

    Inputs:
    - x: Input data, of shape (N, D)
    - y: Output data, of shape (N, D)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - grad_x: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    diff = x - y
    loss = 0.5 * np.sum(diff * diff) / N
    grad_x = diff / N
    return loss, grad_x


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax (cross-entropy) loss function.

    Inputs:
    - x: Numpy array of shape (N, C) giving predicted class scores, where
      x[i, c] gives the predicted score for class c on input sample i
    - y: Numpy array of shape (N,) giving ground-truth labels, where
      y[i] = c means that input sample i has ground truth label c, where
      0 <= c < C.

    Returns a tuple of:
    - loss: Scalar giving the loss
    - grad_x: Numpy array of shape (N, C) giving the gradient of the loss with
      with respect to x
    """
    # Number of training examples
    N = x.shape[0]

    # Shift the logits by subtracting the maximum value in each row for numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)

    # Compute the softmax scores
    exp_scores = np.exp(x_shifted)
    softmax_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute the cross-entropy loss
    correct_class_scores = softmax_scores[np.arange(N), y]
    loss = -np.sum(np.log(correct_class_scores)) / N

    # Compute the gradient of the loss with respect to x
    grad_x = softmax_scores.copy()
    grad_x[np.arange(N), y] -= 1
    grad_x /= N

    return loss, grad_x


def l2_regularization(w, reg):
    """
    Computes loss and gradient for L2 regularization of a weight matrix:

    loss = (reg / 2) * sum_i w_i^2

    Where the sum ranges over all elements of w.

    Inputs:
    - w: Numpy array of any shape
    - reg: float giving the regularization strength

    Returns:
    """
    loss, grad_w = (0.5 * reg *np.sum(w * w)), reg * w
    ###########################################################################

    return loss, grad_w


"""
    Επεξήγηση των Υλοποιήσεων:
    
1.	fc_forward:
o	Υπολογίζει την έξοδο ενός πλήρως συνδεδεμένου layer μέσω της συνάρτησης x.dot(w) + b.
o	Επιστρέφει την έξοδο και το cache (που περιέχει τις εισόδους x, w, b για χρήση στο backward pass).
2.	fc_backward:
o	Υπολογίζει τις παραγώγους σε σχέση με τις εισόδους x, τα βάρη w και τις προκατα-λήψεις b χρησιμοποιώντας 
     τις upstream παραγώγους grad_out και το cache.
o	Επιστρέφει τις παραγώγους grad_x, grad_w, grad_b.
3.	relu_forward:
o	Υπολογίζει την έξοδο της συνάρτησης ReLU με την εφαρμογή της np.maximum(0, x).
o	Επιστρέφει την έξοδο και το cache (που περιέχει την είσοδο x).
4.	relu_backward:
o	Υπολογίζει τις παραγώγους της συνάρτησης ReLU σε σχέση με την είσοδο x χρησι-μοποιώντας
    τις upstream παραγώγους grad_out και το cache.
o	Επιστρέφει την παράγωγο grad_x που λαμβάνεται πολλαπλασιάζοντας το grad_out με τη δυαδική μάσκα (x > 0).
5.	l2_regularization:
o	Υπολογίζει την απώλεια L2 και την παράγωγο των βαρών w ως προς την απώλεια χρησιμοποιώντας την κανονικοποίηση reg.
o	Επιστρέφει την απώλεια και την παράγωγο grad_w.

Εκτέλεση των Ελέγχων Παραγώγων

Με αυτές τις υλοποιήσεις, μπορείτε να εκτελέσετε το αρχείο gradcheck_layers.py για να ελέγξετε την ορθότητα των παραγώγων.
Σιγουρευτείτε ότι οι διαφορές παραγώγων που αναφέρονται είναι μικρότερες από το 10^-9, όπως απαιτείται.

Αποτελέσματα Gradient Check

Έτρεξα ελέγχους παραγώγων (gradient checks) για τις υλοποιήσεις των forward και backward passes για πλήρως 
συνδεδεμένα layers, ReLU, softmax loss και L2 regularization. 
Οι διαφορές μεταξύ των αριθμητικών και των υπολογισμένων παραγώγων είναι εξαιρετικά μικρές,
όπως φαίνεται από τα αποτελέσματα.

Συμπέρασμα

Οι πολύ μικρές διαφορές (κοντά στο μηδέν) μεταξύ των αριθμητικών παραγώγων και των υπολογισμένων τιμών δείχνουν ότι
 οι υλοποιήσεις μου για τα fully connected layers, ReLU, softmax loss και L2 regularization είναι σωστές.
Αυτό επιβεβαιώνει ότι οι παράγωγοι υπολογίζονται με ακρίβεια, διασφαλίζοντας την ορθότητα των backward passes.
Σημείωση
Λάβετε υπόψη ότι ο αριθμητικός έλεγχος παραγώγων δεν ελέγχει αν το forward pass έχει υλοποιηθεί σωστά.
 Ελέγχει μόνο αν το backward pass υπολογίζει σωστά τις παραγώγους του forward pass. 
 Για να διασφαλίσουμε την ορθότητα του forward pass, θα πρέπει να γίνουν ξεχωριστοί έλεγχοι και δοκιμές για την ακρίβεια
  των εξόδων του forward pass.


Τα αποτελεσματα :
C:\Users\mayts\Desktop\pythonProject\.venv\Scripts\python.exe 
C:\Users\mayts\Desktop\pythonProject\assignment2\gradcheck_layers.py 

Running numeric gradient check for fc
  grad_x difference:  9.215908036708242e-10
  grad_w difference:  4.332205705281922e-10
  grad_b difference:  3.039564155926655e-10
Running numeric gradient check for relu
  grad_x difference:  4.042588486186105e-11
Running numeric gradient check for softmax loss
  grad_x difference:  2.1950857451158434e-10
Running numeric gradient check for L2 regularization
  grad_w difference:  8.439052928688184e-11
"""