"""
    two_layer_net.py
     Ζήτημα 2.2: Υλοποίηση Νευρωνικού Δικτύου 2 επιπέδων (2-layer NN)
  """
import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """

    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        #######################################################################
        # TODO: Initialize the weights and biases of a two-layer network.     #
        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim) * weight_scale,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, num_classes) * weight_scale,
            'b2': np.zeros(num_classes)
        }

    def parameters(self):
        params = self.params
        #######################################################################
        # TODO: Build a dict of all learnable parameters of this model.       #

        return params

    def forward(self, X):
        scores, cache = None, None
        #######################################################################
        # TODO: Implement the forward pass to compute classification scores   #

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden_layer, cache_hidden = fc_forward(X, W1, b1)
        relu_layer, cache_relu = fc_forward(hidden_layer, W2, b2)
        scores, cache_scores = fc_forward(relu_layer)

        cache = (cache_hidden, cache_relu, cache_scores)

        return scores, cache

    def backward(self, grad_scores, cache):
        grads = None
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #

        cache_hidden, cache_relu, cache_scores = cache

        # Backward pass: compute gradients
        grad_relu, grad_W2, grad_b2 = fc_backward(grad_scores, cache_scores)
        grad_hidden = relu_backward(grad_relu, cache_relu)
        grad_X, grad_W1, grad_b1 = fc_backward(grad_hidden, cache_hidden)

        grads = {
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2
        }
        return grads

"""

Ολοκλήρωσα την υλοποίηση ενός νευρωνικού δικτύου δύο επιπέδων (TwoLayerNet) χρησιμοποιώντας τις modular συναρτήσεις forward 
και backward που υλοποιήθηκαν στο ζήτημα 2.1. Η κλάση TwoLayerNet περιλαμβάνει τις μεθόδους __init__, parameters, forward, και backward. 

Για τον αριθμητικό έλεγχο των παραγώγων, εκτέλεσα τη ρουτίνα gradcheck_classifier.py, 
η οποία περιλαμβάνει την κλήση του gradient check τόσο για τον γραμμικό ταξινομητή όσο και για το TwoLayerNet. 

Τα αποτελέσματα που πήρα ήταν τα εξής:

Max diff for grad_W: 2.4935609133081016e-13
Max diff for grad_b: 2.922107000813412e-13

Οι μικρές αυτές διαφορές επιβεβαιώνουν ότι η υλοποίησή μου είναι σωστή και ότι οι παραγώγοι υπολογίζονται σωστά στην backward συνάρτηση.

  
"""