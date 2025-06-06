import numpy as np
# from scipy import optimize


class MultinomialConditionalLogit:
    def __init__(self, n_choices, kind="multinomial"):
        self.n_choices = n_choices
        self.beta = None
        self.kind = kind

    def beta_transform(self, beta):
        """
        Transform the beta coefficients which are optimized by the optimizer to what the model sees
        beta will always have the shape of (n_features + 1, n_choices).
        """
        if len(beta.shape) == 1:
            return beta.reshape(-1, 1)

        else:
            beta = beta.reshape(-1, self.n_choices)
            return beta

    @staticmethod
    def softmax(logits, temperature = 1.0):
        """Compute the softmax with temprature"""

        # First, Scale with temperature/
        logits = logits / temperature
        
        max_logits = np.max(logits, keepdims = True)

        # Subtract the max logit
        logits -= max_logits

        # Now compute the softmax
        exp_logits = np.exp(logits)

        # Normalize the logits to get probabilities
        softmax_values = exp_logits/np.sum(exp_logits, axis = -1, keepdims = True)
        return softmax_values

    def likelihood(self, beta, X, y):
        """
        Calculate the likelihood of the multinomial logit model.

        Parameters:
        beta : array-like
            Coefficients for the model.
        X : array-like
            Design matrix (features).
        y : array-like
            Response variable (choices).

        Returns:
        float
            Negative log-likelihood value.
        """
        linear_combination = np.dot(X, beta)

        # Compute logit
        softmax_values = MultinomialConditionalLogit.softmax(linear_combination) #noqa: F841

        # Compute the log likelihood


        
