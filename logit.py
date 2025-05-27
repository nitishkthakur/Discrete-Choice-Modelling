import numpy as np
from scipy import optimize


class MultinomialConditionalLogit:
    def __init__(self, n_choices, kind='multinomial'):
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

        # Calculate the numerator and denominator for the softmax function
        exp_linear_combination = np.exp(linear_combination)
        sum_exp = np.sum(exp_linear_combination, axis=1)

