# Bayesian Knowledge Tracing Models
# Extention of hmmlearn
#
# Author: Caitlyn Clabaugh <ceclabaugh@gmail.com>

import numpy as np
from .hmm import MultinomialHMM

__all__ = ["BKT"]

class BKT(MultinomialHMM):
    """Standard Bayesian Knowledge Tracing

    Parameters
    ----------

    p_init : float, optional
        Prior probability that a student already knows a skill.

    p_transit : float, optional
        Probability that a student's knowledge of a skill transitions
        from 'unknown' to 'known' after an opportunity to apply it.

    p_slip : float, optional
        Probability that a student makes a mistake when applying a
        known skill.

    p_guess : float, optional
        Probability that a student answers correctly, by chance, though 
        the skill is unknown.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    emissionprob\_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmlearn.bkt import BKT
    >>> BKT()
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    BKT(p_init=0.5,...
	"""
    def __init__(self, p_init=0.0, p_transit=0.0, p_slip=0.25, p_guess=0.25,
        tol=1e-2, verbose=False):

        self.p_init = p_init
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess

        # Initialize transitions matrix
        # Force probability of forgetting to 0
        # [[from known to known, from known to unknown],
        #  [from unknown to known, from unknown to unknown]]
        transmat_prior =  np.array([[1.0, 0.0],
                                    [self.p_transit, (1.0-self.p_transit)]])

         # Implement as MultinomialHMM with two states: {known, unknown}
        MultinomialHMM.__init__(self, n_components=2,
                                transmat_prior=transmat_prior,
                                tol=tol, verbose=verbose,
                                params="st",    # Do not update emissions matrix, use guess & slip
                                init_params="") # Custom priors, transitions, and emissions matrices

        # Override priors matrix
        self.startprob_ = np.array([self.p_init, (1.0-self.p_init)])

        # Override emissions matrix with slip and guess probabilities
        # [[ P(known | right), P(known | wrong) ],
        #  [ P(unknown | right), P(unknown | wrong) ]]
        self.emissionprob_ = np.array([[(1.0-self.p_slip), self.p_slip],
                                        [self.p_guess, (1.0-self.p_guess)]])


    #def _do_mstep(self, stats):
        #super(MultinomialHMM, self)._do_mstep(stats)
        
        # Override emissions matrix with slip and guess probabilities
        #self.emissionprob_ = [ [(1.0-self.p_slip), self.p_slip], [self.p_guess, (1.0-self.p_guess)] ]

