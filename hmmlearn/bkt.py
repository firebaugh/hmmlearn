# Bayesian Knowledge Tracing Models
# Extention of hmmlearn
#
# Author: Caitlyn Clabaugh <ceclabaugh@gmail.com>

import numpy as np
from .hmm import MultinomialHMM
from .utils import normalize

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

        # Implement as MultinomialHMM with two states: {known, unknown}
        MultinomialHMM.__init__(self, n_components=2,
                                tol=tol, verbose=verbose,
                                params="st", # NO UPDATE of emissions (i.e., guess & slip)
                                init_params="")


    def _init(self, X, lengths=None):
        self._check_input(X)

        super(BKT, self)._init(X, lengths=lengths)

        # Override priors matrix
        self.startprob_ = np.array([self.p_init, (1.0-self.p_init)])

        # Override transitions matrix
        # Force probability of forgetting to 0
        # [[from known to known, from known to unknown],
        #  [from unknown to known, from unknown to unknown]]
        self.transmat_ =  np.matrix([[1.0, 0.0],
                                    [self.p_transit, (1.0-self.p_transit)]])

        # Override emissions matrix with slip and guess probabilities
        # [[ P(known | right), P(known | wrong) ],
        #  [ P(unknown | right), P(unknown | wrong) ]]
        self.emissionprob_ = np.matrix([[(1.0-self.p_slip), self.p_slip],
                                        [self.p_guess, (1.0-self.p_guess)]])


    def _check_input(self, X):
        """Check that the format of ``X`` is a column vector of
        single non-negative values, representing each observation
        of performance.

        For example ``[[0]; [0]; [1]; [1]]``.
        """
        symbols = np.concatenate(X)
        if len(symbols) == 1:       # not enough data
            raise ValueError("expected at least 1 observation "
                             "but none found.")
        elif (symbols < 0).any():   # contains negative integers
            raise ValueError("expected non-negative features "
                             "for each observation.")
        elif X.shape[1] > 1:        # contains to many features
            raise ValueError("expected only 1 feature but got {0} "
                             "for each observation.".format(X.shape[1]))
        else:
            return True


    def _do_mstep(self, stats):
        # Update probability of prior knowledge
        startprob_ = self.startprob_prior - 1.0 + stats['start']
        self.startprob_ = np.where(self.startprob_ == 0.0, self.startprob_, startprob_)
        normalize(self.startprob_)

        # Update probabilities of transitioning from unknown to known
        transmat_ = self.transmat_prior - 1.0 + stats['trans']
        self.transmat_ = np.where(self.transmat_ == 0.0, self.transmat_, transmat_)
        normalize(self.transmat_, axis=1)

        # Assumes ZERO probability of forgetting
        self.transmat_[0] = [1.0, 0.0]

