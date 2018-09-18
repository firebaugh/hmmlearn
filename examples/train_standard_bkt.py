"""
Standard BKT on Toy Dataset
---------------------------

Toy dataset is...
"""
import numpy as np
from hmmlearn.bkt import BKT

# reshape because our data has a single feature (performance)
X = np.array([[1,1,0,0,0]]).T


###############################################################################
# Run BKT
print("fitting BKT ...", end="")

# Make an BKT instance and execute fit
model = BKT(p_init=0.5, p_transit=0.5, p_slip=0.05, p_guess=0.05).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("done")

###############################################################################
# Print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
print()