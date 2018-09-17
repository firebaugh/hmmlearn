"""
Standard BKT on Toy Dataset
---------------------------

Toy dataset is...
"""
import numpy as np
from hmmlearn.bkt import BKT

# reshape because our data has a single feature (performance)
X = np.array([0,0,0,1,0,1,1,1,1]).reshape(-1,1)


###############################################################################
# Run BKT
print("fitting BKT ...", end="")

# Make an HMM instance and execute fit
model = BKT().fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("done")

###############################################################################
# Print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
print()