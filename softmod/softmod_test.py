import numpy as np
import pandas as pd
from softmod import SoftMod
from sklearn.metrics import normalized_mutual_info_score
import os

os.system("rm -rf log/karate")

elist = pd.read_pickle("data/karate.pkl")
ans = pd.read_pickle("data/karate_com.pkl")

model = SoftMod(2, elist, learning_rate=0.01, lambda_phi=0.1)


mod = model.optimize(max_iter=1000, logdir="log/karate",
                    stop_threshold=0.000001)
com = model.get_hard_communirty()
scom = model.get_soft_community()
nmi = normalized_mutual_info_score(ans, com)

Theta = [[0.0,0.0] for n in range(34)]
for n, c in enumerate(ans):
    Theta[n][c] = 1.0

mod2 = model.calculate_mod_for_theta(Theta)
