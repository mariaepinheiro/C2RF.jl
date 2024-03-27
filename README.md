#Mixed-Integer Linear Optimization for Cardinality- Constrained Random Forest

This repository contains all the information needed to reproduce the results for the paper __________ authored by Jan Pablo Burgard, Maria Eduarda Pinheiro, Martin Schmidt.

# Abstract

# Preliminary needed tools

- Gurobi: https://juliapackages.com/p/gurobi
- C2RF.jl package: https://github.com/mariaepinheiro/C2RF.jl

 # How to use the Cardinality-Constrained Random Forest
 C2RF and pC2RFmain return the final classification, $\alpha$ (the weight of each tree) in the final classification, the objective function value and the GAP (If Gurobi is used).
 
 Both approaches need the following input values:
 
 ## Arguments
 - $R \in \{-1,1\}^{t \times m}$ : The Matrix of classification of $m$ unlabeled points in $t$ trees.
 - $\lambda$: how many unlabeled points belong to the positive class
 - $\ell$ and $u$ : lower and upper bound for each entrance of $\alpha$.
 - maxtime: time limit
 - solver: By default we use solver=1, which means we are using Gurobi. For that, it is necessary a Gurobi license. If choose any different value, SCIP is used, however no branching priority is considered.

 - 

 
