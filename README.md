# Learning to relax

This repository contains code to reproduce the empirical results in [our paper](https://arxiv.org/abs/2310.02246) on learning-augmented linear system solvers, specifically successive over-relaxation (SOR) and Symmetric SOR-preconditioned conjugate gradient (CG).
It also includes new experiments on accelerating a simulation of the heat equation in two dimensions, as reported [here](https://openreview.net/forum?id=bCssNn4ZPe).

## Script descriptions:
<tt>asymptotic.m</tt>: computes bounds on the performance of SOR across different relaxation parameters  
<tt>cg.m</tt>: computes bounds on the performance of CG across different relaxation parameters  
<tt>comparators.m</tt>: computes average SOR performance across distributions over linear systems  
<tt>contextual.m</tt>: evaluates contextual bandit algorithms for setting the SOR relaxation parameter  
<tt>h2d.m</tt>: evaluates different approaches for accelerating a heat equation simulation  
<tt>learning.m</tt>: compares several fixed settings of the SOR relaxation parameter with a bandit algorithm  

## Reference:
Mikhail Khodak, Edmond Chow, Maria-Florina Balcan, Ameet Talwalkar. *Learning to relax: Setting solver parameters across a sequence of linear system instances*. 2023.
