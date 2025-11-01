The files in this repository containn the code for the paper "Gaussian Processes simplify Differential Equations" by Jonghyeon Lee, Boumediene Hamzi, Yannis Kevrekidis and Houman Owhadi.

The notebook 'Brusselator' has the code for discovering the Poincaré normal form for the Brusselator; it is easy to adapt the code by changing $A$ and $B$ (as well as $\mu = \frac{B-(A^2+1)}{\sqrt{4A^2-(B-A^2-1)^2}}$).

Cole Hopf transformation demonstrates the effect of learning the Matérn kernel parameter on the accuracy of the recovered Cole-Hopf transformation between a single pair of initial conditions of the Burgers and heat equations; this can be modified to multiple initial conditions or solutions of Burgers at multiple time points.

Differential equation, 1D example contains the code for the second example in our paper involving two 1st order PDEs.
