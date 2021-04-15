Authors: Chataigner, Cousin, Crepey, Dixon, and Gueye. 

If this code is used for research purposes, please cite as:

M. Chataigner, A. Cousin, S. Crepey, M.F. Dixon, and D. Gueye, [Beyond Surrogate Modeling: Learning the Local Volatility Via Shape Constraints](http://mypages.iit.edu/~mdixon7/preprints/local_vol.pdf), working paper, 2020.
M. Chataigner, S. Crepey and M. Dixon, [Deep Local Volatility](https://www.mdpi.com/2227-9091/8/3/82), Risks 8(3), 82, Special Issue on Machine Learning in Finance, Eds. Thorsten Schmidt, 2020.


# Overview
Run the Master.ipynb or view the Master.html file to see a demo showing the backtesting of the GP, SSVI and Neural Network.
The file also compares the backtesting results.
## Other notebooks
Notebook Tikhonov-Backtest.ipynb shows backtests based on local volatility from trinomial tree pricing : Crepey, S. (2002). Calibration of the local volatility in a trinomial tree using Tikhonov regularization. Inverse Problems, 19(1), 91.

Notebook dupireNN.ipynb implements neural network local volatility from price with the Dupire formula.

Notebook dupireNN.ipynb implements neural network local volatility from price with the Dupire formula but without any constraint.

Notebook NeuralNetwork.ipynb implements neural network local volatility from implied volatility with neural network and from price.

Notebook SSVI.ipynb implements neural network local volatility from implied volatility with SSVI. It also provides backtests for SSVI.

Notebook GPBacktest.ipynb shows backtests based on local volatility from Gaussian Processes.


Notebook Result-Comparison.ipynb allows diagnosis of model performance based results from csv files.

Notebook Tools.ipynb is dedicated to conversion of dataset to .dat file.


The code folder contains python script for our library :
- BS.py offers some service related to black-scholes model (implied volatility calibration, pricing ...).
- SSVI.py implements arbitrage free calibration of Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative Finance, 14(1), 59-71.
- SSVIUnconstrained.py implements standard SVI calibration. Interpolation of SVI parameters is inspired from Gatheral (2014) but an unconstrained raw SVI parametrization is fitted for each slice.
- backtest.py implements Monte Carlo and PDE repricing in order to assess local volatility.
- bootstrapping.py extracts discount curve and dividend curve.
- dataSetConstruction.py provides some features engineering features.
- loadData.py extracts original data from excel, dat and csv files.
- neuralNetwork.py implements various architectures and run tensorflow models.
- plotTools.py provides plotting and performance diagnosis features.



SSVI calibration is inspired from Matlab code  Philipp Rindler (2020). Gatherals and Jacquier's Arbitrage-Free SVI Volatility Surfaces (https://www.mathworks.com/matlabcentral/fileexchange/49962-gatherals-and-jacquier-s-arbitrage-free-svi-volatility-surfaces), MATLAB Central File Exchange. Retrieved June 22, 2020.

The BS folder contains some additional Python scripts for implied volatility estimation, using the Bisection algorithm,  written by M. Dixon.

These notebooks are fully compatible with a local notebook environment provided that Tensorflow is installed for Python 3 with a version at or above 2.0.
For deployment on a Google colab environment, please be aware of files and foldr strcture (e.g. variable "workingFolder") in order to properly load data, python scripts and experiment results.

The Matlab code implements the shape constrained GP code and is written by A.  Cousin and D. Gueye.


Several days of DAX index option chain data is provided in the data folder.
S&P 500 and Eurostoxx 50 option prices are also included for more recent data.
The notebooks in the notebooks folder indicate which data file you should load for execution.
=======
The Matlab code implements the shape constrained GP code and is written by A.  Cousin and D. Gueye.
