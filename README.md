# Non-parametric local volatility
Authors: Chataigner, Crepey and Dixon. If this code is used for research purposes, please cite as:




# Overview
Notebook dupireNN.ipynb implements neural network local volatility from price with the Dupire formula.

Notebook dupireNN.ipynb implements neural network local volatility from price with the Dupire formula but without any constraint.

Notebook NeuralNetwork.ipynb implements neural network local volatility from implied volatility with neural network.

Notebook LocVolGatheral.ipynb implements neural network local volatility from implied volatility with SSVI.

The code folder contains python script for our library :
- BS.py offers some service related to black-scholes model (implied volatility calibration, pricing ...).
- SSVI.py implements arbitrage free calibration of Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative Finance, 14(1), 59-71.
- backtest.py implements Monte Carlo and PDE repricing in order to assess local volatility.
- bootstrapping.py extracts discount curve and dividend curve.
- dataSetConstruction.py provides some features engineering features.
- loadData.py extracts original data from excel, dat and csv files.
- neuralNetwork.py implements various architectures and run tensorflow models.
- plotTools.py provides plotting and performance diagnosis features.

Due to GitHub size limitations (25mb max per file), outputs from our notebook have been deleted. Only the code remains in this notebook.

SSVI calibration is inspired from Matlab code  Philipp Rindler (2020). Gatherals and Jacquier's Arbitrage-Free SVI Volatility Surfaces (https://www.mathworks.com/matlabcentral/fileexchange/49962-gatherals-and-jacquier-s-arbitrage-free-svi-volatility-surfaces), MATLAB Central File Exchange. Retrieved June 22, 2020.

The BS folder contains some additional Python scripts for implied volatility estimation, using the Bisection algorithm,  written by M. Dixon.

These notebooks are fully compatible with Google Colab but can also be used in a local notebook environment provided that Tensorflow is installed for Python 3 with a version at or above 2.0.
For local deployment, please ignore cells starting with "from google.colab import files".

# Data
Several days of DAX index option chain data is provided in the data folder.
S&P 500 and Eurostoxx 50 option prices are also included for more recent data.
The notebook indicates which data file you should load for execution.
