import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from IPython.display import display
import sklearn as skl

import sys
formerPath = sys.path
sys.path.append('./code/')
sys.path.append('./BS/')

import os
formerStdOut = sys.stdout

import bootstrapping
import dataSetConstruction
import backtest
import BS
import loadData
import plotTools
import SSVI
import SSVIFerhati
import neuralNetwork


import importlib

sys.stdout = formerStdOut


###############################################################################################################################################
#Loading data
###############################################################################################################################################

def loadAndFormatData(source):
    if source == "SPX" :
        workingFolder = "./data/" #"./"
        fileName = "Option_SPX_18_Mai_2019Feuille2.xlsm"
        asOfDate = "2019-05-18"
        trainingSet, testingSet, bootstrap, S0 = loadData.loadCBOTData(workingFolder, fileName, asOfDate)
    elif source == "DAX" :
        #Read csv files as dataFrames
        workingFolder = "./data/09082001/"
        trainingSet, testingSet, bootstrap, S0 = loadData.loadDataFromCSV(workingFolder,
                                                                          "9_8_2001__filterdax")
    elif source == "DAX dat files":
        workingFolder = "./data/09082001/"
        trainingSet, testingSet, bootstrap, S0 = loadData.loadDataFromDat(workingFolder,
                                                                          "9_8_2001__filterdax")
    elif source == "Eurostoxx" :
        workingFolder = "./data/"
        asOfDate = "2019-01-10"
        filename = "Data_EuroStoxx50_20190110_all_for_Marc.xlsx"
        trainingSet, testingSet, bootstrap, S0 = loadData.loadESXData(workingFolder, filename, asOfDate)
    elif source == "Reload" : 
        workingFolder = "./data/"
        trainingSet, testingSet, bootstrap, S0 = loadData.loadFormattedData(workingFolder)
    else :
        raise Exception("Select a correct source among {SPX, DAX, DAX dat files, Eurostoxx, Reload}")
    
    dataSet = trainingSet #Training set
    dataSetTest = testingSet #Testing set
    dfCurve = dataSetConstruction.savingData(bootstrap, 
                                             dataSet, 
                                             dataSetTest, 
                                             workingFolder)
    KMin, KMax, midS0, scaler, scaledDataSet, scaledDataSetTest = dataSetConstruction.minMaxScaling(dataSet, 
                                                                                                    dataSetTest,
                                                                                                    S0)
    volLocaleGridDf = dataSetConstruction.generateOuterRectangleGrid(dataSet, dataSetTest, bootstrap, S0)
    
    return dataSet, dataSetTest, S0, bootstrap, KMin, KMax, midS0, scaler, scaledDataSet, scaledDataSetTest, volLocaleGridDf





###############################################################################################################################################
#Neural Network implied volatility
###############################################################################################################################################





def trainNeuralNetworkImpliedVolatility(dataSet, hyperparameters, scaler, computeLocalVolatility):
    #Execute this cell if you want to fit neural network with implied volatilities
    isPenalized = (hyperparameters["lambdaLocVol"] > 0.0) or (hyperparameters["lambdaSoft"] > 0.0) or (hyperparameters["lambdaGamma"] > 0.0)
    modelName = "convexSoftGatheralVolModel" if isPenalized else "unconstrainedConvexSoftGatheralVolModel"
    res = neuralNetwork.create_train_model_gatheral(neuralNetwork.NNArchitectureVanillaSoftGatheral,
                                                    dataSet,
                                                    computeLocalVolatility,
                                                    hyperparameters,
                                                    scaler,
                                                    modelName = modelName)


    y_pred4G, volLocale4G, dNN_T4G, gNN_K4G, lossSerie4G = res

    #Error plot
    plotTools.plotEpochLoss(lossSerie4G)
    return res



def evaluateNeuralNetworkImpliedVolatility(dataSet, 
                                           dataSetTest,
                                           hyperparameters,
                                           scaler,
                                           computeLocalVolatility,
                                           KMin,
                                           KMax,
                                           S0, midS0,
                                           bootstrap,
                                           tensorflowModelPath,
                                           csvResultsPath):
    # Evaluate results on the training set, you can execute that cell without training the model
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("Training Set")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    resTrain = neuralNetwork.create_eval_model_gatheral(neuralNetwork.NNArchitectureVanillaSoftGatheral,
                                                        dataSet,
                                                        computeLocalVolatility,
                                                        hyperparameters,
                                                        scaler,
                                                        modelName = tensorflowModelPath)
    y_pred4G, volLocale4G, dNN_T4G, gNN_K4G, lossSerie4G = resTrain

    plotTools.modelSummaryGatheral(y_pred4G, 
                                   volLocale4G, 
                                   dNN_T4G, 
                                   gNN_K4G, 
                                   dataSet,
                                   yMin = KMin,
                                   yMax = KMax, 
                                   S0 = S0, 
                                   bootstrap = bootstrap, 
                                   thresholdPrice = None,
                                   savePath = csvResultsPath + "Train")

    print("ATM Local Volatility : ")
    print(volLocale4G.loc[(midS0,slice(None))])
    
    print("------------------------------------------------------------------------------------")
    print("Log Moneyness coordinates")
    print("------------------------------------------------------------------------------------")
    #Diagnosis for training results with logMoneyness scale
    plotTools.modelSummaryGatheral(y_pred4G, 
                                   volLocale4G, 
                                   dNN_T4G, 
                                   gNN_K4G, 
                                   dataSet,
                                   logMoneynessScale = True,
                                   S0 = S0, 
                                   bootstrap = bootstrap, 
                                   thresholdPrice = None,
                                   yMin = KMin + 0.0001,
                                   yMax = KMax)
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("Testing Set")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    
    # Evaluate results on the testing dataset, you can execute that cell without training the model
    resTest = neuralNetwork.create_eval_model_gatheral(neuralNetwork.NNArchitectureVanillaSoftGatheral,
                                                       dataSetTest,
                                                       computeLocalVolatility,
                                                       hyperparameters,
                                                       scaler,
                                                       modelName = tensorflowModelPath)
    y_pred4TestG, volLocale4TestG, dNN_T4TestG, gNN_K4TestG, lossSerie4TestG = resTest

    plotTools.modelSummaryGatheral(y_pred4TestG, 
                                   volLocale4TestG, 
                                   dNN_T4TestG, 
                                   gNN_K4TestG, 
                                   dataSetTest,
                                   yMin = KMin,
                                   yMax = KMax,
                                   S0 = S0, 
                                   bootstrap = bootstrap, 
                                   thresholdPrice = None,
                                   savePath = csvResultsPath + "Test")
    
    print("------------------------------------------------------------------------------------")
    print("Log Moneyness coordinates")
    print("------------------------------------------------------------------------------------")
    #Diagnosis for testing results with logMoneyness scale
    plotTools.modelSummaryGatheral(y_pred4TestG, 
                                   volLocale4TestG, 
                                   dNN_T4TestG, 
                                   gNN_K4TestG, 
                                   dataSetTest,
                                   logMoneynessScale = True,
                                   S0 = S0, 
                                   bootstrap = bootstrap, 
                                   thresholdPrice = None,
                                   yMin = KMin + 0.0001,
                                   yMax = KMax)
    
    return resTrain, resTest


def evaluateNeuralNetworkArbitrableImpliedVolatility(dataSet, 
                                                     dataSetTest,
                                                     hyperparameters,
                                                     scaler,
                                                     KMin,
                                                     KMax,
                                                     S0, midS0,
                                                     bootstrap):
    resTrain, resTest = evaluateNeuralNetworkImpliedVolatility(dataSet, 
                                                               dataSetTest,
                                                               hyperparameters,
                                                               scaler,
                                                               False,
                                                               KMin,
                                                               KMax,
                                                               S0, midS0,
                                                               bootstrap,
                                                               "unconstrainedConvexSoftGatheralVolModel",
                                                               "./Results/NeuralUnconstrainedImpliedVol")
    return resTrain, resTest







def evaluateNeuralNetworkArbitrableFreeImpliedVolatility(dataSet, 
                                                         dataSetTest,
                                                         hyperparameters,
                                                         scaler,
                                                         KMin,
                                                         KMax,
                                                         S0, midS0,
                                                         bootstrap):
    resTrain, resTest = evaluateNeuralNetworkImpliedVolatility(dataSet, 
                                                               dataSetTest,
                                                               hyperparameters,
                                                               scaler,
                                                               True,
                                                               KMin,
                                                               KMax,
                                                               S0, midS0,
                                                               bootstrap,
                                                               "convexSoftGatheralVolModel",
                                                               "./Results/NeuralImpliedVol")
    return resTrain, resTest









###############################################################################################################################################
#Backtesting local volatility
###############################################################################################################################################




def backTestLocalVolatility(localVolatilityFunction, 
                            refinedGrid,
                            dataSetTest,
                            nbTimeStep,
                            nbPaths, 
                            KMin, 
                            KMax,
                            S0,
                            bootstrap,
                            modelName):
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("Estimating local volatility")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    
    volLocalGridRefined = localVolatilityFunction(refinedGrid["Strike"].values.flatten(),
                                                  refinedGrid["Maturity"].values.flatten())
    volLocalGridTest = localVolatilityFunction(dataSetTest.index.get_level_values("Strike").values.flatten(),
                                               dataSetTest.index.get_level_values("Maturity").values.flatten())

    #Local volatility function for backtests
    nnVolLocaleRefined = lambda x,y : backtest.interpolatedMCLocalVolatility(volLocalGridRefined, x, y)

    #Local volatility function for backtests
    nnVolLocaleTest = lambda x,y : backtest.interpolatedMCLocalVolatility(volLocalGridTest, x, y)

    plotTools.plotSerie(volLocalGridRefined,
                        Title = 'Local Volatility on refined grid',
                        az=105,
                        yMin=KMin,
                        yMax=KMax,
                        zAsPercent=False)

    plotTools.plotSerie(volLocalGridTest,
                        Title = 'Local Volatility on testing set',
                        az=30,
                        yMin=KMin,
                        yMax=KMax,
                        zAsPercent=True)

    plotTools.plotHeatMap(volLocalGridRefined,
                          Title = 'Local Volatility heatmap on refined Grid',
                          az=30,
                          yMin=KMin,
                          yMax=KMax,
                          zAsPercent=True)
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("Monte Carlo Backtest")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    
    #Local volatility in Monte Carlo diffusion is obtained through linear interpolation on local volatility values computed on refined grid.
    mcResVolLocaleRefined = backtest.MonteCarloPricerVectorized(S0,
                                                                dataSetTest,
                                                                bootstrap,
                                                                nbPaths,
                                                                nbTimeStep,
                                                                nnVolLocaleRefined)

    #Diagnose backtest results
    plotTools.predictionDiagnosis(mcResVolLocaleRefined["Price"], 
                                  dataSetTest["Price"], 
                                  " Monte Carlo Price (Refined grid) ", 
                                  yMin=KMin,
                                  yMax = KMax)
    workingFolder = "./Results/"
    mcResVolLocaleRefined.to_csv(workingFolder + "mcResVolLocaleRefined" + modelName + ".csv")
    
    #Same from local volatility values computed on testing grid.
    mcResVolLocaleTest = backtest.MonteCarloPricerVectorized(S0,
                                                             dataSetTest,
                                                             bootstrap,
                                                             nbPaths,
                                                             nbTimeStep,
                                                             nnVolLocaleTest)

    #Diagnose backtest results
    plotTools.predictionDiagnosis(mcResVolLocaleTest["Price"], 
                                  dataSetTest["Price"], 
                                  " Monte Carlo Price (Testing Grid) ", 
                                  yMin=KMin,
                                  yMax = KMax)
    workingFolder = "./Results/"
    mcResVolLocaleTest.to_csv(workingFolder + "mcResVolLocaleTest" + modelName + ".csv")
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("PDE Backtest")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    
    #PDE backtest with a cranck nicholson scheme
    pdeResVolLocaleRefined = backtest.PDEPricerVectorized(dataSetTest, S0, nnVolLocaleRefined, bootstrap)

    #Backtest diagnosis
    plotTools.predictionDiagnosis(pdeResVolLocaleRefined, 
                                  dataSetTest["Price"], 
                                  " PDE Price (Refined grid) ", 
                                  yMin=KMin,
                                  yMax=KMax)
    
    pdeResVolLocaleRefined.to_csv(workingFolder + "pdeResVolLocaleRefined" + modelName + ".csv")
    
    #PDE backtest with a cranck nicholson scheme
    pdeResVolLocaleTest = backtest.PDEPricerVectorized(dataSetTest, S0, nnVolLocaleTest, bootstrap)

    #Backtest diagnosis
    plotTools.predictionDiagnosis(pdeResVolLocaleTest, 
                                  dataSetTest["Price"], 
                                  " PDE Price (Testing Grid) ", 
                                  yMin=KMin,
                                  yMax=KMax)
    
    pdeResVolLocaleTest.to_csv(workingFolder + "pdeResVolLocaleTest" + modelName + ".csv")
    
    return volLocalGridRefined, volLocalGridTest, mcResVolLocaleRefined, mcResVolLocaleTest, pdeResVolLocaleRefined, pdeResVolLocaleTest






















###############################################################################################################################################
#Neural network Price
###############################################################################################################################################














def trainNeuralNetworkPrice(dataSet, hyperparameters, scaler, computeLocalVolatility):
    #Execute this cell if you want to fit neural network with implied volatilities
    isPenalized = (hyperparameters["lambdaLocVol"] > 0.0) or (hyperparameters["lambdaSoft"] > 0.0) or (hyperparameters["lambdaGamma"] > 0.0)
    modelName = "convexSoftVolModel" if isPenalized else "unconstrainedConvexSoftVolModel"
    #Execute this cell if you want to fit neural network with prices
    res = neuralNetwork.create_train_model(neuralNetwork.NNArchitectureVanillaSoftDupire,
                                           dataSet,
                                           computeLocalVolatility,
                                           hyperparameters,
                                           scaler,
                                           modelName = "convexSoftVolModel")
    y_pred4, volLocale4, dNN_T4, gNN_K4, lossSerie4 = res

    plotTools.plotEpochLoss(lossSerie4)
    return res




def evaluateNeuralNetworkPrice(dataSet, 
                               dataSetTest,
                               hyperparameters,
                               scaler,
                               computeLocalVolatility,
                               KMin,
                               KMax,
                               S0, midS0,
                               bootstrap,
                               tensorflowModelPath,
                               csvResultsPath):
    # Evaluate results on the training set, you can execute that cell without training the model
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("Training Set")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    resTrain = neuralNetwork.create_eval_model(neuralNetwork.NNArchitectureVanillaSoftDupire,
                                               dataSet,
                                               computeLocalVolatility,
                                               hyperparameters,
                                               scaler,
                                               modelName = tensorflowModelPath)
    y_pred4, volLocale4, dNN_T4, gNN_K4, lossSerie4 = resTrain

    plotTools.modelSummary(y_pred4,
                           volLocale4,
                           dNN_T4,
                           gNN_K4,
                           dataSet,
                           S0,
                           bootstrap,
                           yMin = KMin,
                           yMax = KMax,
                           thresholdPrice = None,
                           removeNaN = False,
                           savePath = csvResultsPath + "Train")

    print("ATM Local Vol :")
    print(volLocale4.loc[(midS0,slice(None))])
    
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("Testing Set")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    
    # Evaluate results on the testing dataset, you can execute that cell without training the model
    resTest = neuralNetwork.create_eval_model(neuralNetwork.NNArchitectureVanillaSoftDupire,
                                              dataSetTest,
                                              computeLocalVolatility,
                                              hyperparameters,
                                              scaler,
                                              modelName = tensorflowModelPath)
    y_pred4Test, volLocale4Test, dNN_T4Test, gNN_K4Test, lossSerie4Test = resTest

    plotTools.modelSummary(y_pred4Test,
                           volLocale4Test,
                           dNN_T4Test,
                           gNN_K4Test,
                           dataSetTest,
                           S0,
                           bootstrap,
                           yMin = KMin,
                           yMax = KMax,
                           thresholdPrice = None,
                           removeNaN = False,
                           savePath = csvResultsPath + "Test")
    
    return resTrain, resTest



def evaluateNeuralNetworkArbitrableFreePrice(dataSet, 
                                             dataSetTest,
                                             hyperparameters,
                                             scaler,
                                             KMin,
                                             KMax,
                                             S0, midS0,
                                             bootstrap):
    resTrain, resTest = evaluateNeuralNetworkPrice(dataSet, 
                                                   dataSetTest,
                                                   hyperparameters,
                                                   scaler,
                                                   True,
                                                   KMin,
                                                   KMax,
                                                   S0, midS0,
                                                   bootstrap,
                                                   "convexSoftVolModel",
                                                   "./Results/NeuralPrice")
    return resTrain, resTest




def AutomaticHyperparametersSelection(hyperparametersInit,
                                      neuralArchitecture,
                                      constrainLocalVolatility,
                                      scaledDataSet,
                                      scaler):
    hyperparameters = hyperparametersInit
    #Random selection of several hyperparameters 
    neuralNetwork.selectHyperparametersRandom(hyperparameters,
                                              ["lambdaLocVol","lambdaSoft","lambdaGamma"],
                                              neuralArchitecture,
                                              "hyperParameters",
                                              constrainLocalVolatility, 
                                              100,
                                              scaledDataSet,
                                              scaler,
                                              trainedOnPrice = False,
                                              logGrid = True)
    
    #hyperparameters["lambdaLocVol"] = 100
    #hyperparameters["lambdaSoft"] = 100 
    #hyperparameters["lambdaGamma"] = 10000
    hyperparameters["lambdaLocVol"] = 0.01#0.01 #100
    hyperparameters["lambdaSoft"] = 0.01#10#10 #100 
    hyperparameters["lambdaGamma"] = 100#10 #10000

    #marginal selection of hyperparameters
    neuralNetwork.selectHyperparameters(hyperparameters, 
                                        "lambdaLocVol", 
                                        neuralArchitecture, 
                                        "hyperParameters", 
                                        True, 
                                        scaledDataSet,
                                        scaler,
                                        trainedOnPrice = False,
                                        logGrid = True)

    neuralNetwork.selectHyperparameters(hyperparameters, 
                                        "DupireVarCap", 
                                        neuralArchitecture, 
                                        "hyperParameters", 
                                        True, 
                                        scaledDataSet,
                                        scaler,
                                        trainedOnPrice = False,
                                        logGrid = True)

    neuralNetwork.selectHyperparameters(hyperparameters, 
                                        "lambdaLocVol", 
                                        neuralArchitecture, 
                                        "hyperParameters", 
                                        True, 
                                        scaledDataSet,
                                        scaler,
                                        trainedOnPrice = False,
                                        logGrid = True)

    hyperparameters["lambdaLocVol"] = 100

    neuralNetwork.selectHyperparameters(hyperparameters, 
                                        "lambdaLocVol", 
                                        neuralArchitecture, 
                                        "hyperParameters", 
                                        True, 
                                        scaledDataSet,
                                        scaler,
                                        trainedOnPrice = False,
                                        logGrid = True)
    hyperparameters["nbUnits"] = 40

    neuralNetwork.selectHyperparameters(hyperparameters, 
                                        "nbUnits", 
                                        neuralArchitecture, 
                                        "hyperParameters", 
                                        True, 
                                        scaledDataSet,
                                        scaler,
                                        trainedOnPrice = False,
                                        logGrid = False)

    hyperparameters["nbUnits"] = 200

    neuralNetwork.selectHyperparameters(hyperparameters, 
                                        "lambdaLocVol", 
                                        neuralArchitecture, 
                                        "hyperParameters", 
                                        True, 
                                        scaledDataSet,
                                        scaler,
                                        trainedOnPrice = False,
                                        logGrid = True)

    hyperparameters["nbUnits"] = 40

    neuralNetwork.selectHyperparameters(hyperparameters, 
                                        "nbUnits", 
                                        neuralArchitecture, 
                                        "hyperParameters", 
                                        True, 
                                        scaledDataSet,
                                        scaler,
                                        trainedOnPrice = False,
                                        logGrid = False)

    hyperparameters["nbUnits"] = 200
    
    return hyperparameters

#hyperparametersC = AutomaticHyperparametersSelection(hyperparametersInit,
#                                                     neuralArchitecture,
#                                                     True,
#                                                     scaledDataSet,
#                                                     scaler)


def trainSSVIModel(trainingSet, S0, bootstrap, isConstrained):
    if isConstrained :
        SSVIModel = SSVI.SSVIModel(S0, bootstrap)
    else :
        SSVIModel = SSVIFerhati.SSVIModelFerhati(S0, bootstrap)
        SSVIModel.lambdaList = [0.0, 0.0, 0.0, 0.0, 0.0] #[1e-3, 1e-3, 1e-3, 1e-3, 1e-5]
        #SSVIModel.automaticHyperparametersTuning(dataSet)
    SSVIModel.fit(trainingSet)
    
    return SSVIModel

def evalSSVIModel(dataSet, ssviModel, KMin, KMax, S0, bootstrap, fileName):
    serie = ssviModel.eval(dataSet)
    
    plotTools.predictionDiagnosis(serie , 
                                  dataSet[BS.impliedVolColumn], 
                                  " SSVI Implied vol ", 
                                  yMin=KMin,
                                  yMax=KMax)
    
    impPrice = plotTools.plotImpliedVolPrices(np.square(serie) * serie.index.get_level_values("Maturity"),
                                              bootstrap,
                                              S0, 
                                              dataSet,
                                              yMin = KMin,
                                              yMax = KMax,
                                              thresholdPrice = None)

    #ImpVolPutSSVI = BS.vectorizedImpliedVolatilityCalibration(S0, bootstrap, 
    #                                                          dataSet["Maturity"],
    #                                                          dataSet["Strike"],
    #                                                          dataSet["OptionType"],
    #                                                          impPrice,
    #                                                          removeNaN = False)
    
    #ImpVolPutSSVI = pd.Series(ImpVolPutSSVI, index = dataSet.index).sort_index()
    
    #plotTools.predictionDiagnosis(ImpVolPutSSVI, 
    #                              dataSet[BS.impliedVolColumn], 
    #                              " Implied vol ", 
    #                              yMin=KMin,
    #                              yMax=KMax)
    
    #plotTools.predictionDiagnosis(ImpVolPutSSVI, 
    #                              serie, 
    #                              " Implied vol ", 
    #                              yMin=KMin,
    #                              yMax=KMax)
    
    
    #dT, hk, dK, locVolSSVI, density = finiteDifferenceSVI(dataSet, interpolateWithSSVI)
    dT, hk, dK, locVolSSVI, density = SSVIFerhati.finiteDifferenceSVI(dataSet, ssviModel.eval)
    
    plotTools.diagnoseLocalVol(dT,
                               locVolSSVI,
                               density,
                               ssviModel.eval(dataSet),
                               dataSet,
                               az=320,
                               yMin=KMin,
                               yMax=KMax)
    
    plotTools.saveDataModel(plotTools.removeDuplicateIndex(impPrice), 
                            plotTools.removeDuplicateIndex(locVolSSVI), 
                            plotTools.removeDuplicateIndex(serie) , 
                            "./Results/" + fileName) 
    
    return serie, dT, hk, dK, locVolSSVI, density





def loadGPResults(dataSet, dataSetTest, S0, bootstrap, KMin, KMax, volLocaleGridDf):
    pathFolder = "./Results/"

    GPRefined = loadData.loadResults("ArbitrageFreeGP/GP_output_Put_Price_fine_grid.xlsx")
    GPTrain = loadData.loadResults("ArbitrageFreeGP/GP_output_Put_Price_training_set.xlsx")
    GPTest = loadData.loadResults("ArbitrageFreeGP/GP_output_Put_Price_testing_set.xlsx")
    GPLocVol = loadData.loadResults("ArbitrageFreeGP/local_vol_nx_70_nt_22_alloc_9.xlsx")
    GPTrain = loadData.formatGPDatasets(GPTrain, dataSet, GPLocVol, S0, bootstrap)
    GPTest = loadData.formatGPDatasets(GPTest, dataSetTest, GPLocVol, S0, bootstrap)
    GPResults = pd.concat([GPTrain, GPTest]).sort_index()
    GPTrain.name = "Arbitrage-Free GP"
    GPTest.name = "Arbitrage-Free GP"
    GPResults.name = "Arbitrage-Free GP"
    
    nnGP = lambda x,y : backtest.interpolatedMCLocalVolatility(GPLocVol, x, y)
    plotTools.plotSerie(nnGP(dataSetTest["Strike"], dataSetTest["Maturity"]),
                        Title = 'Interpolated GP local volatility on testing nodes',
                        az=30,
                        yMin=KMin,
                        yMax=KMax, 
                        zAsPercent=True)
    locVolGP = nnGP(volLocaleGridDf["Strike"], volLocaleGridDf["Maturity"])
    plotTools.plotSerie(locVolGP,
                        Title = 'Interpolated GP local volatility on backtesting nodes',
                        az=30,
                        yMin=KMin,
                        yMax=KMax, 
                        zAsPercent=True)

    plotTools.plotSerie(locVolGP[locVolGP <= 2.0],
                        Title = '(Truncated) Interpolated GP local volatility on backtesting nodes',
                        az=105,
                        yMin=KMin,
                        yMax=KMax, 
                        zAsPercent=True)

    plotTools.plotHeatMap(nnGP(volLocaleGridDf["Strike"], volLocaleGridDf["Maturity"]),
                          Title = 'SSVI Local Volatility on refined Grid',
                          az=30,
                          yMin=KMin,
                          yMax=KMax,
                          zAsPercent=True)

    logMin = np.log(KMin/S0), 
    logMax = 0.1 #np.log(KMax/S0),

    plotTools.plotSerie(plotTools.convertToLogMoneyness(nnGP(dataSetTest["Strike"], dataSetTest["Maturity"]), S0),
                        Title = 'Interpolated GP local volatility on testing nodes',
                        az=30,
                        yMin=logMin,
                        yMax=logMax, 
                        zAsPercent=True)

     
    plotTools.plot2Series(plotTools.convertToLogMoneyness(nnGP(dataSet["Strike"], dataSet["Maturity"]), S0), 
                          plotTools.convertToLogMoneyness(nnGP(dataSetTest["Strike"], dataSetTest["Maturity"]), S0), 
                          yMin=logMin,
                          yMax=logMax,
                          az = 340,
                          Title = 'Interpolated Implied Vol Surface on testing nodes and training nodes')

    return nnGP



def compareResults(dataSet, dataSetTest, S0, bootstrap):
    print("Loading GP results : ")
    GPRefined = loadData.loadResults("ArbitrageFreeGP/GP_output_Put_Price_fine_grid.xlsx")
    GPTrain = loadData.loadResults("ArbitrageFreeGP/GP_output_Put_Price_training_set.xlsx")
    GPTest = loadData.loadResults("ArbitrageFreeGP/GP_output_Put_Price_testing_set.xlsx")
    GPLocVol = loadData.loadResults("ArbitrageFreeGP/local_vol_nx_70_nt_22_alloc_9.xlsx")
    GPTrain = loadData.formatGPDatasets(GPTrain, dataSet, GPLocVol, S0, bootstrap)
    GPTest = loadData.formatGPDatasets(GPTest, dataSetTest, GPLocVol, S0, bootstrap)
    GPResults = pd.concat([GPTrain, GPTest]).sort_index()
    GPTrain.name = "Arbitrage-Free GP"
    GPTest.name = "Arbitrage-Free GP"
    GPResults.name = "Arbitrage-Free GP"
    
    
    GPUnconstrainedRefined = loadData.loadResults("UnconstrainedGP/GP_output_Put_Price_fine_grid.xlsx")
    GPUnconstrainedTest = loadData.loadResults("UnconstrainedGP/GP_output_Put_Price_testing_set.xlsx")
    GPUnconstrainedTrain = loadData.loadResults("UnconstrainedGP/GP_output_Put_Price_training_set.xlsx")
    GPUnconstrainedTrain = loadData.formatGPDatasets(GPUnconstrainedTrain, dataSet, 
                                                     pd.Series(np.zeros_like(GPLocVol.values).flatten(), 
                                                               index = GPLocVol.index), 
                                                     S0, bootstrap)
    GPUnconstrainedTest = loadData.formatGPDatasets(GPUnconstrainedTest, dataSetTest, 
                                                    pd.Series(np.zeros_like(GPLocVol.values).flatten(), 
                                                              index = GPLocVol.index), 
                                                    S0, bootstrap)
    GPUnconstrainedResults = pd.concat([GPUnconstrainedTrain, GPUnconstrainedTest]).sort_index() 
    GPUnconstrainedTrain.name = "Arbitrable GP"
    GPUnconstrainedTest.name = "Arbitrable GP"
    GPUnconstrainedResults.name = "Arbitrable GP"
    
    print("Loading Neural results : ")
    NeuralTest = loadData.loadResults('NeuralImpliedVolTest.csv')
    NeuralTrain = loadData.loadResults('NeuralImpliedVolTrain.csv')
    NeuralResults = pd.concat([NeuralTrain, NeuralTest]).sort_index()
    NeuralTrain.name = "Constrained NN"
    NeuralTest.name = "Constrained NN"
    NeuralResults.name = "Constrained NN"

    NeuralUnconstrainedTest = loadData.loadResults('NeuralUnconstrainedImpliedVolTest.csv') 
    NeuralUnconstrainedTrain = loadData.loadResults('NeuralUnconstrainedImpliedVolTrain.csv')
    NeuralUnconstrainedResults = pd.concat([NeuralUnconstrainedTrain, NeuralUnconstrainedTest]).sort_index()
    NeuralUnconstrainedTrain.name = "Unconstrained NN"
    NeuralUnconstrainedTest.name = "Unconstrained NN"
    NeuralUnconstrainedResults.name = "Unconstrained NN"
    
    
    print("Loading Neural results with price: ")
    NeuralPriceTest = loadData.loadResults('NeuralPriceTest.csv')
    NeuralPriceTrain = loadData.loadResults('NeuralPriceTrain.csv')
    NeuralPriceResults = pd.concat([NeuralPriceTrain, NeuralPriceTest]).sort_index()
    NeuralPriceTrain.name = "Constrained Price NN"
    NeuralPriceTest.name = "Constrained Price NN"
    NeuralPriceResults.name = "Constrained Price NN"
    
    print("Loading SSVI results : ")
    SSVITest = loadData.loadResults('SSVIConstrainedTesting.csv')
    SSVITrain = loadData.loadResults('SSVIConstrainedTraining.csv')
    SSVIResults = pd.concat([SSVITrain, SSVITest]).sort_index()
    SSVITrain.name = "Arbitrage-free SSVI"
    SSVITest.name = "Arbitrage-free SSVI"
    SSVIResults.name = "Arbitrage-free SSVI"

    SSVIUnconstrainedTest = loadData.loadResults('SSVIUnconstrainedTesting.csv')
    SSVIUnconstrainedTrain = loadData.loadResults('SSVIUnconstrainedTraining.csv')
    SSVIUnconstrainedResults = pd.concat([SSVIUnconstrainedTrain, SSVIUnconstrainedTest]).sort_index()
    SSVIUnconstrainedTrain.name = "Unconstrained SSVI"
    SSVIUnconstrainedTest.name = "Unconstrained SSVI"
    SSVIUnconstrainedResults.name = "Unconstrained SSVI"
    
    print("-----------------------------------------------------------------------------------------------------------------")
    print("Plotting constrained implied volatilities : ")
    plotTools.plot2dSmiles(SSVIResults,
                           GPResults,
                           NeuralResults,
                           dataSet,
                           dataSetTest,
                           plotMarketData = False,
                           nbObservationThreshold = 0,
                           maturityList = [0.055, 0.074, 0.093, 0.189, 0.37, 0.841, 1.09,2.585],
                           showDiff = False,
                           useLogMoneyness = True,
                           gpQuantiles = [GPResults["ImpliedVolMin"], GPResults["ImpliedVolMax"]],
                           legend = True)
    
    plotTools.plot2dSmiles(SSVIResults,
                           GPResults,
                           NeuralResults,
                           dataSet,
                           dataSetTest,
                           plotMarketData = False,
                           nbObservationThreshold = 0,
                           maturityList = [0.055, 0.074, 0.093, 0.189, 0.37, 0.841, 1.09,2.585],
                           showDiff = True,
                           useLogMoneyness = True,
                           gpQuantiles = [GPResults["ImpliedVolMin"], GPResults["ImpliedVolMax"]],
                           legend = True)
    
    print("-----------------------------------------------------------------------------------------------------------------")
    print("Plotting unconstrained implied volatilities : ")
    plotTools.plot2dSmiles(SSVIUnconstrainedResults,
                           GPUnconstrainedResults,
                           NeuralUnconstrainedResults,
                           dataSet,
                           dataSetTest,
                           plotMarketData = False,
                           nbObservationThreshold = 0,
                           maturityList = [0.055, 0.074, 0.093, 0.189, 0.37, 0.841, 1.09,2.585],
                           showDiff = False,
                           useLogMoneyness = True,
                           gpQuantiles = [GPUnconstrainedResults["ImpliedVol99Pct"], GPUnconstrainedResults["ImpliedVol1Pct"]],
                           legend = True)
    plotTools.plot2dSmiles(SSVIUnconstrainedResults,
                           GPUnconstrainedResults,
                           NeuralUnconstrainedResults,
                           dataSet,
                           dataSetTest,
                           plotMarketData = False,
                           nbObservationThreshold = 0,
                           maturityList = [0.055, 0.074, 0.093, 0.189, 0.37, 0.841, 1.09,2.585],
                           showDiff = True,
                           useLogMoneyness = True,
                           gpQuantiles = [GPUnconstrainedResults["ImpliedVolMin"], GPUnconstrainedResults["ImpliedVolMax"]],
                           legend = True)
    print("-----------------------------------------------------------------------------------------------------------------")
    print("Plotting constrained Price : ")
    plotTools.plot2dPriceSmiles(SSVIResults,
                                GPResults,
                                NeuralResults,
                                dataSet,
                                dataSetTest,
                                plotMarketData = False,
                                nbObservationThreshold = 0,
                                maturityList = [0.055, 0.074, 0.093, 0.189, 0.37, 0.841, 1.09,2.585],
                                showDiff = False,
                                useLogMoneyness = True,
                                gpQuantiles = [GPResults["HullMin"], GPResults["HullMax"]],
                                legend = True)
    
    print("-----------------------------------------------------------------------------------------------------------------")
    print("Plotting unconstrained Price : ")
    plotTools.plot2dPriceSmiles(SSVIUnconstrainedResults,
                                GPUnconstrainedResults,
                                NeuralUnconstrainedResults,
                                dataSet,
                                dataSetTest,
                                plotMarketData = False,
                                nbObservationThreshold = 0,
                                maturityList = [0.055, 0.074, 0.093, 0.189, 0.37, 0.841, 1.09,2.585],
                                showDiff = False,
                                useLogMoneyness = True,
                                gpQuantiles = [GPUnconstrainedResults["HullMin"], GPUnconstrainedResults["HullMax"]],
                                legend = True)
    
    print("-----------------------------------------------------------------------------------------------------------------")
    print("Summarize RMSEs : ")
    dataRMSE = [[backtest.rmse(dataSet["Price"], SSVITrain["Price"]),
                 backtest.rmse(dataSet["Price"], GPTrain["Price"]),
                 backtest.rmse(dataSet["Price"], NeuralTrain["Price"]),
                 backtest.rmse(dataSet["Price"], NeuralPriceTrain["Price"]),
                 backtest.rmse(dataSet["Price"], SSVIUnconstrainedTrain["Price"]),
                 backtest.rmse(dataSet["Price"], GPUnconstrainedTrain["Price"]),
                 backtest.rmse(dataSet["Price"], NeuralUnconstrainedTrain["Price"])],
                [backtest.rmse(dataSetTest["Price"], SSVITest["Price"]),
                 backtest.rmse(dataSetTest["Price"], GPTest["Price"]),
                 backtest.rmse(dataSetTest["Price"], NeuralTest["Price"]),
                 backtest.rmse(dataSetTest["Price"], NeuralPriceTest["Price"]),
                 backtest.rmse(dataSetTest["Price"], SSVIUnconstrainedTest["Price"]),
                 backtest.rmse(dataSetTest["Price"], GPUnconstrainedTest["Price"]),
                 backtest.rmse(dataSetTest["Price"], NeuralUnconstrainedTest["Price"])]]
    
    dfRMSE = pd.DataFrame(dataRMSE, index = ["Train","Test"], 
                          columns = ["SSVI", "GP", "Neural Network", "Neural Network with price",
                                     "SSVI Unconstrained","GP Unconstrained","Neural Network Unconstrained"])
    print("Absolute Price RMSEs")
    display(dfRMSE.round(decimals = 3))
    
    dataRelRMSE = [[backtest.rmse(dataSet["Price"], SSVITrain["Price"], relative=True),
                    backtest.rmse(dataSet["Price"], GPTrain["Price"], relative=True),
                    backtest.rmse(dataSet["Price"], NeuralTrain["Price"], relative=True),
                    backtest.rmse(dataSet["Price"], NeuralPriceTrain["Price"], relative=True),
                    backtest.rmse(dataSet["Price"], SSVIUnconstrainedTrain["Price"], relative=True),
                    backtest.rmse(dataSet["Price"], GPUnconstrainedTrain["Price"], relative=True),
                    backtest.rmse(dataSet["Price"], NeuralUnconstrainedTrain["Price"], relative=True)],
                   [backtest.rmse(dataSetTest["Price"], SSVITest["Price"], relative=True),
                    backtest.rmse(dataSetTest["Price"], GPTest["Price"], relative=True),
                    backtest.rmse(dataSetTest["Price"], NeuralTest["Price"], relative=True),
                    backtest.rmse(dataSetTest["Price"], NeuralPriceTest["Price"], relative=True),
                    backtest.rmse(dataSetTest["Price"], SSVIUnconstrainedTest["Price"], relative=True),
                    backtest.rmse(dataSetTest["Price"], GPUnconstrainedTest["Price"], relative=True),
                    backtest.rmse(dataSetTest["Price"], NeuralUnconstrainedTest["Price"], relative=True)]]

    dfRelRMSE = pd.DataFrame(dataRelRMSE, index = ["Train","Test"], 
                             columns = ["SSVI", "GP", "Neural Network", "Neural Network with price",
                                        "SSVI Unconstrained","GP Unconstrained","Neural Network Unconstrained"])
    print("Relative Price RMSEs (%)")
    display((dfRelRMSE * 100).round(decimals = 2))
    
    
    dataRMSEImp = [[backtest.rmse(dataSet["ImpVolCalibrated"], SSVITrain["ImpliedVol"]),
                    backtest.rmse(dataSet["ImpVolCalibrated"], GPTrain["ImpliedVol"]),
                    backtest.rmse(dataSet["ImpVolCalibrated"], NeuralTrain["ImpliedVol"]),
                    backtest.rmse(dataSet["ImpVolCalibrated"], NeuralPriceTrain["ImpliedVol"]),
                    backtest.rmse(dataSet["ImpVolCalibrated"], SSVIUnconstrainedTrain["ImpliedVol"]),
                    backtest.rmse(dataSet["ImpVolCalibrated"], GPUnconstrainedTrain["ImpliedVol"]),
                    backtest.rmse(dataSet["ImpVolCalibrated"], NeuralUnconstrainedTrain["ImpliedVol"])],
                   [backtest.rmse(dataSetTest["ImpVolCalibrated"], SSVITest["ImpliedVol"]),
                    backtest.rmse(dataSetTest["ImpVolCalibrated"], GPTest["ImpliedVol"]),
                    backtest.rmse(dataSetTest["ImpVolCalibrated"], NeuralTest["ImpliedVol"]),
                    backtest.rmse(dataSetTest["ImpVolCalibrated"], NeuralPriceTest["ImpliedVol"]),
                    backtest.rmse(dataSetTest["ImpVolCalibrated"], SSVIUnconstrainedTest["ImpliedVol"]),
                    backtest.rmse(dataSetTest["ImpVolCalibrated"], GPUnconstrainedTest["ImpliedVol"]),
                    backtest.rmse(dataSetTest["ImpVolCalibrated"], NeuralUnconstrainedTest["ImpliedVol"])]]

    dfRMSEImp = pd.DataFrame(dataRMSEImp, index = ["Train","Test"], 
                             columns = ["SSVI", "GP", "Neural Network", "Neural Network with price",
                                        "SSVI Unconstrained","GP Unconstrained","Neural Network Unconstrained"])
    print("Absolute Implied volatility RMSEs")
    display(dfRMSEImp.round(decimals = 4))
    
    dataRMSERelImp = [[backtest.rmse(dataSet["ImpVolCalibrated"], SSVITrain["ImpliedVol"], relative=True),
                       backtest.rmse(dataSet["ImpVolCalibrated"], GPTrain["ImpliedVol"], relative=True),
                       backtest.rmse(dataSet["ImpVolCalibrated"], NeuralTrain["ImpliedVol"], relative=True),
                       backtest.rmse(dataSet["ImpVolCalibrated"], NeuralPriceTrain["ImpliedVol"], relative=True),
                       backtest.rmse(dataSet["ImpVolCalibrated"], SSVIUnconstrainedTrain["ImpliedVol"], relative=True),
                       backtest.rmse(dataSet["ImpVolCalibrated"], GPUnconstrainedTrain["ImpliedVol"], relative=True),
                       backtest.rmse(dataSet["ImpVolCalibrated"], NeuralUnconstrainedTrain["ImpliedVol"], relative=True)],
                      [backtest.rmse(dataSetTest["ImpVolCalibrated"], SSVITest["ImpliedVol"], relative=True),
                       backtest.rmse(dataSetTest["ImpVolCalibrated"], GPTest["ImpliedVol"], relative=True),
                       backtest.rmse(dataSetTest["ImpVolCalibrated"], NeuralTest["ImpliedVol"], relative=True),
                       backtest.rmse(dataSetTest["ImpVolCalibrated"], NeuralPriceTest["ImpliedVol"], relative=True),
                       backtest.rmse(dataSetTest["ImpVolCalibrated"], SSVIUnconstrainedTest["ImpliedVol"], relative=True),
                       backtest.rmse(dataSetTest["ImpVolCalibrated"], GPUnconstrainedTest["ImpliedVol"], relative=True),
                       backtest.rmse(dataSetTest["ImpVolCalibrated"], NeuralUnconstrainedTest["ImpliedVol"], relative=True)]]

    dfRMSERelImp = pd.DataFrame(dataRMSERelImp, index = ["Train","Test"], 
                                columns = ["SSVI", "GP", "Neural Network", "Neural Network with price",
                                           "SSVI Unconstrained","GP Unconstrained","Neural Network Unconstrained"])
    print("Relative Implied volatility RMSEs (%)")
    display((dfRMSERelImp * 100).round(decimals = 2))
    
    
    workingFolder = "./Results/"
    backtestPDENeural = backtest.loadMCPrices(workingFolder + "pdeResVolLocaleRefinedNeuralImpliedVolatility.csv", 
                                              parseHeader=None)
    backtestMCNeural = backtest.loadMCPrices(workingFolder + "mcResVolLocaleRefinedNeuralImpliedVolatility.csv", 
                                             parseHeader=0)
    
    backtestPDENeuralPrice = backtest.loadMCPrices(workingFolder + "pdeResVolLocaleRefinedNeuralPrice.csv", 
                                                   parseHeader=None)
    backtestMCNeuralPrice = backtest.loadMCPrices(workingFolder + "mcResVolLocaleRefinedNeuralPrice.csv", 
                                                  parseHeader=0)

    backtestPDEGP = backtest.loadMCPrices(workingFolder + "pdeResVolLocaleRefinedGP.csv", 
                                          parseHeader=None)
    backtestMCGP = backtest.loadMCPrices(workingFolder + "mcResVolLocaleRefinedGP.csv", 
                                         parseHeader=0)

    backtestPDESSVI = backtest.loadMCPrices(workingFolder + "pdeResVolLocaleRefinedSSVI.csv", 
                                            parseHeader=None)
    backtestMCSSVI = backtest.loadMCPrices(workingFolder + "mcResVolLocaleRefinedSSVI.csv", 
                                           parseHeader=0)

    dfRMSEBacktest = [[backtest.rmse(dataSetTest["Price"], backtestMCSSVI["Price"]),
                       backtest.rmse(dataSetTest["Price"], backtestMCGP["Price"]),
                       backtest.rmse(dataSetTest["Price"], backtestMCNeural["Price"]), 
                       backtest.rmse(dataSetTest["Price"], backtestMCNeuralPrice["Price"])],
                      [backtest.rmse(dataSetTest["Price"], backtestPDESSVI[2]),
                       backtest.rmse(dataSetTest["Price"], backtestPDEGP[2]),
                       backtest.rmse(dataSetTest["Price"], backtestPDENeural[2]),
                       backtest.rmse(dataSetTest["Price"], backtestPDENeuralPrice[2])]]

    dfRMSEBacktest = pd.DataFrame(dfRMSEBacktest, index = ["Monte Carlo","PDE"], 
                                  columns = ["SSVI", "GP", "Neural Network", "Neural Network with price"])
    print("Absolute Price backtesting RMSEs")
    display(dfRMSEBacktest.round(decimals = 3))
    
    
    dfRMSERelBacktest = [[backtest.rmse(dataSetTest["Price"], backtestMCSSVI["Price"], relative=True),
                          backtest.rmse(dataSetTest["Price"], backtestMCGP["Price"], relative=True),
                          backtest.rmse(dataSetTest["Price"], backtestMCNeural["Price"], relative=True),
                          backtest.rmse(dataSetTest["Price"], backtestMCNeuralPrice["Price"], relative=True)],
                         [backtest.rmse(dataSetTest["Price"], backtestPDESSVI[2], relative=True),
                          backtest.rmse(dataSetTest["Price"], backtestPDEGP[2], relative=True),
                          backtest.rmse(dataSetTest["Price"], backtestPDENeural[2], relative=True),
                          backtest.rmse(dataSetTest["Price"], backtestPDENeuralPrice[2], relative=True)]]

    dfRMSERelBacktest = pd.DataFrame(dfRMSERelBacktest, index = ["Monte Carlo","PDE"], 
                                     columns = ["SSVI", "GP", "Neural Network", "Neural Network with price"])
    print("Relative Price backtesting RMSEs (%)")
    display((dfRMSERelBacktest * 100).round(decimals = 2))
    
    return
