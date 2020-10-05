import numpy as np
import pandas as pd
from scipy import interpolate
import BS
import bootstrapping
import sklearn as skl
from sklearn import preprocessing
import matplotlib.pyplot as plt

################################################################ Default dataset interpolator
#Linear interpolation combined with Nearest neighbor extrapolation
def customInterpolator(interpolatedData, newStrike, newMaturity):
  strikeRef = np.ravel(interpolatedData.index.get_level_values("Strike").values)
  maturityRef = np.ravel(interpolatedData.index.get_level_values("Maturity").values)
  xym = np.vstack((strikeRef, maturityRef)).T

  fInterpolation = interpolate.griddata(xym,
                                        interpolatedData.values.flatten(),
                                        (newStrike, newMaturity),
                                        method = 'linear',
                                        rescale=True)

  fExtrapolation =  interpolate.griddata(xym,
                                         interpolatedData.values.flatten(),
                                         (newStrike, newMaturity),
                                         method = 'nearest',
                                         rescale=True)
    
  return np.where(np.isnan(fInterpolation), fExtrapolation, fInterpolation)

  #return interpolate.interp2d(strikeRef, maturityRef,
  #                            interpolatedData.values.flatten(),
  #                            kind='linear')(newStrike, newMaturity)

#Get local volatility from Crépey (2002) by nearest neighbour interpolation
def interpolatedLocalVolatility(localVol, priceGrid):
    
    strikePrice = priceGrid.index.get_level_values("Strike").values.flatten()
    maturityPrice = priceGrid.index.get_level_values("Maturity").values.flatten()
    coordinates = customInterpolator(localVol["LocalVolatility"], strikePrice, maturityPrice)
 

    return pd.Series(coordinates, index = priceGrid.index)

########################################################## Dataset creation
#Generate a proper dataset from implied volatility
#If priceDf is provided then prices are taken from priceDf and implied volatility is only used to get greeks
def generateData(impliedVol,
                 S0,
                 bootstrap,
                 localVolatilityRef = None,
                 priceDf = None,
                 spotValue = True):
  #Get grid coordinates
  if priceDf is None :
    x_train = impliedVol.index.to_frame()
    #Get implied vol by interpolating another grid
    x_train["ImpliedVol"] = impliedVol
    x_train["OptionType"] = - np.ones_like(x_train["ImpliedVol"]) #Put by default
  else :
    x_train = pd.MultiIndex.from_arrays([priceDf["Strike"], priceDf["Maturity"]], 
                                        names=('Strike', 'Maturity')).to_frame()
    #Get implied vol by interpolating another grid
    x_train["ImpliedVol"] = customInterpolator(impliedVol, 
                                               x_train["Strike"], 
                                               x_train["Maturity"])
    x_train["OptionType"] = np.where(priceDf["OptionType"] == 1,
                                     np.ravel(priceDf["OptionType"].values),
                                     -1)

  #Get sensitivities and prices
  impliedPriceFunction = lambda x : BS.generalizedGreeks(x["OptionType"],
                                                         S0,
                                                         x["Strike"] ,
                                                         x["Maturity"],
                                                         x["ImpliedVol"],
                                                         bootstrap.discountIntegral(x["Maturity"]),
                                                         bootstrap.dividendIntegral(x["Maturity"]))
  
  res = np.reshape(np.array(list(zip(x_train.apply(impliedPriceFunction,axis=1).values))),
                   (x_train.shape[0], 6))  # put greeks
  prices = res[:,0] if priceDf is None else priceDf["Price"].values
  deltas = res[:,1]
  vegas = res[:,2]
  delta_ks = res[:,3]
  gamma_ks = res[:,4]
  delta_Ts = res[:,5]
  
  #Vega for optional loss weighting
  sigmaRef = 0.25
  impliedPriceFunction = lambda x : BS.generalizedGreeks(x["OptionType"],
                                                         S0,
                                                         x["Strike"] ,
                                                         x["Maturity"],
                                                         sigmaRef,
                                                         bootstrap.discountIntegral(x["Maturity"]),
                                                         bootstrap.dividendIntegral(x["Maturity"]))
  
  res1 = np.reshape(np.array(list(zip(x_train.apply(impliedPriceFunction,axis=1).values))),
                    (x_train.shape[0], 6))  # put greeks
  
  #Get adjusted strike for the change of variables
  changedVar = bootstrap.changeOfVariable(x_train["Strike"], x_train["Maturity"])
  
  multiIndex = x_train["ImpliedVol"].index

  #Gather all data as a Dataframe 
  cols = ["Price", "Delta", "Vega", "Delta Strike", "Gamma Strike", 
          "Theta", "ChangedStrike", "DividendFactor", "Strike", "Maturity", "ImpliedVol", "VegaRef", "OptionType"]

  dfData = np.vstack((prices, deltas, vegas, delta_ks, gamma_ks, delta_Ts) + 
                     changedVar + (x_train["Strike"], x_train["Maturity"],
                                   x_train["ImpliedVol"], res1[:,2], x_train["OptionType"]))
  
  df = pd.DataFrame(dfData.T , columns=cols, index = multiIndex)

  #Add pricing with spot delivery
  if spotValue : 
    KAvailable = multiIndex.get_level_values("Strike").unique()
    TSpot = np.zeros_like(KAvailable)
    indexSpot = pd.MultiIndex.from_arrays([np.array(KAvailable), TSpot], names=('Strike', 'Maturity'))

    #For put
    priceSpot = np.maximum(KAvailable- S0,0)
    deltaSpot = -np.sign(np.maximum(KAvailable- S0,0))
    gammaSpot = np.zeros_like(deltaSpot)
    vegasSpot = gammaSpot
    deltaKSpot = np.sign(np.maximum(KAvailable- S0,0))
    thetaSpot = 1000000 * deltaKSpot
    optType = - np.ones_like(KAvailable)

    #Ignore implied vol for T=0
    impliedSpot = np.zeros_like(thetaSpot)

    changedVarSpot = bootstrap.changeOfVariable(KAvailable, TSpot)
    
    dfDataSpot = np.vstack((priceSpot, deltaSpot, vegasSpot, deltaKSpot, gammaSpot, thetaSpot) +
                          changedVarSpot + (KAvailable, TSpot, impliedSpot, vegasSpot, optType))
    dfSpotPut = pd.DataFrame(dfDataSpot.T , columns=cols, index = indexSpot)

    priceSpot = np.maximum(S0 - KAvailable, 0)
    deltaSpot = np.sign(np.maximum(S0 - KAvailable, 0))
    deltaKSpot = - np.sign(np.maximum(S0 - KAvailable, 0))
    thetaSpot = 1000000 * deltaKSpot
    optType = np.ones_like(KAvailable)

    dfDataSpot = np.vstack((priceSpot, deltaSpot, vegasSpot, deltaKSpot, gammaSpot, thetaSpot) +
                           changedVarSpot + (KAvailable, TSpot, impliedSpot, vegasSpot, optType))
    dfSpotCall = pd.DataFrame(dfDataSpot.T, columns=cols, index=indexSpot)


    df = df.append(pd.concat([dfSpotCall, dfSpotPut])).sort_index()

  #Add forward logmoneyness if we want to calibrate local volatility from implied volatilities
  df["logMoneyness"] = np.log(df["ChangedStrike"] / S0)
  df["impliedTotalVariance"] = np.square(df["ImpliedVol"]) *  df["Maturity"]
  if localVolatilityRef is not None :
    df["locvol"] = interpolatedLocalVolatility(localVolatilityRef, df["Price"])

  return df.sort_index()
  
########################################################################## Sanity Check
def checkCallPutParity(dataSet, S0, strike, bootstrap):
    #Checking call put parity
    obs = -4
    maturity = dataSet.iloc[obs]["Maturity"]
    strike = dataSet.iloc[obs]["Strike"]
    isPut = dataSet.iloc[obs]["Option type"]
    print("Strike : ", strike, " ; Maturity : ", maturity)
    forwardPrice = (S0 * np.exp(-bootstrap.dividendIntegral(maturity))  - np.exp(-bootstrap.discountIntegral(maturity)) * strike)
    print("Forward Price : ", forwardPrice)

    #Put price minus call price
    CMinusP = (dataSet.iloc[obs]["Option price"] - dataSet.loc[(strike,round(maturity,3))]["Price"]) if isPut else (dataSet.iloc[obs]["Option price"] - dataSet.loc[(strike,round(maturity,3))]["Price"])
    print("Call minus put : ", forwardPrice)
    
    relativeError = abs(CMinusP - forwardPrice) / CMinusP
    print("Call put parity absolute relative error : ", relativeError)
    
    tolerance = 1e-4
    if relativeError > tolerance :
        print("Call put parity is violated")
    else : 
        print("Call put parity is respected")
    return



########################################################################## Saving data
def savingData(bootstrap,
               dataSet, 
               dataSetTest,
               workingFolder):
    tGrid = np.linspace(0, 30, 301)
    exportedRiskFreeIntegral = bootstrap.discountIntegral(tGrid),
    exportedDivSpreadIntegral = bootstrap.dividendIntegral(tGrid)
    exportedRRiskCurvespline = bootstrap.discountShortRate(tGrid),
    exportedDivSpline = bootstrap.dividendShortRate(tGrid)
    dfCurve = pd.DataFrame(np.vstack([exportedRiskFreeIntegral, exportedDivSpreadIntegral, exportedRRiskCurvespline, exportedDivSpline]).T,
                           columns=["riskFreeIntegral","divSpreadIntegral","riskCurvespline","divSpline"], 
                           index = tGrid)
    
    print("Saving bootstrapping under dfCurve.csv")
    dfCurve.to_csv(workingFolder + "dfCurve.csv")
    
    print("Saving training set under trainingDataSet.csv")
    dataSet.to_csv(workingFolder + "trainingDataSet.csv")
    
    print("Saving bootstrapping under testingDataSet.csv")
    dataSetTest.to_csv(workingFolder + "testingDataSet.csv")
    
    return dfCurve
    
    
######################################################################### Scaling
def transformCustomMinMax(df, scaler):
  return pd.DataFrame(scaler.transform(df),
                      index = df.index, 
                      columns = df.columns)
                      
#Reverse operation min-max scaling
def inverseTransformMinMax(df, scaler):
  return pd.DataFrame(scaler.inverse_transform(df),
                      index = df.index, 
                      columns = df.columns)
                      
#Same thing but for a particular column
def inverseTransformColumnMinMax(originalDf, scaler, column):
  colIndex = originalDf.columns.get_loc(column.name)
  maxCol = scaler.data_max_[colIndex]
  minCol = scaler.data_min_[colIndex]
  return pd.Series(minCol + (maxCol - minCol) * column, index = column.index).rename(column.name)  
  
#Reverse transform of min-max scaling but for greeks   
def inverseTransformColumnGreeksMinMax(originalDf, 
                                       scaler,
                                       columnDerivative,
                                       columnFunctionName,
                                       columnVariableName,
                                       order = 1):
  colFunctionIndex = originalDf.columns.get_loc(columnFunctionName)
  maxColFunction = scaler.data_max_[colFunctionIndex]
  minColFunction = scaler.data_min_[colFunctionIndex]
  scaleFunction = (maxColFunction - minColFunction)
  
  colVariableIndex = originalDf.columns.get_loc(columnVariableName)
  maxColVariable = scaler.data_max_[colVariableIndex]
  minColVariable = scaler.data_min_[colVariableIndex]
  scaleVariable = (maxColVariable - minColVariable) ** order

  return pd.Series(scaleFunction * columnDerivative / scaleVariable , 
                   index = columnDerivative.index).rename(columnDerivative.name) 
    
#Tools functions for min-max scaling
def transformCustomId(df, scaler):
  return pd.DataFrame(df,
                      index = df.index, 
                      columns = df.columns)

def inverseTransformId(df, scaler):
  return pd.DataFrame(df,
                      index = df.index, 
                      columns = df.columns)

def inverseTransformColumnId(originalDf, scaler, column):
  return pd.Series(column, index = column.index).rename(column.name)  

def inverseTransformColumnGreeksId(originalDf, scaler, 
                                 columnDerivative, 
                                 columnFunctionName, 
                                 columnVariableName,
                                 order = 1):
  return pd.Series(columnDerivative , index = columnDerivative.index).rename(columnDerivative.name)


def checkCallPutParity(rawData, S0, bootstrap):
    callPrice = rawData[rawData["OptionType"] == 1.0]["Price"].sort_index().reset_index()
    putPrice = rawData[rawData["OptionType"] == -1.0]["Price"].sort_index().reset_index()
    mergeDf = pd.merge(callPrice, putPrice,
                       left_on=["Maturity", "Strike"],
                       right_on=["Maturity", "Strike"],
                       how="inner")

    callPrice = mergeDf["Price_x"]
    putPrice = mergeDf["Price_y"]
    strike = pd.Series(mergeDf["Strike"].values, index=mergeDf.index)
    maturity = pd.Series(mergeDf["Maturity"].values, index=mergeDf.index)

    integratedDividend = np.exp(- bootstrap.dividendIntegral(maturity))
    integratedDiscount = np.exp(- bootstrap.discountIntegral(maturity))

    callPutForward = (callPrice - putPrice)
    forward = (S0 * integratedDividend - strike * integratedDiscount)
    relativeError = np.abs((callPutForward - forward) / callPutForward)

    print("Call put parity max absolute relative error : ",
          relativeError.max(), " , ",
          strike[relativeError.idxmax()], " , ",
          maturity[relativeError.idxmax()])
    print("Call put parity mean absolute relative error : ",
          relativeError.mean())
    plt.plot(maturity.values, relativeError.values)

    tolerance = 1e-4
    if relativeError.max() > tolerance:
        print("Call put parity is violated")
    else:
        print("Call put parity is respected")
    return