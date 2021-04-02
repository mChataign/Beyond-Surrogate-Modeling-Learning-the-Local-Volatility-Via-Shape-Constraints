import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as mtick
import BS
import bootstrapping
import math

impliedVolColumn = BS.impliedVolColumn

######################################################################### Training loss

#Plot loss for each epoch 
def plotEpochLoss(lossSerie):
  fig = plt.figure(figsize=(20,10))
  ax = fig.gca()
  
  ax.plot(lossSerie , "-", color="black")
  ax.set_xlabel("Epoch number", fontsize=18, labelpad=20)
  ax.set_ylabel("Logarithmic Loss", fontsize=18, labelpad=20)
  ax.set_title("Training Loss evolution", fontsize=24)
  ax.tick_params(labelsize=16)
  ax.set_facecolor('white')
  plt.show()
  return


######################################################################### Plotting smiles

#Plot a surface as a superposition of curves
def plotMultipleCurve(data,
                      Title = 'True Price Surface',
                      yMin = 0,
                      yMax = 1,
                      zAsPercent = False):
  

  dataCurve = data[(data.index.get_level_values("Strike") <= yMax) * (data.index.get_level_values("Strike") >= yMin)]

  fig = plt.figure(figsize=(20,10))
  ax = fig.gca()

  for t in np.linspace(0,0.8,9) :
    k = dataCurve[dataCurve.index.get_level_values("Maturity") >= t].index.get_level_values("Maturity").unique().min()
    curveK = dataCurve[dataCurve.index.get_level_values("Maturity")==k]
    dataSerie = pd.Series(curveK.values * (100 if zAsPercent else 1) ,
                          index = curveK.index.get_level_values("Strike"))
    ax.plot(dataSerie , "--+", label=str(k))
  ax.legend()  
  ax.set_xlabel(data.index.names[0], fontsize=18, labelpad=20)
  ax.set_ylabel(data.name, fontsize=18, labelpad=20)
  if zAsPercent :
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
  ax.set_title(Title, fontsize=24)
  ax.tick_params(labelsize=16)
  ax.set_facecolor('white')
  plt.show()
  return


######################################################################### Plot surface

#Plotting function for surface
#xTitle : title for x axis
#yTitle : title for y axis
#zTitle : title for z axis
#Title : plot title
#az : azimuth i.e. angle of view for surface
#yMin : minimum value for y axis
#yMax : maximum value for y axis
#zAsPercent : boolean, if true format zaxis as percentage 
def plot2GridCustom(coordinates, zValue,
                    coordinates2, zValue2,
                    xTitle = "Maturity",
                    yTitle = "Strike",
                    zTitle = "Price",
                    Title = 'True Price Surface', 
                    az=320, 
                    yMin = 0,
                    yMax = 1,
                    zAsPercent = False):
  y = coordinates[:,0]
  filteredValue = (y > yMin) & (y < yMax)
  x = coordinates[:,1][filteredValue]
  y = coordinates[:,0][filteredValue]
  z = zValue[filteredValue].flatten()
  
  y2 = coordinates2[:,0]
  filteredValue2 = (y2 > yMin) & (y2 < yMax)
  x2 = coordinates2[:,1][filteredValue2]
  y2 = coordinates2[:,0][filteredValue2]
  z2 = zValue2[filteredValue2].flatten()
  
  fig = plt.figure(figsize=(20,10))
  ax = fig.gca(projection='3d')
  
  ax.set_xlabel(xTitle, fontsize=18, labelpad=20)
  ax.set_ylabel(yTitle, fontsize=18, labelpad=20)
  ax.set_zlabel(zTitle, fontsize=18, labelpad=10)
  
  cmap=plt.get_cmap("inferno")
  colors=cmap(z * 100 if zAsPercent else z)[np.newaxis, :, :3]
  ax.scatter(x2, y2, z2, marker='o', color="r", alpha=1, s=40)
  ax.scatter(x, y, z, marker='o', color="b", alpha=1, s=40)
  #surf = ax.plot_trisurf(x, y,
  #                       z * 100 if zAsPercent else z ,
  #                       linewidth=1.0,
  #                       antialiased=True, 
  #                       cmap = cmap,
  #                       color=(0,0,0,0))
  #scaleEdgeValue = surf.to_rgba(surf.get_array())
  #surf.set_edgecolors(scaleEdgeValue) 
  #surf.set_alpha(0)


  if zAsPercent :
    ax.zaxis.set_major_formatter(mtick.PercentFormatter())
  ax.view_init(elev=10., azim=az)
  #ax.set_title(Title, fontsize=24)
  ax.set_facecolor('white')

  plt.tick_params(labelsize=16)

  
  plt.show()


  return

#Plotting function from a pandas series
def plot2Series(data, 
                data2,
                Title = 'True Price Surface',
                az=320,
                yMin = 0,
                yMax = 1, 
                zAsPercent = False):
  

  plot2GridCustom(data.index.to_frame().values, 
                  data.values,
                  data2.index.to_frame().values, 
                  data2.values,
                  xTitle = data.index.names[1],
                  yTitle = data.index.names[0],
                  zTitle = data.name,
                  Title = Title, 
                  az=az, 
                  yMin = yMin, 
                  yMax = yMax, 
                  zAsPercent = zAsPercent)
  return



def convertToLogMoneyness(formerSerie, S0):
  maturity = formerSerie.index.get_level_values("Maturity")
  logMoneyness = np.log(S0 / formerSerie.index.get_level_values("Strike"))
  newIndex = pd.MultiIndex.from_arrays([np.array(logMoneyness.values), np.array(maturity.values)],
                                       names=('LogMoneyness', 'Maturity'))
  if type(formerSerie) == type(pd.Series()) :
    return pd.Series(formerSerie.values , index=newIndex)
  return pd.DataFrame(formerSerie.values, index = newIndex, columns= formerSerie.columns)



#Plotting function for surface
#xTitle : title for x axis
#yTitle : title for y axis
#zTitle : title for z axis
#Title : plot title
#az : azimuth i.e. angle of view for surface
#yMin : minimum value for y axis
#yMax : maximum value for y axis
#zAsPercent : boolean, if true format zaxis as percentage 
def plotGridCustom(coordinates, zValue,
                   xTitle = "Maturity",
                   yTitle = "Strike",
                   zTitle = "Price",
                   Title = 'True Price Surface', 
                   az=320, 
                   yMin = 0,
                   yMax = 1,
                   zAsPercent = False):
  y = coordinates[:,0]
  filteredValue = (y > yMin) & (y < yMax)
  x = coordinates[:,1][filteredValue]
  y = coordinates[:,0][filteredValue]
  z = zValue[filteredValue].flatten()
  
  fig = plt.figure(figsize=(15,9))
  ax = fig.gca(projection='3d')
  
  fontsize = 15
  pad = 20
  ax.set_xlabel(xTitle, color = "k", fontsize=fontsize, labelpad=pad * 1.0)
  ax.set_ylabel(yTitle, color = "k", fontsize=fontsize, labelpad=pad * 1.0)
  ax.set_zlabel(zTitle, color = "k", fontsize=fontsize, labelpad=pad * 1.0)
  
  cmap=plt.get_cmap("jet")#("inferno")
  colors=cmap(z * 100 if zAsPercent else z)[np.newaxis, :, :3]
  surf = ax.plot_trisurf(x, y,
                         z * 100 if zAsPercent else z ,
                         linewidth=1.0,
                         antialiased=True, 
                         cmap = cmap,
                         color=(0,0,0,0))
  scaleEdgeValue = surf.to_rgba(surf.get_array())
  surf.set_edgecolors(scaleEdgeValue) 
  surf.set_alpha(0)

  if zAsPercent :
    ax.zaxis.set_major_formatter(mtick.PercentFormatter())
  ax.view_init(elev=40., azim=az)
  ax.set_ylim(np.amax(y), np.amin(y))
  ax.set_title(Title, fontsize=fontsize * 1.2)#, rotation='vertical', x=0.1, y=0.8)
  ax.set_facecolor('white')

  plt.tick_params(axis = "y", labelsize=fontsize * 0.9, pad = pad * 0.4, color = [1,0,0,1])
  plt.tick_params(axis = "z", labelsize=fontsize * 0.9, pad = pad * 0.5, color = [1,0,0,1])
  plt.tick_params(axis = "x", labelsize=fontsize * 0.9, pad = pad * 0.05, color = [1,0,0,1])
  
  plt.tight_layout()
  
  plt.show()


  return



#Plotting function from a dataframe
def plotSurface(data, 
                zName, 
                Title = 'True Price Surface', 
                az=320,
                yMin = 0,
                yMax = 1,
                zAsPercent = False):
  plotGridCustom(dataSet.index.to_frame().values, 
                 data[zName].values,
                 xTitle = dataSet.index.names[1],
                 yTitle = dataSet.index.names[0],
                 zTitle = zName,
                 Title = Title, 
                 az=az, 
                 yMin = yMin, 
                 yMax = yMax, 
                 zAsPercent=zAsPercent)
  return

#Plotting function from a pandas series
def plotSerie(data,
              Title = 'True Price Surface',
              az=320,
              yMin = 0,
              yMax = 1, 
              zAsPercent = False):
  

  plotGridCustom(data.index.to_frame().values, 
                 data.values,
                 xTitle = data.index.names[1],
                 yTitle = data.index.names[0],
                 zTitle = data.name,
                 Title = Title, 
                 az=az, 
                 yMin = yMin, 
                 yMax = yMax, 
                 zAsPercent = zAsPercent)
  return

######################################################################### Training Diagnostic 

def selectIndex(df, indexToKeep):
    return df.loc[indexToKeep][ ~df.loc[indexToKeep].index.duplicated(keep='first') ]

#Plot predicted value, benchmark value, absoluate error and relative error
#It also compute RMSE between predValue and refValue
#predValue : approximated value 
#refValue : benchamrk value
#quantityName : name for approximated quantity
#az : azimuth i.e. angle of view for surface
#yMin : minimum value for y axis
#yMax : maximum value for y axis
def predictionDiagnosis(predValue, 
                        refValue, 
                        quantityName, 
                        az=320,
                        yMin = 0,
                        yMax = 1,
                        threshold = None):
  if threshold is not None :
      filterIndex = refValue[refValue >= threshold].index
      predictionDiagnosis(selectIndex(predValue, filterIndex), 
                          selectIndex(refValue, filterIndex),
                          quantityName,
                          az=az,
                          yMin = yMin,
                          yMax = yMax,
                          threshold = None)
      return
  
  predValueFiltered = predValue[predValue.index.get_level_values("Maturity") > 0.001]
  refValueFiltered = refValue[refValue.index.get_level_values("Maturity") > 0.001]
  title = "Predicted " + quantityName + " surface"
  plotSerie(predValueFiltered.rename(quantityName), 
            Title = title, 
            az=az,
            yMin = yMin,
            yMax = yMax)
  
  title = "True " + quantityName + " surface"
  plotSerie(refValueFiltered.rename(quantityName), 
            Title = title, 
            az=az,
            yMin = yMin,
            yMax = yMax)
  
  title = quantityName + " surface error"
  absoluteError = np.abs(predValueFiltered - refValueFiltered) 
  plotSerie(absoluteError.rename(quantityName + " Absolute Error"),
            Title = title,
            az=az,
            yMin = yMin,
            yMax = yMax)
  
  title = quantityName + " surface error"
  relativeError = np.abs(predValueFiltered - refValueFiltered) / refValueFiltered
  plotSerie(relativeError.rename(quantityName + " Relative Error (%)"),
            Title = title,
            az=az,
            yMin = yMin,
            yMax = yMax, 
            zAsPercent = True)
  
  print("RMSE : ", np.sqrt(np.mean(np.square(absoluteError))) )
  
  print("RMSE Relative: ", np.sqrt(np.mean(np.square(relativeError))) )
  
  return

def saveDataModel(predictedPrices, volLocal, impliedVol, name):
  data = np.vstack([predictedPrices.sort_index().values, volLocal.sort_index().values, impliedVol.sort_index().values]).T
  dataDf = pd.DataFrame(data, index = predictedPrices.sort_index().index, 
                        columns = ["Price", "LocalVolatility", "ImpliedVol"])
  dataDf.to_csv(name + ".csv")
  return
  
def removeDuplicateIndex(df):
    return selectIndex(df, df.index)

#Diagnose Price, theta, gamma and local volatility
def modelSummary(price, 
                 volLocale, 
                 delta_T, 
                 gamma_K, 
                 benchDataset,
                 S0,
                 bootstrap,
                 sigma=0.3, 
                 az=40,
                 yMin = 0,
                 yMax = 1,
                 thresholdPrice = None, 
                 removeNaN = False,
                 savePath = None):
    
  if thresholdPrice is not None :
     filterPrice = benchDataset["Price"] >= thresholdPrice
     keptPrices = benchDataset["Price"][filterPrice].index
     modelSummary(selectIndex(price, keptPrices),
                  selectIndex(volLocale, keptPrices),
                  selectIndex(delta_T, keptPrices),
                  selectIndex(gamma_K, keptPrices),
                  selectIndex(benchDataset, keptPrices),
                  S0,
                  bootstrap,
                  sigma=sigma,
                  az=az,
                  yMin = yMin,
                  yMax = yMax,
                  thresholdPrice = None, 
                  removeNaN = removeNaN,
                  savePath = None)
     return
  
  nbArbitrageViolations = ((delta_T < 0) + (gamma_K < 0)).sum()
  print("Number of static arbitrage violations : ", nbArbitrageViolations)
  print("Arbitrable total variance : ", price[((delta_T < 0) + (gamma_K < 0))])
  priceRef = benchDataset["Price"]
  predictionDiagnosis(price, 
                      priceRef, 
                      "Price",
                      az=320,
                      yMin = yMin,
                      yMax = yMax)
  
  volLocaleRef = benchDataset["locvol"] if "locvol" in benchDataset.columns else pd.Series(np.ones_like(priceRef),
                                                                                           index = priceRef.index)
  predictionDiagnosis(volLocale, 
                      volLocaleRef, 
                      "Local volatility",
                      az=az,
                      yMin = yMin,
                      yMax = yMax)
  
  dTRef = benchDataset["Theta"]
  predictionDiagnosis(delta_T, 
                      dTRef, 
                      "Theta",
                      az=340,
                      yMin = yMin,
                      yMax = yMax)
  
  gKRef = benchDataset["Gamma Strike"]
  predictionDiagnosis(gamma_K, 
                      gKRef, 
                      "Gamma Strike",
                      az=340,
                      yMin = yMin,
                      yMax = yMax)
  
  #Calibrate implied volatilities for each predicted price in testing set
  ImpVol = BS.vectorizedImpliedVolatilityCalibration(S0, bootstrap,
                                                     benchDataset["Maturity"],
                                                     benchDataset["Strike"],
                                                     benchDataset["OptionType"],
                                                     price, 
                                                     removeNaN = removeNaN)
  ImpVol = pd.Series(ImpVol, index = price.index).sort_index().dropna()
  
  predictionDiagnosis(ImpVol, 
                      selectIndex(benchDataset['ImpliedVol'], ImpVol.index),
                      " Implied vol ",
                      yMin=yMin,
                      yMax=yMax,
                      az = az)
  if savePath is not None : 
      saveDataModel(removeDuplicateIndex(price), 
                    removeDuplicateIndex(volLocale), 
                    removeDuplicateIndex(ImpVol), savePath)  
  return



######################################################################### Implied volatility

#Execute calibration of implied volatility from estimated price and benchmark price
#Then plot esitmated implied vol, absolute and relative error
def plotImpliedVol(priceSurface, 
                   refImpliedVol, 
                   bootstrap,
                   az=40,
                   yMin = 0,
                   yMax = 1,
                   relativeErrorVolMax = 10, 
                   removeNaN = False):
    return plotImpliedVolConcrete(priceSurface[priceSurface.index.get_level_values("Maturity") > 0.001],
                                  refImpliedVol[refImpliedVol.index.get_level_values("Maturity") > 0.001],
                                  bootstrap = bootstrap,
                                  az=az,
                                  yMin = yMin,
                                  yMax = yMax,
                                  relativeErrorVolMax = relativeErrorVolMax, 
                                  removeNaN = removeNaN)

def plotImpliedVolConcrete(priceSurface,
                           refImpliedVol,
                           bootstrap,
                           az=40,
                           yMin = 0,
                           yMax = 1,
                           relativeErrorVolMax = 10, 
                           removeNaN = False):
    df = priceSurface.index.to_frame()
    df["Price"] = priceSurface

    epsilon = 1e-9
    calibrationFunction = lambda x : BS.bissectionMethod(S0,
                                                         bootstrap.discountIntegral(x["Maturity"])/x["Maturity"],
                                                         bootstrap.dividendIntegral(x["Maturity"])/x["Maturity"],
                                                         0.2,
                                                         x["Maturity"],
                                                         x["Strike"],
                                                         x["Price"],
                                                         epsilon,
                                                         x["OptionType"],
                                                         removeNaN = removeNaN)[1]

    impliedVol = df.apply(calibrationFunction, axis = 1).rename("Implied Volatility")
    impliedVolError = np.abs(impliedVol-refImpliedVol).rename('Absolute Error')
    relativeImpliedVolError = (impliedVolError / refImpliedVol).rename("Relative error (%)")
    
    plotSerie(impliedVol.dropna(), 
              Title = 'Implied volatility surface', 
              az=az,
              yMin = yMin,
              yMax = yMax)

    plotSerie(impliedVolError.dropna(), 
              Title = 'Implied volatility error', 
              az=az,
              yMin = yMin,
              yMax = yMax)
    
    plotSerie(relativeImpliedVolError.clip(0, relativeErrorVolMax / 100.0).dropna(),
              Title = 'Implied volatility relative error', 
              az=az,
              yMin = yMin,
              yMax = yMax,
              zAsPercent = True)
  
    print("Implied volalitity RMSE : ", np.sqrt(np.nanmean(np.square(impliedVolError))) )

    return impliedVol

def plotEachStrike(df):
    for strike in df.rename({"Strike" : "StrikeColumn"}, axis=1).groupby("StrikeColumn"):
        if True : # strike[0] == df["logMoneyness"].min() :
            impliedTotVariance = np.square(strike[1][impliedVolColumn]) * strike[1]["Maturity"]
            plt.plot(strike[1]["Maturity"].values, impliedTotVariance.values, label = str(strike[0]))
    plt.legend()
    return

def plotEachSmile(df):
    for smile in df.rename({"Maturity" : "MaturityColumn"}, axis=1).groupby("MaturityColumn"):
        impliedTotVariance = np.square(smile[1][impliedVolColumn]) * smile[1]["MaturityColumn"]
        plt.plot(smile[1]["logMoneyness"].values, impliedTotVariance.values, label = str(smile[0]))
    plt.legend()
    return

def plotEachSmilePred(df, interpMethod):
    pred = interpMethod(df)
    for smile in pred.groupby("Maturity"):
        maturities = smile[1].index.get_level_values("Maturity")
        impliedTotVariance = np.multiply(np.square(smile[1].values), maturities)
        plt.plot(df[df["Maturity"]==smile[0]]["logMoneyness"],
                 impliedTotVariance,
                 label = str(smile[0]))
    plt.legend()
    return


def plotImpliedVolPrices(totalVariance, bootstrap, S0, benchDataset, 
                         logMoneynessScale = False,
                         yMin=0,
                         yMax=1, 
                         thresholdPrice = None):
  MaturityPred = totalVariance.index.get_level_values("Maturity")
  StrikePred = totalVariance.index.get_level_values("Strike")
  impliedVolPred = np.sqrt(totalVariance / MaturityPred)
  def priceFromImpliedVolatility(vol, bootstrap, T, K, S0):
    q = bootstrap.dividendIntegral(T) / T
    r = bootstrap.discountIntegral(T) / T
    return BS.bsformula( -1, S0, K, r, T, vol, q=q)
  resP = list(map(lambda x : priceFromImpliedVolatility(x[0], bootstrap, x[1], x[2], S0)[0],
                  zip(impliedVolPred.values, MaturityPred, StrikePred)))
  
  priceImpli = pd.Series(resP, index = totalVariance.index).rename("Price")
  refDataset = selectIndex(benchDataset, totalVariance.index) 
  if logMoneynessScale:
    if S0 < 0 :
      raise Exception("Precise a correct the spot underlying value ")
    pricePred = convertToLogMoneyness(priceImpli, S0)
    benchDatasetScaled = convertToLogMoneyness(refDataset, S0)
    yMinScaled = np.log(S0 / yMax)
    yMaxScaled = np.log(S0 / yMin)
    azimutIncrement = 180
  else:
    pricePred = priceImpli
    benchDatasetScaled = refDataset
    yMinScaled = yMin
    yMaxScaled = yMax
    azimutIncrement = 0
  
  priceRef = benchDatasetScaled["Price"]
  predictionDiagnosis(pricePred,
                      priceRef,
                      "Price",
                      az=320 + azimutIncrement,
                      yMin=yMinScaled,
                      yMax=yMaxScaled, 
                      threshold = thresholdPrice)
  return priceImpli



# Diagnose Price, theta, gamma and local volatility
def modelSummaryGatheral(totalVariance,
                         volLocale,
                         delta_T,
                         gamma_K,
                         benchDataset,
                         sigma=0.3,
                         az=40,
                         yMin=0,
                         yMax=1,
                         logMoneynessScale=False,
                         S0 = -1,
                         thresholdPrice = None,
                         bootstrap = None,
                         savePath = None):
    
    if thresholdPrice is not None :
        filterPrice = benchDataset["Price"] >= thresholdPrice
        keptPrices = benchDataset["Price"][filterPrice].index
        modelSummaryGatheral(selectIndex(totalVariance, keptPrices) ,
                             selectIndex(volLocale, keptPrices),
                             selectIndex(delta_T, keptPrices),
                             selectIndex(gamma_K, keptPrices),
                             selectIndex(benchDataset, keptPrices),
                             sigma=sigma,
                             az=az,
                             yMin = yMin,
                             yMax = yMax,
                             logMoneynessScale=logMoneynessScale,
                             S0 = S0,
                             thresholdPrice = None,
                             bootstrap = bootstrap,
                             savePath = savePath)
        return
    
    nbArbitrageViolations = ((delta_T < 0) + (gamma_K < 0)).sum()
    print("Number of static arbitrage violations : ", nbArbitrageViolations)
    print("Arbitrable total variance : ", totalVariance[((delta_T < 0) + (gamma_K < 0))])
    
    
    
    refDataset = selectIndex(benchDataset, totalVariance.index) 
    if logMoneynessScale:
        if S0 < 0 :
            raise Exception("Precise a correct the spot underlying value ")
        totalVariancePred = convertToLogMoneyness(totalVariance, S0)
        volLocalePred = convertToLogMoneyness(volLocale, S0)
        delta_TPred = convertToLogMoneyness(delta_T, S0)
        gKRefPred = convertToLogMoneyness(gamma_K, S0)
        benchDatasetScaled = convertToLogMoneyness(refDataset, S0)
        yMinScaled = np.log(S0 / yMax)
        yMaxScaled = np.log(S0 / yMin)
        azimutIncrement = 180
    else:
        totalVariancePred = totalVariance
        volLocalePred = volLocale
        delta_TPred = delta_T
        gKRefPred = gamma_K
        benchDatasetScaled = refDataset
        yMinScaled = yMin
        yMaxScaled = yMax
        azimutIncrement = 0

    totalVarianceRef = benchDatasetScaled["impliedTotalVariance"]
    predictionDiagnosis(totalVariancePred,
                        totalVarianceRef,
                        "Implied Variance",
                        az=320 + azimutIncrement,
                        yMin=yMinScaled,
                        yMax=yMaxScaled)

    volLocaleRef = benchDatasetScaled["locvol"]
    predictionDiagnosis(volLocalePred,
                        volLocaleRef,
                        "Local volatility",
                        az=az + azimutIncrement,
                        yMin=yMinScaled,
                        yMax=yMaxScaled)

    impliedVolPred = np.sqrt(totalVariancePred / benchDatasetScaled["Maturity"]) # np.sqrt(totalVariance / refDataset["Maturity"])
    predictionDiagnosis(impliedVolPred,
                        benchDatasetScaled[impliedVolColumn],
                        "Implied volatility",
                        az=az + azimutIncrement,
                        yMin=yMinScaled,
                        yMax=yMaxScaled)

    dTRef = benchDatasetScaled["Theta"]
    predictionDiagnosis(delta_TPred,
                        dTRef,
                        "Theta",
                        az=340 + azimutIncrement,
                        yMin=yMinScaled,
                        yMax=yMaxScaled)

    gKRef = benchDatasetScaled["Gamma Strike"]
    predictionDiagnosis(gKRefPred,
                        gKRef,
                        "Density",
                        az=340 + azimutIncrement,
                        yMin=yMinScaled,
                        yMax=yMaxScaled)
    
    
    ImpPrice = plotImpliedVolPrices(totalVariance, bootstrap, S0, benchDataset,
                                    logMoneynessScale = logMoneynessScale,
                                    yMin=yMin,
                                    yMax=yMax)
    if savePath is not None : 
        saveDataModel(ImpPrice[ ~ImpPrice.index.duplicated(keep='first') ], 
                      volLocalePred[ ~volLocalePred.index.duplicated(keep='first') ], 
                      impliedVolPred[ ~impliedVolPred.index.duplicated(keep='first') ], 
                      savePath)  
    return



# Plotting function for surface
# xTitle : title for x axis
# yTitle : title for y axis
# zTitle : title for z axis
# Title : plot title
# az : azimuth i.e. angle of view for surface
# yMin : minimum value for y axis
# yMax : maximum value for y axis
# zAsPercent : boolean, if true format zaxis as percentage
def plot2GridCustomWithViolation(coordinates, zValue,
                                 coordinates2, zValue2,
                                 coordinatesViolation, zViolationValue,
                                 coordinatesViolation2, zViolationValue2,
                                 xTitle="Maturity",
                                 yTitle="Strike",
                                 zTitle="Price",
                                 Title='True Price Surface',
                                 az=320,
                                 yMin=0,
                                 yMax=1,
                                 zAsPercent=False):
    y = coordinates[:, 0]
    filteredValue = (y > yMin) & (y < yMax)
    x = coordinates[:, 1][filteredValue]
    y = coordinates[:, 0][filteredValue]
    z = zValue[filteredValue].flatten()

    y2 = coordinates2[:, 0]
    filteredValue2 = (y2 > yMin) & (y2 < yMax)
    x2 = coordinates2[:, 1][filteredValue2]
    y2 = coordinates2[:, 0][filteredValue2]
    z2 = zValue2[filteredValue2].flatten()

    y3 = coordinatesViolation[:, 0]
    filteredValue3 = (y3 > yMin) & (y3 < yMax)
    x3 = coordinatesViolation[:, 1][filteredValue3]
    y3 = coordinatesViolation[:, 0][filteredValue3]
    z3 = zViolationValue[filteredValue3].flatten()

    y4 = coordinatesViolation2[:, 0]
    filteredValue4 = (y4 > yMin) & (y4 < yMax)
    x4 = coordinatesViolation2[:, 1][filteredValue4]
    y4 = coordinatesViolation2[:, 0][filteredValue4]
    z4 = zViolationValue2[filteredValue4].flatten()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection='3d')

    ax.set_xlabel(xTitle, fontsize=18, labelpad=20)
    ax.set_ylabel(yTitle, fontsize=18, labelpad=20)
    ax.set_zlabel(zTitle, fontsize=18, labelpad=10)

    cmap = plt.get_cmap("inferno")
    colors = cmap(z * 100 if zAsPercent else z)[np.newaxis, :, :3]
    ax.scatter(x2, y2, z2, marker='o', color="r", alpha=1, s=40)
    #ax.scatter(x, y, z, marker='o', color="b", alpha=1, s=40)
    ax.scatter(x3, y3, z3, marker='P', color="k", alpha=1, s=200)
    ax.scatter(x4, y4, z4, marker='P', color="b", alpha=1, s=200)

    surf = ax.plot_trisurf(x, y,
                          z * 100 if zAsPercent else z ,
                          linewidth=1.0,
                          antialiased=True,
                          cmap = cmap,
                          color=(0,0,0,0))
    scaleEdgeValue = surf.to_rgba(surf.get_array())
    surf.set_edgecolors(scaleEdgeValue)
    surf.set_alpha(0)

    if zAsPercent:
        ax.zaxis.set_major_formatter(mtick.PercentFormatter())
    ax.view_init(elev=10., azim=az)
    ax.set_title(Title, fontsize=24)
    ax.set_facecolor('white')

    plt.tick_params(labelsize=16)

    plt.show()

    return


# Plotting function from a pandas series
def plot2SeriesWithViolation(data,
                             data2,
                             dataViolation,
                             dataViolation2,
                             Title='True Price Surface',
                             az=320,
                             yMin=0,
                             yMax=1,
                             zAsPercent=False):
    plot2GridCustomWithViolation(data.index.to_frame().values,
                                 data.values,
                                 data2.index.to_frame().values,
                                 data2.values,
                                 dataViolation.index.to_frame().values,
                                 dataViolation.values,
                                 dataViolation2.index.to_frame().values,
                                 dataViolation2.values,
                                 xTitle=data.index.names[1],
                                 yTitle=data.index.names[0],
                                 zTitle=data.name,
                                 Title=Title,
                                 az=az,
                                 yMin=yMin,
                                 yMax=yMax,
                                 zAsPercent=zAsPercent)
    return

# Plotting function for surface
# xTitle : title for x axis
# yTitle : title for y axis
# zTitle : title for z axis
# Title : plot title
# az : azimuth i.e. angle of view for surface
# yMin : minimum value for y axis
# yMax : maximum value for y axis
# zAsPercent : boolean, if true format zaxis as percentage
def plotGridCustomWithViolation(coordinates, zValue,
                                coordinatesViolation, zViolationValue,
                                xTitle="Maturity",
                                yTitle="Strike",
                                zTitle="Price",
                                Title='True Price Surface',
                                az=320,
                                yMin=0,
                                yMax=1,
                                zAsPercent=False):
    y = coordinates[:, 0]
    filteredValue = (y > yMin) & (y < yMax)
    x = coordinates[:, 1][filteredValue]
    y = coordinates[:, 0][filteredValue]
    z = zValue[filteredValue].flatten()

    y3 = coordinatesViolation[:, 0]
    filteredValue3 = (y3 > yMin) & (y3 < yMax)
    x3 = coordinatesViolation[:, 1][filteredValue3]
    y3 = coordinatesViolation[:, 0][filteredValue3]
    z3 = zViolationValue[filteredValue3].flatten()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection='3d')

    ax.set_xlabel(xTitle, fontsize=18, labelpad=20)
    ax.set_ylabel(yTitle, fontsize=18, labelpad=20)
    ax.set_zlabel(zTitle, fontsize=18, labelpad=10)

    cmap = plt.get_cmap("inferno")
    colors = cmap(z * 100 if zAsPercent else z)[np.newaxis, :, :3]
    ax.scatter(x, y, z, marker='o', color="b", alpha=1, s=40)
    ax.scatter(x3, y3, z3, marker='P', color="k", alpha=1, s=200)
    # surf = ax.plot_trisurf(x, y,
    #                       z * 100 if zAsPercent else z ,
    #                       linewidth=1.0,
    #                       antialiased=True,
    #                       cmap = cmap,
    #                       color=(0,0,0,0))
    # scaleEdgeValue = surf.to_rgba(surf.get_array())
    # surf.set_edgecolors(scaleEdgeValue)
    # surf.set_alpha(0)

    if zAsPercent:
        ax.zaxis.set_major_formatter(mtick.PercentFormatter())
    ax.view_init(elev=10., azim=az)
    ax.set_title(Title, fontsize=24)
    ax.set_facecolor('white')

    plt.tick_params(labelsize=16)

    plt.show()

    return


# Plotting function from a pandas series
def plotSeriesWithViolation(data,
                            dataViolation,
                            Title='True Price Surface',
                            az=320,
                            yMin=0,
                            yMax=1,
                            zAsPercent=False):
    plotGridCustomWithViolation(data.index.to_frame().values,
                                data.values,
                                data[dataViolation].index.to_frame().values,
                                data[dataViolation].values,
                                xTitle=data.index.names[1],
                                yTitle=data.index.names[0],
                                zTitle=data.name,
                                Title=Title,
                                az=az,
                                yMin=yMin,
                                yMax=yMax,
                                zAsPercent=zAsPercent)
    return

def diagnoseLocalVol(dT,
                     locVolSSVI,
                     density,
                     interpolatedImpliedVol,
                     benchDataSet,
                     az=320,
                     yMin=0,
                     yMax=1,
                     zAsPercent=False):
    #Number of violations
    arbitrageViolation = np.logical_or(dT < 0, density < 0)
    print("Number of arbitrage violations : ", arbitrageViolation.sum())
    print("Arbitrable volatility : ", benchDataSet[impliedVolColumn][arbitrageViolation])
    print("Interpolated Arbitrable volatility : ", interpolatedImpliedVol[arbitrageViolation])
    plotSeriesWithViolation(dT,
                            arbitrageViolation,
                            Title='Numerator - dT',
                            az=az,
                            yMin=yMin,
                            yMax=yMax,
                            zAsPercent=zAsPercent)
    plotSeriesWithViolation(density,
                            arbitrageViolation,
                            Title='Denominator - density',
                            az=320,
                            yMin=az,
                            yMax=yMax,
                            zAsPercent=zAsPercent)
    plot2SeriesWithViolation(benchDataSet[impliedVolColumn],
                             benchDataSet[impliedVolColumn][arbitrageViolation],
                             interpolatedImpliedVol,
                             interpolatedImpliedVol[arbitrageViolation],
                             Title='Implied and arbitrage Violations',
                             az=az,
                             yMin=yMin,
                             yMax=yMax,
                             zAsPercent=zAsPercent)
    plotSeriesWithViolation(locVolSSVI,
                            arbitrageViolation,
                            Title='Local Volatiltiy',
                            az=az,
                            yMin=yMin,
                            yMax=yMax,
                            zAsPercent=zAsPercent)
    return

def roundMultiIndex(formerIndex):
  roundedValues = np.round(formerIndex.get_level_values("Maturity"), decimals=4)
  newIndex = pd.MultiIndex.from_tuples( list(zip(formerIndex.get_level_values("Strike"), roundedValues)) )
  return newIndex[~newIndex.duplicated()]

def plot2dSmiles(SSVIResults, 
                 GPResults, 
                 NeuralResults, 
                 trainingSet,
                 testingSet,
                 plotMarketData = True,
                 nbObservationThreshold = 0,
                 maturityList = None,
                 showDiff = False,
                 useLogMoneyness = True,
                 gpQuantiles = None,
                 legend = False):

    impVol = SSVIResults["ImpliedVol"] if SSVIResults is not None else None
    impVolGP = GPResults["ImpliedVol"] if GPResults is not None else None
    impVolNN = NeuralResults["ImpliedVol"] if NeuralResults is not None else None
    
    if (trainingSet is not None) and (testingSet is not None) :
        concatDf = pd.concat([trainingSet, testingSet]).sort_index()
        concatDfTrain = trainingSet
        concatDfTrain.index = roundMultiIndex(concatDfTrain.index).rename(["Strike", "Maturity"])
        concatDfTest = testingSet
        concatDfTest.index = roundMultiIndex(concatDfTest.index).rename(["Strike", "Maturity"])
    elif (trainingSet is not None) :
        concatDf = trainingSet
        concatDfTrain = trainingSet
        concatDfTrain.index = roundMultiIndex(concatDfTrain.index).rename(["Strike", "Maturity"])
    else :
        concatDf = testingSet
        concatDfTest = testingSet
        concatDfTest.index = roundMultiIndex(concatDfTest.index).rename(["Strike", "Maturity"])
    
    concatDf = concatDf[~concatDf.index.duplicated(keep='first')]
    concatDf.index = roundMultiIndex(concatDf.index).rename(["Strike", "Maturity"])
    if showDiff :
        impVol = (impVol - (concatDf[impliedVolColumn]).loc[impVol.index]) if SSVIResults is not None else None
        impVolGP = (impVolGP - (concatDf[impliedVolColumn]).loc[impVol.index]) if GPResults is not None else None
        impVolNN = (impVolNN - (concatDf[impliedVolColumn]).loc[impVol.index]) if NeuralResults is not None else None
    
    
    #Select maturities for which the smile has at least nbObservationThreshold points
    if maturityList is None :
        maturitiesCount = concatDf.groupby(level = "Maturity").count()["Price"]
        maturities = maturitiesCount[maturitiesCount >= nbObservationThreshold].index.get_level_values("Maturity").unique() #[1::2]
    else :
        maturities = np.array(maturityList) 
    #maturities = maturities.insert(0,maturitiesCount[maturitiesCount >= nbObservationThreshold].index.get_level_values("Maturity").unique()[0])
    maturities = np.sort(maturities)
    #maturities = maturities.insert(0,maturitiesCount[maturitiesCount >= nbObservationThreshold].index.get_level_values("Maturity").unique()[0])
    np.sort(maturities)
    nbMaturities = maturities.size
    
    
    widthPlot = 4#math.ceil(nbMaturities/heightPlot)
    heightPlot = math.ceil(nbMaturities/widthPlot)#int(np.sqrt(nbMaturities)) +  1 #2 
    nbFrame = heightPlot * widthPlot 

    fig, axs = plt.subplots(heightPlot, widthPlot,figsize=(40,20))
    fig.subplots_adjust( wspace=0.2, hspace=0.4)
    #fig.suptitle('Implied volatility calibrated', fontsize=20)
    plotList = []

    for k in range(nbMaturities):
      dataFiltered = concatDf[concatDf.index.get_level_values("Maturity") == maturities[k]]
      curveBid = dataFiltered["ImpVolBid"]
      curveAsk = dataFiltered["ImpVolAsk"]
      xCoordinate = dataFiltered["logMoneyness"] if useLogMoneyness else dataFiltered["Strike"]
      
      if trainingSet is not None :
          trainFiltered = concatDfTrain[concatDfTrain.index.get_level_values("Maturity") == maturities[k]]
          trainingSpread = trainFiltered["ImpVolBid"].append(trainFiltered["ImpVolAsk"]) 
          trainingCoordinate = trainFiltered["logMoneyness"] if useLogMoneyness else trainFiltered["Strike"]
          trainingCoordinate = trainingCoordinate.loc[trainingSpread.index]
      
      if testingSet is not None :
          testFiltered = concatDfTest[concatDfTest.index.get_level_values("Maturity") == maturities[k]]
          testingSpread = testFiltered["ImpVolBid"].append(testFiltered["ImpVolAsk"]) 
          testingCoordinate = testFiltered["logMoneyness"] if useLogMoneyness else testFiltered["Strike"]
          testingCoordinate = testingCoordinate.loc[testingSpread.index]
      
      if plotMarketData :
        curveQuote = dataFiltered[impliedVolColumn]
      if showDiff :
          curveBid = (curveBid - dataFiltered[impliedVolColumn]) 
          curveAsk = (curveAsk - dataFiltered[impliedVolColumn]) 
          
          trainingSpread = trainingSpread - dataFiltered[impliedVolColumn].loc[trainingSpread.index]
          testingSpread = testingSpread - dataFiltered[impliedVolColumn].loc[testingSpread.index]
      

      x = k // widthPlot
      y = k % widthPlot

      #axs[x,y].set_ylim([0, 0.4])
      if SSVIResults is not None :
        curveSSVI = impVol.loc[dataFiltered[impliedVolColumn].index]
        plotList.append(axs[x,y].plot(xCoordinate, curveSSVI.values, "k-", label = "SSVI"))
      if GPResults is not None :
        curveGP = impVolGP.loc[dataFiltered[impliedVolColumn].index] 
        plotList.append(axs[x,y].plot(xCoordinate, curveGP.values, "g-", label = "GP"))
      if NeuralResults is not None :
        curveNN = impVolNN.loc[dataFiltered[impliedVolColumn].index] 
        plotList.append(axs[x,y].plot(xCoordinate, curveNN.values, "m-", label = "NN"))
      
      if plotMarketData and (not showDiff) :
        plotList.append(axs[x,y].plot(xCoordinate, curveQuote.values, "k+", label = "Mid"))
      
      marker = "o"
      if trainingSet is not None :
        axs[x,y].plot(trainingCoordinate.values, trainingSpread.values, "b" + marker)
      if testingSet is not None :
        axs[x,y].plot(testingCoordinate.values, testingSpread.values, "r" + marker)
      #axs[x,y].fill_between(xCoordinate, curveBid.values, curveAsk.values, alpha = 0.3)
      if gpQuantiles is not None : 
        axs[x,y].fill_between(xCoordinate, 
                              gpQuantiles[0].loc[dataFiltered[impliedVolColumn].index], 
                              gpQuantiles[1].loc[dataFiltered[impliedVolColumn].index], 
                              alpha = 0.3)
      axs[x,y].tick_params(labelsize=26)
      axs[x,y].set_title('Maturity : ' + str(round(maturities[k], 4)), fontsize=40)
      axs[x,y].set_facecolor('white')
      for spine in axs[x,y].spines.values():
        spine.set_visible(True)
        spine.set_color("k")
    
    labels = []
    if SSVIResults is not None :
        labels.append(SSVIResults.name)
    if GPResults is not None :
        labels.append(GPResults.name)
    if NeuralResults is not None :
        labels.append(NeuralResults.name)
    if plotMarketData and (not showDiff):
        labels.append("Mid")
    #labels = labels + ["Ask", "Bid"]
    if legend :
        nbDeleted = 0
        for k in range(nbMaturities, nbFrame):
          x = k // widthPlot
          y = k % widthPlot
          if nbDeleted == 0 : 
            chartBox = axs[x,y].get_position()
            fig.delaxes(axs[x,y])
            fig.legend(plotList[-len(labels):],     # The line objects
                       labels= labels,   # The labels for each line
                       loc="lower left",   # Position of legend
                       borderaxespad=0.1,    # Small spacing around legend box
                       title=None,  # Title for the legend
                       fontsize = '20',
                       title_fontsize = '20',
                       bbox_to_anchor=[chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
          
          nbDeleted = nbDeleted + 1
    plt.show()

def plot2dPriceSmiles(SSVIResults, 
                 GPResults, 
                 NeuralResults, 
                 trainingSet,
                 testingSet,
                 plotMarketData = True,
                 nbObservationThreshold = 0,
                 maturityList = None,
                 showDiff = False,
                 useLogMoneyness = False,
                 gpQuantiles = None,
                 legend = False):

    impPrice = SSVIResults["Price"] if SSVIResults is not None else None
    impPriceGP = GPResults["Price"] if GPResults is not None else None
    impPriceNN = NeuralResults["Price"] if NeuralResults is not None else None
    
    if (trainingSet is not None) and (testingSet is not None) :
        concatDf = pd.concat([trainingSet, testingSet]).sort_index()
        concatDfTrain = trainingSet
        concatDfTrain.index = roundMultiIndex(concatDfTrain.index).rename(["Strike", "Maturity"])
        concatDfTest = testingSet
        concatDfTest.index = roundMultiIndex(concatDfTest.index).rename(["Strike", "Maturity"])
    elif (trainingSet is not None) :
        concatDf = trainingSet
        concatDfTrain = trainingSet
        concatDfTrain.index = roundMultiIndex(concatDfTrain.index).rename(["Strike", "Maturity"])
    else :
        concatDf = testingSet
        concatDfTest = testingSet
        concatDfTest.index = roundMultiIndex(concatDfTest.index).rename(["Strike", "Maturity"])
    
    concatDf = concatDf[~concatDf.index.duplicated(keep='first')]
    concatDf.index = roundMultiIndex(concatDf.index).rename(["Strike", "Maturity"])
    if showDiff :
        impPrice = (impPrice - (concatDf["Price"]).loc[impPrice.index]) if SSVIResults is not None else None
        impPriceGP = (impPriceGP - (concatDf["Price"]).loc[impPrice.index]) if GPResults is not None else None
        impPriceNN = (impPriceNN - (concatDf["Price"]).loc[impPrice.index]) if NeuralResults is not None else None
    
    
    #Select maturities for which the smile has at least nbObservationThreshold points
    if maturityList is None :
        maturitiesCount = concatDf.groupby(level = "Maturity").count()["Price"]
        maturities = maturitiesCount[maturitiesCount >= nbObservationThreshold].index.get_level_values("Maturity").unique() #[1::2]
    else :
        maturities = np.array(maturityList) 
    #maturities = maturities.insert(0,maturitiesCount[maturitiesCount >= nbObservationThreshold].index.get_level_values("Maturity").unique()[0])
    maturities = np.sort(maturities)
    
    
    widthPlot = 4#math.ceil(nbMaturities/heightPlot)
    heightPlot = math.ceil(nbMaturities/widthPlot)#int(np.sqrt(nbMaturities)) +  1 #2 
    nbFrame = heightPlot * widthPlot 

    fig, axs = plt.subplots(heightPlot, widthPlot,figsize=(40,20))
    fig.subplots_adjust( wspace=0.2, hspace=0.4)
    #fig.suptitle('Implied volatility calibrated', fontsize=20)
    plotList = []

    for k in range(nbMaturities):
      dataFiltered = concatDf[concatDf.index.get_level_values("Maturity") == maturities[k]]
      curveBid = dataFiltered["Bid"]
      curveAsk = dataFiltered["Ask"]
      xCoordinate = dataFiltered["logMoneyness"] if useLogMoneyness else dataFiltered["Strike"]
      
      if trainingSet is not None :
          trainFiltered = concatDfTrain[concatDfTrain.index.get_level_values("Maturity") == maturities[k]]
          trainingSpread = trainFiltered["Bid"].append(trainFiltered["Ask"]) 
          trainingCoordinate = trainFiltered["logMoneyness"] if useLogMoneyness else trainFiltered["Strike"]
          trainingCoordinate = trainingCoordinate.loc[trainingSpread.index]
      
      if testingSet is not None :
          testFiltered = concatDfTest[concatDfTest.index.get_level_values("Maturity") == maturities[k]]
          testingSpread = testFiltered["Bid"].append(testFiltered["Ask"]) 
          testingCoordinate = testFiltered["logMoneyness"] if useLogMoneyness else testFiltered["Strike"]
          testingCoordinate = testingCoordinate.loc[testingSpread.index]
      
      if plotMarketData :
        curveQuote = dataFiltered["Price"]
      if showDiff :
          curveBid = (curveBid - dataFiltered["Price"]) 
          curveAsk = (curveAsk - dataFiltered["Price"]) 
          
          trainingSpread = trainingSpread - dataFiltered["Price"].loc[trainingSpread.index]
          testingSpread = testingSpread - dataFiltered["Price"].loc[testingSpread.index]
      

      x = k // widthPlot
      y = k % widthPlot

      #axs[x,y].set_ylim([0, 0.4])
      if SSVIResults is not None :
        curveSSVI = impPrice.loc[dataFiltered["Price"].index]
        plotList.append(axs[x,y].plot(xCoordinate, curveSSVI.values, "k-", label = "SSVI"))
      if GPResults is not None :
        curveGP = impPriceGP.loc[dataFiltered["Price"].index] 
        plotList.append(axs[x,y].plot(xCoordinate, curveGP.values, "g-", label = "GP"))
      if NeuralResults is not None :
        curveNN = impPriceNN.loc[dataFiltered["Price"].index] 
        plotList.append(axs[x,y].plot(xCoordinate, curveNN.values, "m-", label = "NN"))
      
      if plotMarketData and (not showDiff) :
        plotList.append(axs[x,y].plot(xCoordinate, curveQuote.values, "k+", label = "Mid"))
      
      marker = "o"
      if trainingSet is not None :
        axs[x,y].plot(trainingCoordinate.values, trainingSpread.values, "b" + marker)
      if testingSet is not None :
        axs[x,y].plot(testingCoordinate.values, testingSpread.values, "r" + marker)
      #axs[x,y].fill_between(xCoordinate, curveBid.values, curveAsk.values, alpha = 0.3)
      if gpQuantiles is not None : 
        axs[x,y].fill_between(xCoordinate, 
                              gpQuantiles[0].loc[dataFiltered["Price"].index], 
                              gpQuantiles[1].loc[dataFiltered["Price"].index], 
                              alpha = 0.3)
      axs[x,y].tick_params(labelsize=26)
      axs[x,y].set_title('Maturity : ' + str(round(maturities[k], 4)), fontsize=40)
      axs[x,y].set_facecolor('white')
      for spine in axs[x,y].spines.values():
        spine.set_visible(True)
        spine.set_color("k")
    
    labels = []
    if SSVIResults is not None :
        labels.append(SSVIResults.name)
    if GPResults is not None :
        labels.append(GPResults.name)
    if NeuralResults is not None :
        labels.append(NeuralResults.name)
    if plotMarketData and (not showDiff):
        labels.append("Mid")
    #labels = labels + ["Ask", "Bid"]
    if legend :
        nbDeleted = 0
        for k in range(nbMaturities, nbFrame):
          x = k // widthPlot
          y = k % widthPlot
          if nbDeleted == 0 : 
            chartBox = axs[x,y].get_position()
            fig.delaxes(axs[x,y])
            fig.legend(plotList[-len(labels):],     # The line objects
                       labels= labels,   # The labels for each line
                       loc="lower left",   # Position of legend
                       borderaxespad=0.1,    # Small spacing around legend box
                       title=None,  # Title for the legend
                       fontsize = '20',
                       title_fontsize = '20',
                       bbox_to_anchor=[chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
          
          nbDeleted = nbDeleted + 1
    plt.show()

def plot2dSmilesTotalVariance(SSVIResults,
                              GPResults,
                              NeuralResults,
                              trainingSet,
                              testingSet,
                              plotMarketData = True,
                              nbObservationThreshold = 0,
                              maturityList = None,
                              showDiff = False,
                              useLogMoneyness = False,
                              gpQuantiles = None,
                              legend = False):
    impliedTotVarianceFunction = lambda x : x * x.index.get_level_values("Maturity")
    impVar = impliedTotVarianceFunction(SSVIResults["ImpliedVol"]) if SSVIResults is not None else None
    impVarGP = impliedTotVarianceFunction(GPResults["ImpliedVol"]) if GPResults is not None else None
    impVarNN = impliedTotVarianceFunction(NeuralResults["ImpliedVol"]) if NeuralResults is not None else None
    
    if (trainingSet is not None) and (testingSet is not None) :
        concatDf = pd.concat([trainingSet, testingSet]).sort_index()
        concatDfTrain = trainingSet
        concatDfTrain.index = roundMultiIndex(concatDfTrain.index).rename(["Strike", "Maturity"])
        concatDfTest = testingSet
        concatDfTest.index = roundMultiIndex(concatDfTest.index).rename(["Strike", "Maturity"])
    elif (trainingSet is not None) :
        concatDf = trainingSet
        concatDfTrain = trainingSet
        concatDfTrain.index = roundMultiIndex(concatDfTrain.index).rename(["Strike", "Maturity"])
    else :
        concatDf = testingSet
        concatDfTest = testingSet
        concatDfTest.index = roundMultiIndex(concatDfTest.index).rename(["Strike", "Maturity"])
    
    concatDf = concatDf[~concatDf.index.duplicated(keep='first')]
    concatDf.index = roundMultiIndex(concatDf.index).rename(["Strike", "Maturity"])
    if showDiff :
        impVar = (impVar - impliedTotVarianceFunction(concatDf[impliedVolColumn]).loc[impVar.index]) if SSVIResults is not None else None
        impVarGP = (impVarGP - impliedTotVarianceFunction(concatDf[impliedVolColumn]).loc[impVar.index]) if GPResults is not None else None
        impVarNN = (impVarNN - impliedTotVarianceFunction(concatDf[impliedVolColumn]).loc[impVar.index]) if NeuralResults is not None else None
    
    
    #Select maturities for which the smile has at least nbObservationThreshold points
    if maturityList is None :
        maturitiesCount = concatDf.groupby(level = "Maturity").count()["Price"]
        maturities = maturitiesCount[maturitiesCount >= nbObservationThreshold].index.get_level_values("Maturity").unique() #[1::2]
    else :
        maturities = np.array(maturityList) 
    #maturities = maturities.insert(0,maturitiesCount[maturitiesCount >= nbObservationThreshold].index.get_level_values("Maturity").unique()[0])
    maturities = np.sort(maturities)
    nbMaturities = maturities.size
    
    
    widthPlot = 4#math.ceil(nbMaturities/heightPlot)
    heightPlot = math.ceil(nbMaturities/widthPlot)#int(np.sqrt(nbMaturities)) +  1 #2 
    nbFrame = heightPlot * widthPlot 

    fig, axs = plt.subplots(heightPlot, widthPlot,figsize=(40,20))
    fig.subplots_adjust( wspace=0.2, hspace=0.4)
    #fig.suptitle('Implied volatility calibrated', fontsize=20)
    plotList = []

    for k in range(nbMaturities):
      dataFiltered = concatDf[concatDf.index.get_level_values("Maturity") == maturities[k]]
      curveBid = impliedTotVarianceFunction(dataFiltered["ImpVolBid"])
      curveAsk = impliedTotVarianceFunction(dataFiltered["ImpVolAsk"])
      xCoordinate = dataFiltered["logMoneyness"] if useLogMoneyness else dataFiltered["Strike"]
      
      if trainingSet is not None :
          trainFiltered = concatDfTrain[concatDfTrain.index.get_level_values("Maturity") == maturities[k]]
          trainingSpread = impliedTotVarianceFunction(trainFiltered["ImpVolBid"].append(trainFiltered["ImpVolAsk"]))
          trainingCoordinate = trainFiltered["logMoneyness"] if useLogMoneyness else trainFiltered["Strike"]
          trainingCoordinate = trainingCoordinate.loc[trainingSpread.index]
      
      if testingSet is not None :
          testFiltered = concatDfTest[concatDfTest.index.get_level_values("Maturity") == maturities[k]]
          testingSpread = impliedTotVarianceFunction(testFiltered["ImpVolBid"].append(testFiltered["ImpVolAsk"]))
          testingCoordinate = testFiltered["logMoneyness"] if useLogMoneyness else testFiltered["Strike"]
          testingCoordinate = testingCoordinate.loc[testingSpread.index]
      
      if plotMarketData :
        curveQuote = impliedTotVarianceFunction(dataFiltered[impliedVolColumn])
      if showDiff :
          curveBid = (curveBid - impliedTotVarianceFunction(dataFiltered[impliedVolColumn])) 
          curveAsk = (curveAsk - impliedTotVarianceFunction(dataFiltered[impliedVolColumn])) 
          
          trainingSpread = trainingSpread - impliedTotVarianceFunction(dataFiltered[impliedVolColumn]).loc[trainingSpread.index]
          testingSpread = testingSpread - impliedTotVarianceFunction(dataFiltered[impliedVolColumn]).loc[testingSpread.index]
      

      x = k // widthPlot
      y = k % widthPlot

      #axs[x,y].set_ylim([0, 0.4])
      if SSVIResults is not None :
        curveSSVI = impVar.loc[dataFiltered[impliedVolColumn].index]
        plotList.append(axs[x,y].plot(xCoordinate, curveSSVI.values, "k-", label = "SSVI"))
      if GPResults is not None :
        curveGP = impVarGP.loc[dataFiltered[impliedVolColumn].index] 
        plotList.append(axs[x,y].plot(xCoordinate, curveGP.values, "g-", label = "GP"))
      if NeuralResults is not None :
        curveNN = impVarNN.loc[dataFiltered[impliedVolColumn].index] 
        plotList.append(axs[x,y].plot(xCoordinate, curveNN.values, "m-", label = "NN"))
      
      if plotMarketData and (not showDiff) :
        plotList.append(axs[x,y].plot(xCoordinate, curveQuote.values, "k+", label = "Mid"))
      
      marker = "o"
      if trainingSet is not None :
        axs[x,y].plot(trainingCoordinate.values, trainingSpread.values, "b" + marker)
      if testingSet is not None :
        axs[x,y].plot(testingCoordinate.values, testingSpread.values, "r" + marker)
      #axs[x,y].fill_between(xCoordinate, curveBid.values, curveAsk.values, alpha = 0.3)
      if gpQuantiles is not None : 
        axs[x,y].fill_between(xCoordinate, 
                              impliedTotVarianceFunction(gpQuantiles[0].loc[dataFiltered[impliedVolColumn].index]), 
                              impliedTotVarianceFunction(gpQuantiles[1].loc[dataFiltered[impliedVolColumn].index]), 
                              alpha = 0.3)
      axs[x,y].tick_params(labelsize=26)
      axs[x,y].set_title('Maturity : ' + str(round(maturities[k], 4)), fontsize=40)
      axs[x,y].set_facecolor('white')
      for spine in axs[x,y].spines.values():
        spine.set_visible(True)
        spine.set_color("k")
    
    labels = []
    if SSVIResults is not None :
        labels.append(SSVIResults.name)
    if GPResults is not None :
        labels.append(GPResults.name)
    if NeuralResults is not None :
        labels.append(NeuralResults.name)
    if plotMarketData and (not showDiff):
        labels.append("Mid")
    #labels = labels + ["Ask", "Bid"]
    if legend :
        nbDeleted = 0
        for k in range(nbMaturities, nbFrame):
          x = k // widthPlot
          y = k % widthPlot
          if nbDeleted == 0 : 
            chartBox = axs[x,y].get_position()
            fig.delaxes(axs[x,y])
            fig.legend(plotList[-len(labels):],     # The line objects
                       labels= labels,   # The labels for each line
                       loc="lower left",   # Position of legend
                       borderaxespad=0.1,    # Small spacing around legend box
                       title=None,  # Title for the legend
                       fontsize = '20',
                       title_fontsize = '20',
                       bbox_to_anchor=[chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
          
          nbDeleted = nbDeleted + 1
    plt.show()
