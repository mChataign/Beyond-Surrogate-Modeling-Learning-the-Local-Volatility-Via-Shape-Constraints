import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as mtick
import BS


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
  
  fig = plt.figure(figsize=(20,10))
  ax = fig.gca(projection='3d')
  
  ax.set_xlabel(xTitle, fontsize=18, labelpad=20)
  ax.set_ylabel(yTitle, fontsize=18, labelpad=20)
  ax.set_zlabel(zTitle, fontsize=18, labelpad=10)
  
  cmap=plt.get_cmap("inferno")
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
  ax.view_init(elev=10., azim=az)
  ax.set_title(Title, fontsize=24)
  ax.set_facecolor('white')

  plt.tick_params(labelsize=16)

  
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
                        yMax = 1):
  
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
  
  return

#Diagnose Price, theta, gamma and local volatility
def modelSummary(price, 
                 volLocale, 
                 delta_T, 
                 gamma_K, 
                 benchDataset,
                 sigma=0.3, 
                 az=40,
                 yMin = 0,
                 yMax = 1):
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
                   relativeErrorVolMax = 10):
    return plotImpliedVolConcrete(priceSurface[priceSurface.index.get_level_values("Maturity") > 0.001],
                                  refImpliedVol[refImpliedVol.index.get_level_values("Maturity") > 0.001],
                                  bootstrap = bootstrap,
                                  az=az,
                                  yMin = yMin,
                                  yMax = yMax,
                                  relativeErrorVolMax = relativeErrorVolMax)

def plotImpliedVolConcrete(priceSurface,
                           refImpliedVol,
                           bootstrap,
                           az=40,
                           yMin = 0,
                           yMax = 1,
                           relativeErrorVolMax = 10):
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
                                                         x["OptionType"])[1]

    impliedVol = df.apply(calibrationFunction, axis = 1).rename("Implied Volatility")
    impliedVolError = np.abs(impliedVol-refImpliedVol).rename('Absolute Error')
    relativeImpliedVolError = (impliedVolError / refImpliedVol).rename("Relative error (%)")
    
    plotSerie(impliedVol, 
              Title = 'Implied volatility surface', 
              az=az,
              yMin = yMin,
              yMax = yMax)

    plotSerie(impliedVolError, 
              Title = 'Implied volatility error', 
              az=az,
              yMin = yMin,
              yMax = yMax)
    
    plotSerie(relativeImpliedVolError.clip(0, relativeErrorVolMax / 100.0),
              Title = 'Implied volatility relative error', 
              az=az,
              yMin = yMin,
              yMax = yMax,
              zAsPercent = True)
  
    print("Implied volalitity RMSE : ", np.sqrt(np.mean(np.square(impliedVolError))) )

    return impliedVol

def plotEachStrike(df):
    for strike in df.rename({"Strike" : "StrikeColumn"}, axis=1).groupby("StrikeColumn"):
        if True : # strike[0] == df["logMoneyness"].min() :
            impliedTotVariance = np.square(strike[1]["ImpliedVol"]) * strike[1]["Maturity"]
            plt.plot(strike[1]["Maturity"].values, impliedTotVariance.values, label = str(strike[0]))
    plt.legend()
    return

def plotEachSmile(df):
    for smile in df.rename({"Maturity" : "MaturityColumn"}, axis=1).groupby("MaturityColumn"):
        impliedTotVariance = np.square(smile[1]["ImpliedVol"]) * smile[1]["MaturityColumn"]
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
                         logMoneynessScale=False):
    nbArbitrageViolations = ((delta_T < 0) + (gamma_K < 0)).sum()
    print("Number of static arbitrage violations : ", nbArbitrageViolations)
    print("Arbitrable total variance : ", totalVariance[((delta_T < 0) + (gamma_K < 0))])

    refDataset = benchDataset.loc[totalVariance.index]
    if logMoneynessScale:
        totalVariancePred = convertToLogMoneyness(totalVariance)
        volLocalePred = convertToLogMoneyness(volLocale)
        delta_TPred = convertToLogMoneyness(delta_T)
        gKRefPred = convertToLogMoneyness(gamma_K)
        benchDatasetScaled = convertToLogMoneyness(refDataset)
        yMinScaled = np.log(S0[0] / yMax)
        yMaxScaled = np.log(S0[0] / yMin)
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

    priceRef = benchDatasetScaled["impliedTotalVariance"]
    predictionDiagnosis(totalVariancePred,
                        priceRef,
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
                        benchDatasetScaled["ImpliedVol"],
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
    print("Arbitrable volatility : ", benchDataSet["ImpliedVol"][arbitrageViolation])
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
    plot2SeriesWithViolation(benchDataSet["ImpliedVol"],
                             benchDataSet["ImpliedVol"][arbitrageViolation],
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
