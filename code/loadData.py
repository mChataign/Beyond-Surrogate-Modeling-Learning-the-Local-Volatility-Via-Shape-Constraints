import pandas as pd
import numpy as np
import sys
import csv
from time import *
from datetime import *
from io import StringIO
import bootstrapping
import BS
import dataSetConstruction

def rmse(a,b):
    return np.sqrt(np.mean(np.square(a-b)))

################################################################################ Parsing dat files
def parseDatFile(fileName):
  s = open(fileName).read()
  
  defPos=s.find("[option]")
  finPos=s.find("[dividend]")
  df = pd.read_csv(StringIO(s[defPos:finPos].replace("\n\n",";").replace("\n",",").replace(";",";\n")),decimal=".", sep=",", header=None)
  
  matC = pd.to_numeric(df[1].str.split(pat="= ", expand=True)[1]).round(3)
  strikeC = pd.to_numeric(df[3].str.split(pat="= ", expand=True)[1]).round()
  priceC = pd.to_numeric(df[4].str.replace(";","").str.split(pat="= ", expand=True)[1])
  typeC = pd.to_numeric(df[2].str.split(pat="= ", expand=True)[1])
  formattedDat = pd.DataFrame([matC, strikeC, priceC, typeC], index = ["Maturity", "Strike", "Price", "Type"]).transpose().astype({"Type":"int32"})
  
  filteredDat = formattedDat[formattedDat["Type"]==2]
  return filteredDat

def parseModelParamDatFile(fileName):
    s = open(fileName).read()
    
    parts = s.split("\n\n")
    number1 = parts[0]
    repo = parts[1]
    dates = parts[2]
    interestRates = parts[3]
    dividendRates = parts[4]
    
    number2 = parts[5]
    number3 = parts[6]
    
    n = parts[7]
    sigmaRef = parts[8]
    h = parts[9]
    sigmaMax = parts[10]
    sigmaMin = parts[11]
    
    number4 = parts[12]
    underlying = parts[13]
    
    def splitRow(row):
        return np.array(row.split("\t")).astype(np.float)
    
    tree = ("\n".join(parts[14:])).split("\n")
    tree.remove("")
    formattedTree = np.reshape(np.array(list(map(splitRow, tree))), (-1,3))
    
    
    return pd.DataFrame(formattedTree, columns = ["date", "stock(%)", "vol"])

def parseImpliedVolDatFile(fileName):
    s = open(fileName).read()
    
    parts = s.split("\n\n")
    
    def splitRow(row):
        return np.array(row.split("\t")).astype(np.float)
    
    testGrid = ("\n".join(parts)).split("\n")
    testGrid.remove("")
    formattedTestGrid = np.reshape(np.array(list(map(splitRow, testGrid))), (-1,4))
    
    return pd.DataFrame(formattedTestGrid, columns=["Strike","Maturity","Implied vol.","Option price"])


def parseCalibrOutDatFile(fileName):
    s = open(fileName).read()
    
    parts = s.split("\n")
    
    def splitRow(row):
        return np.array(row.split("\t"))
    def filterRow(row):
        return len(row)==10
    def formatRow(row):
        return row.astype(np.float)
    
    #tree = ("\n".join(parts)).split("\n")
    #tree.remove("")
    filteredTrainingData = list(filter(filterRow , 
                                       list(map(splitRow, parts))))
    formattedTrainingData = np.array(list(map(formatRow, filteredTrainingData)))
    
    colNames = ["Active", "Option\ntype", "Maturity", "Strike", "Moneyness", 
                "Option\nprice", "Implied\nvol.", "Calibrated\nvol.","Market vol. -\nCalibrated vol.","MarketPrice"]
    dfTrainingData = pd.DataFrame(formattedTrainingData, columns = colNames)
    dfTrainingData["Active"] = dfTrainingData["Active"].astype(np.int) 
    dfTrainingData["Option\ntype"] = dfTrainingData["Option\ntype"].astype(np.int) 
    return dfTrainingData


def parseDatFiles(fileName):
    s = open(fileName).read()
    
    posUnderlying = s.find("[underlying]")
    posZeroCoupon = s.find("[zero_coupon]")
    posOption = s.find("[option]")
    posDividend = s.find("[dividend]")
    
    underlyingString = s[posUnderlying:posZeroCoupon]
    zeroCouponString = s[posZeroCoupon:posOption]
    optionString = s[posOption:posDividend]
    dividendString = s[posDividend:-2] 
    
    def extractData(subStr, tag):
        parts = subStr.replace(tag + "\n", "").split("\n\n")
        try :
            parts.remove("")
        except ValueError:
            #Not found, we continue
            pass
        
        def parseRow(row):
            return (int(row.split(" = ")[1]) if (row.split(" = ")[0] == "type") else float(row.split(" = ")[1]))
        
        def splitRow(row):
            table = np.array(row.split("\n"))
            parseTable = np.array(list(map(parseRow, table)))
            return np.reshape(parseTable, (-1))
        
        return np.array(list(map(splitRow, parts)))
    
    
    underlying = pd.DataFrame(extractData(underlyingString, "[underlying]"), 
                              columns=["S","Repo"])
    zeroCoupon = pd.DataFrame(extractData(zeroCouponString, "[zero_coupon] "), 
                              columns=["Maturity","Price"])
    option = pd.DataFrame(extractData(optionString, "[option] "), 
                          columns=["Maturity","Type", "Price", "Strike"])
    option["Type"] = option["Type"].astype(np.int) 
    dividend = pd.DataFrame(extractData(dividendString, "[dividend] "), 
                            columns=["Maturity","Amount"])
    return underlying, zeroCoupon, dividend, option


###################################################################### data processing
def cleanData(zeroCouponCurve,
              dividendCurve,
              trainingData,
              testingData,
              underlyingNative,
              localVolatilityNative):
    dividendDf = dividendCurve.set_index('Maturity').sort_index()
    dividendDf.loc[1.0] = 0.0
    dividendDf.sort_index(inplace=True)

    # Format zero coupon curve as a Pandas series
    rateCurveDf = zeroCouponCurve.set_index('Maturity').sort_index()
    # keep only rates expriring before 1 year
    rateCurveDf = rateCurveDf.loc[rateCurveDf.index <= 1.01]

    localVolatility = localVolatilityNative.dropna()

    strikeCol = np.multiply(localVolatility["stock(%)"],
                            underlyingNative).copy()
    localVolatility.insert(0, "Strike", strikeCol)

    roundedDate = localVolatility["date"].round(decimals=3)
    localVolatility = localVolatility.copy()
    localVolatility.loc[:, "date"] = roundedDate
    renameDict = {"date": "Maturity",
                  "vol": "LocalVolatility",
                  "stock(%)": "StrikePercentage"}
    localVolatility = localVolatility.rename(columns=renameDict).set_index(["Strike", "Maturity"])

    # Treatment for training data
    filteredTestingData = testingData[(testingData["Implied vol."] > 0) & (testingData["Option price"] > 0)]
    filteredTestingData = filteredTestingData.copy()
    filteredTestingData.loc[:, "Maturity"] = filteredTestingData["Maturity"].round(decimals=3)
    filteredTestingData.insert(0, "OptionType", np.ones_like(filteredTestingData["Maturity"]))
    renameDict = {"Implied vol.": "ImpliedVol",
                  "Option price": "Price",
                  "Implied delta": "ImpliedDelta",
                  "Implied gamma": "ImpliedGamma",
                  "Implied theta": "ImpliedTheta",
                  "Local delta": "LocalDelta",
                  "Local gamma": "LocalGamma"}
    formattedTestingData = filteredTestingData.rename(columns=renameDict).set_index(["Strike", "Maturity"])[
        "ImpliedVol"]

    # Treatment for testing data
    filteredTrainingData = trainingData[(trainingData["Calibrated\nvol."] > 0) & (trainingData["Option\nprice"] > 0) & (
                trainingData["Option\ntype"] == 2)]

    roundedMat = filteredTrainingData["Maturity"].round(decimals=3)
    filteredTrainingData = filteredTrainingData.copy()
    filteredTrainingData.loc[:, "Maturity"] = roundedMat
    renameDict = {"Option\ntype": "OptionType",
                  "Option\nprice": "Price",
                  "Calibrated\nvol.": "ImpliedVol",  # "LocalImpliedVol",
                  "Implied\nvol.": "LocalImpliedVol"}  # "ImpliedVol"}
    formattedTrainingData = filteredTrainingData.drop(["Active", "Market vol. -\nCalibrated vol."], axis=1).rename(
        columns=renameDict).set_index(["Strike", "Maturity"])

    return dividendDf, rateCurveDf, localVolatility, formattedTestingData, formattedTrainingData


def selectTrainingSet(rawData):
    maturities = np.sort(np.unique(rawData.index.get_level_values("Maturity").values))
    maxTrainingMaturities = maturities[-3]
    filteredData = rawData[rawData.index.get_level_values("Maturity") <= maxTrainingMaturities]

    trainingPercentage = 0.2
    nbTrainingObs = int(rawData.shape[0] * trainingPercentage)
    sampledPercentage = nbTrainingObs / filteredData.shape[0]

    trainingSet = filteredData.sample(n=nbTrainingObs, axis=0)
    testingSet = rawData.drop(trainingSet.index)

    return trainingSet, testingSet

###################################################################### Main functions
def loadDataFromCSV(pathFolder, datFile):
    zeroCouponCurve = pd.read_csv(pathFolder + "discount.csv",decimal=".").apply(pd.to_numeric)
    dividendCurve = pd.read_csv(pathFolder + "dividend.csv",decimal=".").apply(pd.to_numeric)
    trainingData = pd.read_csv(pathFolder + "dataTrain.csv",decimal=".").apply(pd.to_numeric)
    testingData = pd.read_csv(pathFolder + "dataTest.csv",decimal=".").apply(pd.to_numeric)
    underlyingNative = pd.read_csv(pathFolder + "underlying.csv",decimal=".").apply(pd.to_numeric)
    localVolatilityNative = pd.read_csv(pathFolder + "locvol.csv",decimal=".").apply(pd.to_numeric)
    
    filteredDat = parseDatFile(pathFolder + datFile + ".dat")
    
    #return zeroCouponCurve, dividendCurve, trainingData, testingData, underlyingNative["S"].values[0], localVolatilityNative, filteredDat
    S0 = underlyingNative["S"].values[0]
    dividendDf, rateCurveDf, localVolatility, formattedTestingData, formattedTrainingData = cleanData(zeroCouponCurve,
                                                                                                      dividendCurve,
                                                                                                      trainingData,
                                                                                                      testingData,
                                                                                                      S0,
                                                                                                      localVolatilityNative)
    bootstrap = bootstrapping.bootstrapping(rateCurveDf, dividendDf, S0)
    #return dividendDf, rateCurveDf, localVolatility, formattedTestingData, formattedTrainingData, S0, bootstrap

    testingDataSet = dataSetConstruction.generateData(formattedTestingData,
                                                      S0,
                                                      bootstrap,
                                                      localVolatility)
    #trainingDataSet = dataSetConstruction.generateData(formattedTrainingData["ImpliedVol"],
    #                                                   S0,
    #                                                   bootstrap,
    #                                                   localVolatility,
    #                                                   priceDf=filteredDat)
    trainingDataSet = dataSetConstruction.generateData(formattedTrainingData["ImpliedVol"],
                                                       S0,
                                                       bootstrap,
                                                       localVolatility,
                                                       priceDf=formattedTrainingData.reset_index())

    return trainingDataSet, testingDataSet, bootstrap, S0

def loadDataFromDat(pathFolder, datFileName):
    localVolatilityNative = parseModelParamDatFile(pathFolder + datFileName + ".dat.modelparam.dat")
    testingData = parseImpliedVolDatFile(pathFolder + datFileName + ".dat.impliedvol.dat")
    trainingData = parseCalibrOutDatFile(pathFolder + datFileName + ".dat.calibr.out.dat")
    underlyingNative, zeroCouponCurve, dividendCurve, filteredDat = parseDatFiles(pathFolder + datFileName + ".dat")
    
    repricingError = rmse(trainingData["Option\nprice"], trainingData["MarketPrice"])
    print("Tikhonov PDE repricing error on training set : ", repricingError)
    
    #return zeroCouponCurve, dividendCurve, trainingData, testingData, underlyingNative["S"].values[0], localVolatilityNative, filteredDat
    S0 = underlyingNative["S"].values[0]
    dividendDf, rateCurveDf, localVolatility, formattedTestingData, formattedTrainingData = cleanData(zeroCouponCurve,
                                                                                                      dividendCurve,
                                                                                                      trainingData,
                                                                                                      testingData,
                                                                                                      S0,
                                                                                                      localVolatilityNative)
    bootstrap = bootstrapping.bootstrapping(rateCurveDf, dividendDf, S0)
    #return dividendDf, rateCurveDf, localVolatility, formattedTestingData, formattedTrainingData, S0, bootstrap

    testingDataSet = dataSetConstruction.generateData(formattedTestingData,
                                                      S0,
                                                      bootstrap,
                                                      localVolatility)
    #trainingDataSet = dataSetConstruction.generateData(formattedTrainingData["ImpliedVol"],
    #                                                   S0,
    #                                                   bootstrap,
    #                                                   localVolatility,
    #                                                   priceDf=filteredDat)
    trainingDataSet = dataSetConstruction.generateData(formattedTrainingData["ImpliedVol"],
                                                       S0,
                                                       bootstrap,
                                                       localVolatility,
                                                       priceDf=formattedTrainingData.reset_index())

    return trainingDataSet, testingDataSet, bootstrap, S0

def loadCBOTData(pathFolder, fileName, asOfDate):
    sheetName = "quotedata"

    aDf = pd.read_excel(pathFolder + fileName,
                        header=None,
                        sheet_name = sheetName)
    S0 = float(aDf.iloc[0, 1])

    aDf = pd.read_excel(pathFolder + fileName,
                        header=2,
                        sheet_name = sheetName)

    # aDf = aDf[aDf["Converted Expiration Date"] >= asOfDate]
    def formatDate(date):
        if isinstance(date, str):
            return pd.to_datetime(date, format='%m/%d/%Y')
        return pd.Timestamp(date.year, date.day, date.month)

    aDf["Converted Expiration Date"] = aDf["Expiration Date"].map(lambda x: formatDate(x))

    aDf = aDf[(aDf["Vol"] > 0.1) & (aDf["Last Sale"] > 0.01) & (aDf["Last Sale.1"] > 0.01) & (aDf["IV.1"] > 0.001) & (aDf["IV"] > 0.001)]
    closeDate = aDf["Converted Expiration Date"]
    strike = aDf["Strike"].round(decimals=3)
    maturity = (aDf["Converted Expiration Date"] - pd.Timestamp(asOfDate)).map(lambda x: x.days / 365.25).round(decimals=3)

    closeCall = (aDf["Bid"] + aDf["Ask"]) / 2  # aDf["Last Sale"]
    bidCall = aDf["Bid"]
    askCall = aDf["Ask"]
    impliedVolCall = aDf["IV"]
    deltaCall = aDf["Delta"]
    gammaCall = aDf["Gamma"]

    impliedVolPut = aDf["IV.1"]
    closePut = (aDf["Bid.1"] + aDf["Ask.1"]) / 2  # aDf["Last Sale.1"]
    bidPut = aDf["Bid.1"]
    askPut = aDf["Ask.1"]
    deltaPut = aDf["Delta.1"]
    gammaPut = aDf["Gamma.1"]

    callDf = pd.DataFrame(np.vstack(
        [strike, maturity, closeCall, bidCall, askCall, impliedVolCall, deltaCall, gammaCall,
         np.ones_like(deltaCall, dtype=np.int32)]).T,
                          columns=["Strike", "Maturity", "Price", "Bid", "Ask", "ImpliedVol", "Delta", "Gamma",
                                   "OptionType"]).set_index(["Strike", "Maturity"])
    PutDf = pd.DataFrame(np.vstack([strike, maturity, closePut, bidPut, askPut, impliedVolPut, deltaPut, gammaPut,
                                    2 * np.ones_like(deltaCall, dtype=np.int32)]).T,
                         columns=["Strike", "Maturity", "Price", "Bid", "Ask", "ImpliedVol", "Delta", "Gamma",
                                  "OptionType"]).set_index(["Strike", "Maturity"])

    bootstrap = bootstrapping.bootstrappingAveraged(pathFolder + "yieldCurve.dat",
                                                    S0,
                                                    strike,
                                                    asOfDate,
                                                    closeCall,
                                                    closePut,
                                                    maturity)

    rawData = pd.concat([callDf, PutDf])
    rawData = rawData[rawData["OptionType"]==2]

    #impvol = BS.vectorizedImpliedVolatilityCalibration(S0,
    #                                                   bootstrap,
    #                                                   rawData.index.get_level_values("Maturity"),
    #                                                   rawData.index.get_level_values("Strike"),
    #                                                   rawData["OptionType"],
    #                                                   rawData["Price"])

    #rawData["ImpliedVol"] =  impvol

    filteredData = removeDataViolatingStaticArbitrage(rawData.reset_index())
    rawData = filteredData.set_index(["Strike", "Maturity"])
    trainingSet, testingSet = selectTrainingSet(rawData)


    trainingDataSet = dataSetConstruction.generateData(trainingSet["ImpliedVol"],
                                                       S0,
                                                       bootstrap,
                                                       localVolatilityRef = None,
                                                       priceDf=trainingSet.reset_index(),
                                                       spotValue = False)
    testingDataSet = dataSetConstruction.generateData(testingSet["ImpliedVol"],
                                                      S0,
                                                      bootstrap,
                                                      localVolatilityRef = None,
                                                      priceDf=testingSet.reset_index(),
                                                      spotValue = False)


    return trainingDataSet, testingDataSet, bootstrap, S0

def loadESXData(pathFolder, fileName, asOfDate):

    #Stock
    S0 = pd.read_excel(pathFolder + fileName,
                       header=0,
                       sheet_name="Underlying_spot")["S0"].values[0]

    #Raw data
    rawData = pd.read_excel(pathFolder + fileName,
                            header=0,
                            sheet_name="Put_Call_ask_price_all")
    rawData["Strike"] = rawData["K"].astype(np.float).round(decimals=3)
    rawData["Maturity"] = rawData["T"].round(decimals=3)

    #Bootstrapping
    bootstrap = bootstrapping.bootstrappingAveragedExcel(pathFolder + fileName,
                                                         S0,
                                                         rawData["Strike"],
                                                         asOfDate,
                                                         rawData["Call_price"],
                                                         rawData["Put_price"],
                                                         rawData["Maturity"])


    callDf = pd.DataFrame(np.vstack([rawData["Call_price"].values, np.ones_like(rawData["Call_price"].values),
                                     rawData["Strike"].values, rawData["Maturity"].values]).T,
                          index = rawData.index,
                          columns = ["Price", "OptionType", "Strike", "Maturity"])

    putDf = pd.DataFrame(np.vstack([rawData["Put_price"].values, 2 * np.ones_like(rawData["Put_price"].values),
                                    rawData["Strike"].values, rawData["Maturity"].values]).T,
                         index=rawData.index,
                         columns=["Price", "OptionType", "Strike", "Maturity"])

    rawData = pd.concat([callDf, putDf])
    impvol = BS.vectorizedImpliedVolatilityCalibration(S0,
                                                       bootstrap,
                                                       rawData["Maturity"],
                                                       rawData["Strike"],
                                                       rawData["OptionType"],
                                                       rawData["Price"])

    rawData["ImpliedVol"] = impvol
    rawData = rawData.set_index(["Strike","Maturity"])

    trainingSet, testingSet = selectTrainingSet(rawData)



    trainingDataSet = dataSetConstruction.generateData(trainingSet["ImpliedVol"],
                                                       S0,
                                                       bootstrap,
                                                       localVolatilityRef = None,
                                                       priceDf=trainingSet.reset_index(),
                                                       spotValue = False)
    testingDataSet = dataSetConstruction.generateData(testingSet["ImpliedVol"],
                                                      S0,
                                                      bootstrap,
                                                      localVolatilityRef = None,
                                                      priceDf=testingSet.reset_index(),
                                                      spotValue = False)

    return trainingDataSet, testingDataSet, bootstrap, S0


def loadGPLocVol(pathFolder, GPKernel, bootstrap, S0):
    pathGP = pathFolder + ("local_vol_gaussian.csv" if GPKernel == "Gaussian" else "local_vol_matern_5_2.csv")
    print("Loading local volatility from : ", pathGP)

    locVolAresky = pd.read_csv(pathGP, decimal=".").apply(pd.to_numeric).dropna()

    #locVolAresky["Strike"] = locVolAresky["K"].values
    locVolAresky.insert(0, "Strike", locVolAresky["K"].values)

    #locVolAresky["Maturity"] = locVolAresky["T"].round(decimals=3)
    locVolAresky.insert(0, "Maturity", locVolAresky["T"].round(decimals=3))

    renameDict = {"loc_vol": "LocalVolatility"}
    locVolAreskyFormatted = locVolAresky.rename(columns=renameDict).set_index(["Strike", "Maturity"])

    changedVarAresky = bootstrap.changeOfVariable(locVolAreskyFormatted["K"],
                                                  locVolAreskyFormatted["T"])

    #locVolAreskyFormatted["Maturity"] = locVolAreskyFormatted["T"]
    locVolAreskyFormatted.insert(0, "Maturity", locVolAreskyFormatted["T"].round(decimals=3))

    #locVolAreskyFormatted["Strike"] = locVolAreskyFormatted["K"]
    locVolAreskyFormatted.insert(0, "Strike", locVolAreskyFormatted["K"].round(decimals=3))

    #locVolAreskyFormatted["ChangedStrike"] = pd.Series(changedVarAresky[0],
    #                                                   index=locVolAreskyFormatted.index)
    locVolAreskyFormatted.insert(0,
                                 "ChangedStrike",
                                 pd.Series(changedVarAresky[0], index=locVolAreskyFormatted.index))

    #locVolAreskyFormatted["logMoneyness"] = np.log(locVolAreskyFormatted["ChangedStrike"] / S0)
    locVolAreskyFormatted.insert(0, "logMoneyness",
                                 np.log(locVolAreskyFormatted["ChangedStrike"] / S0))
    locVolAreskyFormatted.insert(0, "OptionType",
                                 -np.ones_like(locVolAreskyFormatted["ChangedStrike"]))

    return locVolAreskyFormatted[~locVolAreskyFormatted.index.duplicated(keep='first')]


def removeDataViolatingStaticArbitrageStep(df):
    arbitrableRows = []
    for strike in df.rename({"Strike": "StrikeColumn"}, axis=1).groupby("StrikeColumn"):
        impliedTotVariance = np.square(strike[1]["ImpliedVol"]) * strike[1]["Maturity"]
        sortedStrike = pd.Series(impliedTotVariance.values,
                                 index=strike[1]["Maturity"].values).sort_index().diff().dropna()
        thetaViolation = sortedStrike.diff().dropna()
        if (thetaViolation < 0).sum() > 0:
            arbitrableMaturities = thetaViolation[(thetaViolation < 0)].index
            arbitrableRows.append(strike[1][ strike[1]["Maturity"].isin(arbitrableMaturities) ])
            #plt.plot(strike[1]["Maturity"].values, impliedTotVariance.values, label=str(strike[0]))
            #print("Strike : ", strike[0], " NbViolations : ", (thetaViolation < 0).sum())
            #print((thetaViolation < 0))
            #print(sortedStrike)
    return df.drop(pd.concat(arbitrableRows).index) if (len(arbitrableRows) > 0) else df

def removeDataViolatingStaticArbitrage(df):
    formerDf = df
    condition = True
    maxNbLoop = 500
    iterNb = 0
    while condition and (iterNb < maxNbLoop) :
        dfStep = removeDataViolatingStaticArbitrageStep(formerDf)
        condition = (dfStep.shape[0] < formerDf.shape[0]) and (dfStep.shape[0] > 0)
        formerDf = dfStep
        iterNb = iterNb + 1
    return dfStep
