import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy

import BS

impliedVolColumn = BS.impliedVolColumn

######################################################################################################## Discounting
#Compute the integral and return the linear interpolation function 
def interpIntegral(curve):
    #curve is piece-wise constant
    timeDelta = curve.index.to_series().diff().fillna(0)
    integralStepWise = (curve * timeDelta).cumsum()
    integralStepWise.loc[0] = 0.0
    integralStepWise.sort_index(inplace=True)
    integralSpline = interpolate.interp1d(integralStepWise.index,
                                          integralStepWise, 
                                          fill_value= 'extrapolate', 
                                          kind ='linear')

    timeStep = np.linspace(0, curve.index.max(),100)
    return pd.Series(integralSpline(timeStep),index=timeStep), integralSpline

def bootstrapZeroCoupon(curvePrice, name):
    #Bootstrap short rate curve
    def computeShortRate(curve) :
      shortRateList = [] 
      for i in range(curve.size):
        if i == 0 :
          shortRateList.append(-(np.log(curve.iloc[i]))/(curve.index[i]))
        else : 
          shortRateList.append(-(np.log(curve.iloc[i])-np.log(curve.iloc[i-1]))/(curve.index[i]-curve.index[i-1]))
      return pd.Series(shortRateList,index = curve.index)
    #For t=0 we take the first available point to ensure right continuity
    riskFreeCurve = computeShortRate(curvePrice)
    riskFreeCurve.loc[0.00] = riskFreeCurve.iloc[0]
    riskFreeCurve = riskFreeCurve.sort_index()

    #Bootstrap yield curve
    def zeroYield(x):
      if(float(x.name) < 1):
        return (1/x - 1)/float(x.name)
      else:
        return (x**(-1/float(x.name)) - 1)
    yieldCurve = curvePrice.apply(zeroYield, axis = 1)
    yieldCurve.loc[0.00] = yieldCurve.iloc[0]
    yieldCurve = yieldCurve.sort_index()

    plt.plot(riskFreeCurve, label = "Short rate")

    #Interpolate short rate curve and yield curve
    timeStep = np.linspace(0,0.99,100)
    riskCurvespline = interpolate.interp1d(riskFreeCurve.index,
                                           riskFreeCurve,#riskFreeCurve[name],
                                           fill_value= 'extrapolate',
                                           kind ='next')
    interpolatedCurve = pd.Series(riskCurvespline(timeStep),index=timeStep)
    plt.plot(interpolatedCurve, label="Interpolated short rate")
    plt.legend()
    plt.show()

    plt.plot(yieldCurve, label = "Yield curve")
    yieldCurvespline = interpolate.interp1d(yieldCurve.index,
                                            yieldCurve['Price'],
                                            fill_value= 'extrapolate',
                                            kind ='next')
    interpolatedCurve = pd.Series(yieldCurvespline(timeStep),index=timeStep)
    plt.plot(interpolatedCurve, label = "Interpolated Yield curve")
    plt.legend()
    plt.show()
    
    #Integrate short rate
    interpolatedIntegral, riskFreeIntegral = interpIntegral(riskFreeCurve)
    plt.plot(interpolatedIntegral)
    plt.show()

    return riskFreeCurve, riskCurvespline, yieldCurve, yieldCurvespline, interpolatedIntegral, riskFreeIntegral


############################################################################# Dividend
def bootstrapDividend(curvePrice, underlying, name):
    #Compute cumulative sum of dividend plus spot price
    priceEvolution = underlying - curvePrice['Amount'].cumsum()
    priceEvolution.loc[0] = underlying
    priceEvolution.sort_index(inplace=True)

    #Bootstrap short rate for dividend
    def computeShortRate(curve) :
      shortRateList = [] 
      for i in range(curve.size):
        if i == 0 :
          shortRateList.append(-(np.log(curve.iloc[i+1])-np.log(curve.iloc[i]))/(curve.index[i+1]-curve.index[i]))
        else : 
          shortRateList.append(-(np.log(curve.iloc[i])-np.log(curve.iloc[i-1]))/(curve.index[i]-curve.index[i-1]))
      return pd.Series(shortRateList,index = curve.index).dropna()
    logReturnDividendDf = computeShortRate(priceEvolution)

    #Dividend yield curve
    def divYield(x):
      return ((priceEvolution[x]/priceEvolution.iloc[0])**(1/float(x)) - 1) #np.log(priceEvolution[x]/priceEvolution.iloc[0])/x
    dividendYield = logReturnDividendDf.index.to_series().tail(-1).apply(divYield)
    dividendYield.loc[0.00] = dividendYield.iloc[0]
    dividendYield = dividendYield.sort_index()

    plt.plot(logReturnDividendDf, label = "Short rate")

    #Interpolate short rate curve and yield curve
    timeStep = np.linspace(0,0.99,100)
    logReturnDividendSpline = interpolate.interp1d(logReturnDividendDf.index,
                                                   logReturnDividendDf,#logReturnDividendDf[name],
                                                   fill_value= 'extrapolate',
                                                   kind ='next')
    interpolatedCurve = pd.Series(logReturnDividendSpline(timeStep),index=timeStep)
    plt.plot(interpolatedCurve, label="Interpolated short rate")
    plt.legend()
    plt.show()

    plt.plot(dividendYield, label = "Yield curve")
    yieldCurvespline = interpolate.interp1d(dividendYield.index,
                                            dividendYield.values,
                                            fill_value= 'extrapolate',
                                            kind ='next')
    interpolatedCurve = pd.Series(yieldCurvespline(timeStep),index=timeStep)
    plt.plot(interpolatedCurve, label = "Interpolated Yield curve")
    plt.legend()
    plt.show()
    
    #Integrate short rate
    interpolatedIntegral, logReturnDividendIntegral = interpIntegral(logReturnDividendDf)#logReturnDividendDf[name])
    plt.plot(interpolatedIntegral)
    plt.show()

    return logReturnDividendDf, logReturnDividendSpline, dividendYield, yieldCurvespline, interpolatedIntegral, logReturnDividendIntegral
    
######################################################### Change of measure
#Change of variable for deterministic discount curve and dividend curve
def changeOfVariable(s, t, divSpreadIntegral, riskFreeIntegral):
  def qInterp(m):
    return divSpreadIntegral(m).astype(np.float32)
  q = qInterp(t)
  
  def rInterp(m):
    return riskFreeIntegral(m).astype(np.float32)
  r = rInterp(t)

  factorPrice = np.exp( - q )

  divSpread = q-r

  factorStrike = np.exp( divSpread )
  adjustedStrike = np.multiply(s, factorStrike)
  return adjustedStrike, factorPrice


class bootstrapping:
    def __init__(self,
                 rateCurveDf,
                 dividendDf,
                 underlying):
        #Bootstrap data
        print("Bootstrap discounting")
        riskFreeCurve, riskCurvespline, yieldCurve, yieldCurvespline, interpolatedIntegral, riskFreeIntegral = bootstrapZeroCoupon(rateCurveDf, "Short rate")

        print("Bootstrap dividend")
        spreadDividend, divSpline, yieldDividend, divYieldSpline, interpolatedDivIntegral, divSpreadIntegral = bootstrapDividend(dividendDf, underlying, "Spread")

        #Define member
        self.riskCurvespline = riskCurvespline
        self.yieldCurvespline = yieldCurvespline
        self.riskFreeIntegral = riskFreeIntegral

        self.riskFreeCurve = riskFreeCurve
        self.yieldCurve = yieldCurve
        self.interpolatedIntegral = interpolatedIntegral

        self.divSpline = divSpline
        self.divYieldSpline = divYieldSpline
        self.divSpreadIntegral = divSpreadIntegral

        self.spreadDividend = spreadDividend
        self.yieldDividend = yieldDividend
        self.interpolatedDivIntegral = interpolatedDivIntegral

    def discountShortRate(self, t):
        return self.riskCurvespline(t)

    def dividendShortRate(self, t):
        return self.divSpline(t)

    def discountIntegral(self, t):
        return self.riskFreeIntegral(t)

    def dividendIntegral(self, t):
        return self.divSpreadIntegral(t)

    def changeOfVariable(self, s, t):
        return changeOfVariable(s, t, self.divSpreadIntegral, self.riskFreeIntegral)



class bootstrappingDummy(bootstrapping):
    def __init__(self,
                 rateCurveDf,
                 dividendDf,
                 underlying):


        discountR = 0#rateCurveDf["r"].dropna().iloc[0]
        riskCurvespline = lambda t : discountR
        riskFreeIntegral = lambda t : discountR * t
        def zeroYield(x, t):
          if(float(t) < 1):
            return (1/x - 1)/float(t)
          else:
            return (x**(-1/float(t)) - 1)
        yieldCurvespline =  lambda t : zeroYield(np.exp(-discountR * t), t)

        divq = 0
        divSpline = lambda t : divq
        divYieldSpline = lambda t : zeroYield(np.exp(-divq * t), t)
        divSpreadIntegral =  lambda t : divq * t

        #Define member
        self.riskCurvespline = riskCurvespline
        self.yieldCurvespline = yieldCurvespline
        self.riskFreeIntegral = riskFreeIntegral

        self.riskFreeCurve = None
        self.yieldCurve = None
        self.interpolatedIntegral = None

        self.divSpline = divSpline
        self.divYieldSpline = divYieldSpline
        self.divSpreadIntegral = divSpreadIntegral

        self.spreadDividend = None
        self.yieldDividend = None
        self.interpolatedDivIntegral = None


def getTermFromString(strTerm):
    elts = strTerm.split(" ")
    number = int(elts[0])

    parseDict = {"Mo": 1 / 12,
                 "months": 1 / 12,
                 "Yr": 1,
                 "year": 1,
                 "years": 1,
                 "Wk": 7 / 365.25,
                 "d": 1 / 365.25}
    unit = parseDict[elts[1]]

    return float(number * unit)


def zeroCoupon(tp):
    x = tp[0]
    t = tp[1]
    if (float(t) < 1):
        return 1 / (1 + t * x)
    else:
        return 1 / (1 + x) ** t

def zeroYield(tp):
    x = tp[0]
    t = tp[1]
    if (float(t) < 1):
        return (1 / x - 1) / float(t)
    else:
        return (x ** (-1 / float(t)) - 1)



class NelsonSiegelSvenssonCalibrator:
    def __init__(self):
        self.beta = []
        self.tau = []
        self.verbose = False

    def isCalibrated(self):
        return (len(self.beta) > 0) and (len(self.tau) > 0)

    def objectiveFunction(self, ttms, beta, tau):
        term0 = beta[0]
        term1 = beta[1] * (1 - np.exp(- ttms / tau[0])) / (ttms / tau[0])
        term2 = beta[2] * ((1 - np.exp(- ttms / tau[0])) / (ttms / tau[0]) - np.exp(- ttms / tau[0]))
        term3 = beta[3] * ((1 - np.exp(- ttms / tau[1])) / (ttms / tau[1]) - np.exp(- ttms / tau[1]))

        return term0 + term1 + term2 + term3

    def eval(self, ttms):
        if not self.isCalibrated() :
            raise Exception("Please calibrate the model")
        return self.objectiveFunction(ttms, self.beta, self.tau)
    
    def forward(self, ttms):
        if not self.isCalibrated() :
            raise Exception("Please calibrate the model")
        term0 = self.beta[0]
        term1 = self.beta[1] * np.exp(- ttms / self.tau[0])
        term2 = self.beta[2] * np.exp(- ttms / self.tau[0]) * (ttms / self.tau[0]) 
        term3 = self.beta[3] * np.exp(- ttms / self.tau[1]) * (ttms / self.tau[1])
        
        return term0 + term1 + term2 + term3
    
    def calibrate(self, ttms, rates):
        def rmse(a, b):
            return np.mean(np.square(a - b))
        loss = lambda x : rmse(self.objectiveFunction(ttms, x[:4], x[4:]), rates)

        x0 = np.array([1.277233, -1.865915, 11.597678, -14.833966, 1.654792, 1.789270])
        bounds = [(-100,100), (-100,100), (-100,100), (-100,100), (0,30), (0,30)]
        res = scipy.optimize.minimize(loss, x0, bounds = bounds, method = "L-BFGS-B")
        self.beta = res.x[:4]
        self.tau = res.x[4:]

        if self.verbose :
            print("Parameters ; beta : ", self.beta, " ; tau : ", self.tau)
            print("Error : ", res.fun)

class TikhnovCalibrator:
    def __init__(self, alpha):
        self.trainGrid = None
        self.trainRate = None
        self.alpha = alpha

    def isCalibrated(self):
        return (self.trainGrid is not None) and (self.trainRate is not None)

    def objectiveFunction(self, ttms, alpha):
        timeGrid = np.union1d(ttms, self.trainGrid)
        redundantPoints = np.isin(timeGrid, self.trainGrid, assume_unique=True)
        SSstar = np.diag(redundantPoints.astype(int))
        M = SSstar.shape[1]
        N = self.trainGrid.size
        
        S = np.zeros((N, M), dtype = np.int32)
        j = 0
        for i in range(M):
            if redundantPoints[i] : 
                S[j,i] = 1
                j += 1
        L = np.zeros((M-2, M))
        
        for i in range(M-2):
            L[i,i] = (timeGrid[i+1])/2/(timeGrid[i+1] - timeGrid[i])/(timeGrid[i+2] - timeGrid[i]) * np.sqrt((timeGrid[i] + timeGrid[i+2])/2)
            L[i,i+1] = -(timeGrid[i+1])/2/(timeGrid[i+1] - timeGrid[i])/(timeGrid[i+2] - timeGrid[i+1]) * np.sqrt((timeGrid[i] + timeGrid[i+2])/2)
            L[i,i+2] = -(timeGrid[i+1])/2/(timeGrid[i+2] - timeGrid[i+1])/(timeGrid[i+2] - timeGrid[i]) * np.sqrt((timeGrid[i] + timeGrid[i+2])/2)
        return np.linalg.solve((SSstar + alpha * L.T @ L), S.T @ self.trainRate)[np.isin(timeGrid, ttms, assume_unique=True)]

    def eval(self, ttms):
        if not self.isCalibrated() :
            raise Exception("Please calibrate the model")
        return self.objectiveFunction(ttms, self.alpha)

    def calibrate(self, ttms, rates):
        series = pd.Series(rates, index = ttms).sort_index()
        self.trainGrid = series.index.values
        self.trainRate = series.values

def interpLinear(formerGrid, formerValue, newGrid):
    f = interpolate.interp1d(formerGrid,
                             formerValue,  # riskFreeCurve[name],
                             fill_value='extrapolate',
                             kind='linear')
    return f(newGrid)

class bootstrappingLinReg(bootstrapping):
    def __init__(self,
                 path,
                 S0,
                 Strike,
                 asOfDate,
                 callPrice,
                 putPrice,
                 maturity):
        ############################################################################### Load data yield curve
        usedDiscount = self.loadDiscountCurve(path, asOfDate)
        shortDiscount = usedDiscount[usedDiscount.index <= 0.1]
        plt.plot(usedDiscount.index.values,
                 usedDiscount.values, "+")
        extendedGrid = np.linspace(0.00000001,
                                   max(usedDiscount.index.values),
                                   1000)
        self.nelsonModel = self.nelsonSiegelInterp(usedDiscount)
        usedDiscount = pd.Series(self.nelsonModel.eval(extendedGrid), index = extendedGrid)
        #usedDiscount = pd.Series(interpLinear( usedDiscount.index.values, usedDiscount.values, extendedGrid), index=extendedGrid)
        plt.plot(usedDiscount.index.values,
                 usedDiscount.values)
        plt.title("Interpolated discount yield curve")
        plt.show()
        
        
        plt.plot(shortDiscount.index.values,
                 shortDiscount.values, "+")
        #usedDiscount = pd.Series(interpLinear( usedDiscount.index.values, usedDiscount.values, extendedGrid), index=extendedGrid)
        plt.plot(usedDiscount[usedDiscount.index <= 0.3].index.values,
                 usedDiscount[usedDiscount.index <= 0.3].values)
        plt.title("Interpolated discount yield curve")
        plt.show()

        ################################################################################# Get zero coupon prices from yield curve
        zeroCouponUsed = list(map(zeroCoupon, list(zip(usedDiscount.values, usedDiscount.index))))
        zeroCouponUsed.insert(0, 1)
        zeroCouponCurve = pd.Series(zeroCouponUsed, index=usedDiscount.index.insert(0, 0))


        zeroCouponInterp = interpolate.interp1d(zeroCouponCurve.index,
                                                zeroCouponCurve,  # riskFreeCurve[name],
                                                fill_value='extrapolate',
                                                kind='next')
        plt.plot(zeroCouponCurve)
        plt.title("Discount zero coupon curve")
        plt.show()

        ################################################################################# Get short rate from zero coupon
        zeroCouponVar = zeroCouponCurve.div(zeroCouponCurve.shift()).dropna()
        timeDelta = zeroCouponCurve.index.to_series().diff().dropna()
        #shortRate = -np.log(zeroCouponVar) / timeDelta  # Short rate piecewise constant
        
        #shortRateInterp = interpolate.interp1d(shortRate.index,
        #                                       shortRate,  # riskFreeCurve[name],
        #                                       fill_value='extrapolate',
        #                                       kind='next')
        
        shortRate = pd.Series(self.nelsonModel.forward(zeroCouponVar.index.values), index = zeroCouponVar.index)
        shortRateInterp = lambda x : self.nelsonModel.forward(x)
        
        plt.plot(shortRate)
        plt.plot(pd.Series(self.nelsonModel.forward(zeroCouponVar.index.values), index = zeroCouponVar.index))
        plt.title("Discount short rate curve")
        plt.show()
        
        plt.plot(shortRate[shortRate.index <= 0.3])
        plt.title("Discount short rate curve")
        plt.show()

        ################################################################################# Interpolate yield curve
        yieldInterp = interpolate.interp1d(usedDiscount.index,
                                           usedDiscount,  # riskFreeCurve[name],
                                           fill_value='extrapolate',
                                           kind='next')
        plt.plot(usedDiscount)
        plt.title("Discount yield curve")
        plt.show()

        interpolatedIntegral, riskFreeIntegral = interpIntegral(shortRate)
        plt.plot(interpolatedIntegral)
        plt.title("Integrated Discount yield curve")
        plt.show()

        ################################################################################# Set members
        self.riskCurvespline = shortRateInterp
        self.yieldCurvespline = yieldInterp
        self.riskFreeIntegral = riskFreeIntegral

        self.riskFreeCurve = shortRate
        self.yieldCurve = usedDiscount
        self.interpolatedIntegral = interpolatedIntegral


        ######################################################################### Extract dividend zero coupon and dividend short rate from call put-parity
        divCurve, shortRateDiv = self.getDividendFromCallPutParity(callPrice, putPrice, Strike, maturity, S0)

        ######################################################################### Extract dividend yield from dividend zero coupon
        yieldDiv = pd.Series(list(map(zeroYield, list(zip(divCurve.values, divCurve.index)))),
                             index = divCurve.index)
        plt.plot(yieldDiv)
        plt.title("Dividend yield curve")
        plt.show()

        ######################################################################### Interpolate short rate and dividend yield
        divSpline = interpolate.interp1d(shortRateDiv.index,
                                         shortRateDiv,  # riskFreeCurve[name],
                                         fill_value='extrapolate',
                                         kind='next')


        divYieldSpline = interpolate.interp1d(yieldDiv.index,
                                              yieldDiv,  # riskFreeCurve[name],
                                              fill_value='extrapolate',
                                              kind='next')

        interpolatedDivIntegral, divSpreadIntegral = interpIntegral(shortRateDiv)
        plt.plot(interpolatedDivIntegral)
        plt.title("Integrated dividend short rate curve")
        plt.show()


        ################################################################################# Set members
        # Define member
        self.divSpline = divSpline
        self.divYieldSpline = divYieldSpline
        self.divSpreadIntegral = divSpreadIntegral

        self.spreadDividend = divCurve
        self.yieldDividend = yieldDiv
        self.interpolatedDivIntegral = interpolatedDivIntegral

    def nelsonSiegelInterp(self, curve):
        interpNelson = NelsonSiegelSvenssonCalibrator()
        interpNelson.calibrate(curve.index.values, curve.values)
        return interpNelson
    
    
    def tikhonovInterp(self, curve):
        interpTikhonov = TikhnovCalibrator(0.1)
        interpTikhonov.calibrate(curve.index.values, curve.values)
        return interpTikhonov

    def loadDiscountCurve(self, path, asOfDate):
        # source : https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year=2019
        rawDiscount = pd.read_csv(path, sep="\t", header=1)
        date = rawDiscount["Date"].map(lambda x: pd.Timestamp(x))  # convert date as timestamp
        indexedDiscount = rawDiscount.set_index(date).drop("Date", axis=1)

        renamedDiscount = indexedDiscount.rename(mapper=getTermFromString, axis=1)  # Get term from column name
        usedDiscount = renamedDiscount[renamedDiscount.index <= asOfDate].iloc[-1] * 0.01 # Get latest yield curve
        return usedDiscount

    def getZeroCouponPrice(self, t):
        return np.exp(- self.discountIntegral(t))

    def getDividendFromCallPutParity(self, callPrice, putPrice, 
                                     Strike, maturity, S0):
        forwardCallPut = callPrice - putPrice
        #plt.plot(maturity, zeroCouponInterp(maturity))
        #plt.show()
        
        zeroCouponDiv = ((forwardCallPut + Strike * self.getZeroCouponPrice(maturity)) / S0)
        zeroCouponDiv.loc[zeroCouponDiv.size] = 1.0
        zeroCouponDiv.sort_index(inplace = True)
        
        matCopy = maturity.copy()
        matCopy.loc[matCopy.size] = 0.0
        matCopy.sort_index(inplace = True)
        
        reg = linear_model.LinearRegression()

        def featureMap(mat, order):
            return np.vstack([np.power(mat, i) for i in range(order)]).T

        features = featureMap(matCopy, 4)

        reg.fit(features, zeroCouponDiv.values)

        pred = reg.predict(featureMap(np.unique(matCopy), 4))
        divCurve = pd.Series(pred, index=np.unique(matCopy)).sort_index()
        plt.plot(divCurve)
        plt.plot(matCopy, zeroCouponDiv.values, "+")
        plt.title("Dividend zero coupon curve")
        plt.show()

        divVar = divCurve.div(divCurve.shift()).dropna()
        timeDivDelta = divCurve.index.to_series().diff().dropna()
        shortRateDiv = -np.log(divVar) / timeDivDelta
        plt.plot(shortRateDiv)
        plt.title("Dividend short rate curve")
        plt.show()
        
        self.forwardCallPut = forwardCallPut
        self.zeroCouponDiv = pd.Series(zeroCouponDiv.values, index = matCopy.values) #index = pd.MultiIndex.from_arrays([Strike.values, maturity.values]))
        
        return divCurve, shortRateDiv
        


class bootstrappingLinRegShortRate(bootstrappingLinReg):
    def __init__(self,
                 path,
                 S0,
                 Strike,
                 asOfDate,
                 callPrice,
                 putPrice,
                 maturity):
        super().__init__(path,
                         S0,
                         Strike,
                         asOfDate,
                         callPrice,
                         putPrice,
                         maturity)
                         
    def getDividendFromCallPutParity(self, callPrice, putPrice, 
                                     Strike, maturity, S0):
        forwardCallPut = callPrice - putPrice
        #Tankov version
        
        dividendInt = (np.log(forwardCallPut + Strike * self.getZeroCouponPrice(maturity))).sort_index()
        
        matCopy = maturity.copy().sort_index()
        
        reg = linear_model.LinearRegression()

        def featureMap(mat, order):
            return np.vstack([np.power(mat, i) for i in range(order)]).T

        features = featureMap(matCopy, 3)

        reg.fit(features, dividendInt.values)

        pred = reg.predict(featureMap(np.unique(matCopy), 3))
        divCurve = pd.Series(-(pred - np.log(S0)), index=np.unique(matCopy)).sort_index()
        divCurve.loc[0.0] = 0.0
        divCurve.sort_index(inplace = True)
        
        plt.plot(divCurve)
        plt.plot(matCopy, -(dividendInt.values  - np.log(S0)), "+")
        plt.title("Integrated Dividend short rate curve")
        plt.show()
        
        
        divVar = divCurve.diff().dropna()
        timeDivDelta = divCurve.index.to_series().diff().dropna()
        shortRateDiv = divVar / timeDivDelta
        plt.plot(shortRateDiv)
        #print(divCurve)
        #print(shortRateDiv)
        plt.title("Dividend short rate curve")
        plt.show()
        
        zeroCouponDiv = np.exp(-divCurve)
        plt.plot(zeroCouponDiv)
        plt.title("Dividend zero coupon curve")
        plt.show()
        
        self.forwardCallPut = forwardCallPut
        self.zeroCouponDiv = pd.Series(zeroCouponDiv.values, index = matCopy.values) #index = pd.MultiIndex.from_arrays([Strike.values, maturity.values]))
        
        return zeroCouponDiv, shortRateDiv

class bootstrappingImplied(bootstrappingLinReg):
    def __init__(self,
                 path,
                 S0,
                 Strike,
                 asOfDate,
                 callPrice,
                 putPrice,
                 maturity,
                 callImpVol,
                 putImpVol):
        self.callImpVol = callImpVol
        self.putImpVol = putImpVol
        super().__init__(path,
                         S0,
                         Strike,
                         asOfDate,
                         callPrice,
                         putPrice,
                         maturity)
                         
    def getDividendFromCallPutParity(self, callPrice, putPrice, 
                                     Strike, maturity, S0):
        forwardCallPut = callPrice - putPrice
        #Tankov version
        
        dividendInt = (np.log(forwardCallPut + Strike * self.getZeroCouponPrice(maturity))).sort_index()
        
        matCopy = maturity.copy().sort_index()
        
        callSeries = pd.Series(np.ones_like(callPrice), index = callPrice.index)
        putSeries = pd.Series(-np.ones_like(putPrice), index = putPrice.index)
        impDividendTrain = BS.vectorizedImpliedDividendCalibration(S0,
                                                                   self,
                                                                   matCopy.append(matCopy.copy()),
                                                                   Strike.append(Strike.copy()),
                                                                   callSeries.append(putSeries),
                                                                   callPrice.append(putPrice),
                                                                   self.callImpVol.append(self.putImpVol))
        
        
        impDividendTrain = pd.Series(impDividendTrain, index = matCopy.append(matCopy.copy()).values ).sort_index()
        divCurve = impDividendTrain.sort_index().groupby(by=impDividendTrain.index).mean()
        divCurve.loc[0.0] = 0.0
        divCurve.sort_index(inplace = True)
        
        plt.plot(divCurve)
        #plt.plot(matCopy, -(dividendInt.values  - np.log(S0)), "+")
        plt.plot(impDividendTrain, "+")
        plt.title("Integrated Dividend short rate curve")
        plt.show()
        
        shortTermdivCurve = divCurve[divCurve.index <= 0.3]
        plt.plot(shortTermdivCurve)
        shortMaturity = matCopy[matCopy <= shortTermdivCurve.index[-1]]
        #plt.plot(shortMaturity, -(dividendInt.head(shortMaturity.size).values  - np.log(S0)), "+")
        plt.plot(impDividendTrain[impDividendTrain.index <= 0.3], "+")
        plt.title("Integrated Dividend short rate curve")
        plt.show()
        
        
        divVar = divCurve.diff().dropna()
        timeDivDelta = divCurve.index.to_series().diff().dropna()
        shortRateDiv = divVar / timeDivDelta
        plt.plot(shortRateDiv)
        #print(divCurve)
        #print(shortRateDiv)
        plt.title("Dividend short rate curve")
        plt.show()
        
        zeroCouponDiv = np.exp(-divCurve)
        plt.plot(zeroCouponDiv)
        plt.title("Dividend zero coupon curve")
        plt.show()
        
        self.forwardCallPut = forwardCallPut
        self.zeroCouponDiv = zeroCouponDiv #index = pd.MultiIndex.from_arrays([Strike.values, maturity.values]))
        
        return zeroCouponDiv, shortRateDiv

class bootstrappingAveraged(bootstrappingLinReg):
    def __init__(self,
                 path,
                 S0,
                 Strike,
                 asOfDate,
                 callPrice,
                 putPrice,
                 maturity):
        super().__init__(path,
                         S0,
                         Strike,
                         asOfDate,
                         callPrice,
                         putPrice,
                         maturity)


    def getDividendFromCallPutParity(self, callPrice, putPrice, 
                                     Strike, maturity, S0):
        forwardCallPut = callPrice - putPrice
        #plt.plot(maturity, zeroCouponInterp(maturity))
        #plt.show()
        
        zeroCouponDiv = ((forwardCallPut + Strike * self.getZeroCouponPrice(maturity)) / S0)
        zeroCouponDiv.loc[zeroCouponDiv.size] = 1.0
        zeroCouponDiv.sort_index(inplace = True)
        
        matCopy = maturity.copy()
        matCopy.loc[matCopy.size] = 0.0
        matCopy.sort_index(inplace = True)
        
        print(matCopy.shape)
        print(zeroCouponDiv.shape)
        
        averagesDiv = pd.DataFrame(np.vstack([zeroCouponDiv, matCopy]).T,
                                   index=zeroCouponDiv.index,
                                   columns=["zeroCouponDiv", "Maturity"]).groupby("Maturity").mean()
        averagesDivInterp = interpolate.interp1d(averagesDiv.index,
                                                 np.ravel(averagesDiv.values),  # riskFreeCurve[name],
                                                 fill_value='extrapolate',
                                                 kind='next')
        divCurve = pd.Series(averagesDivInterp(np.unique(matCopy)), index=np.unique(matCopy)).sort_index()
        plt.plot(divCurve)
        plt.plot(matCopy, zeroCouponDiv.values, "+")
        plt.title("Dividend zero coupon curve")
        plt.show()

        divVar = divCurve.div(divCurve.shift()).dropna()
        timeDivDelta = divCurve.index.to_series().diff().dropna()
        shortRateDiv = -np.log(divVar) / timeDivDelta
        plt.plot(shortRateDiv)
        plt.title("Dividend short rate curve")
        plt.show()
        
        self.forwardCallPut = forwardCallPut
        self.zeroCouponDiv = pd.Series(zeroCouponDiv.values, index = matCopy.values) #index = pd.MultiIndex.from_arrays([Strike.values, maturity.values]))
        
        return divCurve, shortRateDiv




class bootstrappingLinRegExcel(bootstrappingLinReg):
    def __init__(self,
                 path,
                 S0,
                 Strike,
                 asOfDate,
                 callPrice,
                 putPrice,
                 maturity):
        super().__init__(path,
                         S0,
                         Strike,
                         asOfDate,
                         callPrice,
                         putPrice,
                         maturity)

    def loadDiscountCurve(self, path, asOfDate):
        # Taken from ECB website
        discountDf = pd.read_excel(path,#workingFolder + "Data_EuroStoxx50_20190110_all_for_Marc.xlsx",
                                   header=0,
                                   sheet_name="DiscountGermany")
        date = discountDf["Maturity"].map(getTermFromString)
        usedDiscount = discountDf.set_index(date)["Par Yield"] * 0.01
        return usedDiscount



class bootstrappingAveragedExcel(bootstrappingAveraged):
    def __init__(self,
                 path,
                 S0,
                 Strike,
                 asOfDate,
                 callPrice,
                 putPrice,
                 maturity):
        super().__init__(path,
                         S0,
                         Strike,
                         asOfDate,
                         callPrice,
                         putPrice,
                         maturity)

    def loadDiscountCurve(self, path, asOfDate):
        # Taken from ECB website
        discountDf = pd.read_excel(path,#workingFolder + "Data_EuroStoxx50_20190110_all_for_Marc.xlsx",
                                   header=0,
                                   sheet_name="DiscountGermany")
        date = discountDf["Maturity"].map(getTermFromString)
        usedDiscount = discountDf.set_index(date)["Par Yield"] * 0.01
        return usedDiscount




class bootstrappingFromData(bootstrapping):
    def __init__(self,
                 dfCurve):

        def zeroYield(x, t):
            zeroCouponPrice = np.exp(-x)
            if t == 0:
                return 0
            if (float(t) < 1):
                return (1 / zeroCouponPrice - 1) / float(t)
            else:
                return (zeroCouponPrice ** (-1 / float(t)) - 1)

        self.riskFreeCurve = dfCurve["riskCurvespline"]
        pairedData = zip(dfCurve["riskFreeIntegral"].values,
                         dfCurve["riskFreeIntegral"].index)
        yieldCurve = pd.Series(list(map(lambda x: zeroYield(x[0], x[1]), pairedData)),
                               dfCurve["riskFreeIntegral"].index)
        self.yieldCurve = yieldCurve
        self.interpolatedIntegral = dfCurve["riskFreeIntegral"]

        self.spreadDividend = dfCurve["divSpline"]
        pairedData = zip(dfCurve["divSpreadIntegral"].values,
                         dfCurve["divSpreadIntegral"].index)
        yieldDividend = pd.Series(list(map(lambda x: zeroYield(x[0], x[1]), pairedData)),
                                  dfCurve["divSpreadIntegral"].index)
        self.yieldDividend = yieldDividend
        self.interpolatedDivIntegral = dfCurve["divSpreadIntegral"]


        self.riskCurvespline = interpolate.interp1d(self.riskFreeCurve.index,
                                                    self.riskFreeCurve,  # riskFreeCurve[name],
                                                    fill_value='extrapolate',
                                                    kind='next')
        self.yieldCurvespline = interpolate.interp1d(self.yieldCurve.index,
                                                     self.yieldCurve,  # riskFreeCurve[name],
                                                     fill_value='extrapolate',
                                                     kind='next')
        self.riskFreeIntegral = interpolate.interp1d(self.interpolatedIntegral.index,
                                                     self.interpolatedIntegral,  # riskFreeCurve[name],
                                                     fill_value='extrapolate',
                                                     kind='next')

        self.divSpline = interpolate.interp1d(self.spreadDividend.index,
                                              self.spreadDividend,  # riskFreeCurve[name],
                                              fill_value='extrapolate',
                                              kind='next')
        self.divYieldSpline = interpolate.interp1d(self.yieldDividend.index,
                                                   self.yieldDividend,  # riskFreeCurve[name],
                                                   fill_value='extrapolate',
                                                   kind='next')
        self.divSpreadIntegral = interpolate.interp1d(self.interpolatedDivIntegral.index,
                                                      self.interpolatedDivIntegral,  # riskFreeCurve[name],
                                                      fill_value='extrapolate',
                                                      kind='next')
