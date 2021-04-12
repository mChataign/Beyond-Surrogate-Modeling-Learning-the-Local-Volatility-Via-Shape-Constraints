#Import tensorflow for 1.x version
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Dense, Input
from tensorflow.compat.v1.keras import Model
import tensorflow.compat.v1.keras.backend as K
import tensorflow.compat.v1.keras.activations as Act
from functools import partial
import time
import dataSetConstruction
import bootstrapping
import numpy as np
import pandas as pd
import scipy
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

impliedVolColumn = bootstrapping.impliedVolColumn

activateScaling = False
transformCustom = dataSetConstruction.transformCustomMinMax if activateScaling else dataSetConstruction.transformCustomId
inverseTransform = dataSetConstruction.inverseTransformMinMax if activateScaling else dataSetConstruction.inverseTransformId
inverseTransformColumn = dataSetConstruction.inverseTransformColumnMinMax if activateScaling else dataSetConstruction.inverseTransformColumnId
inverseTransformColumnGreeks = dataSetConstruction.inverseTransformColumnGreeksMinMax if activateScaling else dataSetConstruction.inverseTransformColumnGreeksId

layerFactory = {}
modelFolder = "./Results/" 

############################################################################################################# Tooling functions
def resetTensorflow():
    tf.reset_default_graph()
    layerFactory.clear()
    return

# Format result from training step
def evalAndFormatResult(price, loss, dataSet, scaler):
    scaledPredPrice = pd.Series(price.flatten(), index=dataSet.index).rename("Price")
    predPrice = inverseTransformColumn(dataSet, scaler, scaledPredPrice)
    lossEpochSerie = pd.Series(loss)
    return predPrice, lossEpochSerie


# Format result from training step when local volatility is computed
def evalAndFormatDupireResult(price, volDupire, theta, gamma,
                              dupireVar, loss, dataSet, scaler):
    predPrice, lossEpoch = evalAndFormatResult(price, loss, dataSet, scaler)

    predDupire = pd.Series(volDupire.flatten(), index=dataSet.index).rename("Dupire")

    scaledTheta = pd.Series(theta.flatten(), index=dataSet.index).rename("Theta")
    predTheta = inverseTransformColumnGreeks(dataSet, scaler, scaledTheta,
                                             "Price", "Maturity")

    scaledGammaK = pd.Series(gamma.flatten(), index=dataSet.index).rename("GammaK")
    predGammaK = inverseTransformColumnGreeks(dataSet, scaler, scaledGammaK,
                                              "Price", "ChangedStrike", order=2)
    print("Number of arbitrage violation : ", np.sum((theta < 0) + (gamma < 0)) )
    return predPrice, predDupire, predTheta, predGammaK, lossEpoch

#Penalization for pseudo local volatility
def intervalRegularization(localVariance, weighting, hyperParameters):
  lowerVolBound = hyperParameters["DupireVolLowerBound"]
  upperVolBound = hyperParameters["DupireVolUpperBound"]
  no_nans = tf.clip_by_value(localVariance, 0, hyperParameters["DupireVarCap"])
  reg = tf.nn.relu(tf.square(lowerVolBound) - no_nans) + tf.nn.relu(no_nans - tf.square(upperVolBound))
  lambdas = hyperParameters["lambdaLocVol"] * tf.reduce_mean(weighting)
  return lambdas * tf.reduce_mean(tf.boolean_mask(reg, tf.is_finite(reg)))

#Add above regularization to the list of penalization
def addDupireRegularisation(priceTensor, tensorList, penalizationList,
                            formattingResultFunction, weighting, hyperParameters):
    updatedPenalizationList = penalizationList + [intervalRegularization(tensorList[-1], weighting, hyperParameters)]
    return priceTensor, tensorList, updatedPenalizationList, formattingResultFunction

#Mini-batch sampling methods for large datasets
def selectMiniBatchWithoutReplacement(dataSet, batch_size):
    nbObs = dataSet.shape[0]
    idx = np.arange(nbObs)
    np.random.shuffle(idx)
    nbBatches = int(np.ceil(nbObs/batch_size))
    xBatchList = []
    lastBatchIndex = 0
    for i in range(nbBatches):
        firstBatchIndex = i*batch_size
        lastBatchIndex = (i+1)*batch_size
        xBatchList.append(dataSet.iloc[idx[firstBatchIndex:lastBatchIndex],:])
    xBatchList.append(dataSet.iloc[idx[lastBatchIndex:],:])
    return xBatchList

def selectMiniBatchWithReplacement(dataSet, batch_size):
    nbObs = dataSet.shape[0]
    nbBatches = int(np.ceil(nbObs/batch_size)) + 1
    xBatchList = []
    lastBatchIndex = 0
    for i in range(nbBatches):
        idx = np.random.randint(nbObs, size = batch_size)
        xBatchList.append(dataSet.iloc[idx,:])
    return xBatchList

def computePriceWeighting(batch, scaler):
    def secondMin(row):
        return row.sort_values().iloc[1]
    coordinates = dataSetConstruction.transformCustomMinMax(batch, scaler)[["ChangedStrike", "logMaturity"]]
    distanceToClosestPoint = pd.DataFrame(scipy.spatial.distance_matrix(coordinates,
                                                                        coordinates), 
                                          index = coordinates.index,
                                          columns = coordinates.index).apply(secondMin)
    return distanceToClosestPoint

############################################################################################################# Price Neural network API

# Train neural network with a decreasing rule for learning rate
# NNFactory :  function creating the architecture
# dataSet : training data
# activateRegularization : boolean, if true add bound penalization to dupire variance
# hyperparameters : dictionnary containing various hyperparameters
# modelName : name under which tensorflow model is saved
def create_train_model(NNFactory,
                       dataSet,
                       activateRegularization,
                       hyperparameters,
                       scaler,
                       modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]
    nbEpoch = hyperparameters["maxEpoch"]
    nbEpochFork = hyperparameters["nbEpochFork"]
    fixedLearningRate = (None if (("FixedLearningRate" in hyperparameters) and hyperparameters["FixedLearningRate"]) else hyperparameters["LearningRateStart"])
    patience = hyperparameters["Patience"]
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    holderExponent = (hyperparameters["HolderExponent"] if ("HolderExponent" in hyperparameters) else 2.0)

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]
    learningRateEpoch = 0
    finalLearningRate = hyperparameters["FinalLearningRate"]

    batch_size = hyperparameters["batchSize"]

    start = time.time()
    # Reset the graph
    resetTensorflow()
    # Placeholders for input and output data
    Strike = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    factorPrice = tf.placeholder(tf.float32, [None, 1])
    StrikePenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    yBid = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yBid')
    yAsk = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yAsk')
    
    weighting = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weighting')
    weightingPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weightingPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])


    # Get scaling for strike
    colStrikeIndex = dataSet.columns.get_loc("ChangedStrike")
    maxColFunction = scaler.data_max_[colStrikeIndex]
    minColFunction = scaler.data_min_[colStrikeIndex]
    scF = (maxColFunction - minColFunction) #0
    scaleTensor = tf.constant(scF, dtype=tf.float32) #1
    StrikeMinTensor = tf.constant(minColFunction, dtype=tf.float32)
    
    
    # Get scaling for maturity
    colMaturityIndex = dataSet.columns.get_loc("logMaturity") if useLogMaturity else dataSet.columns.get_loc("Maturity")
    maxColFunctionMat = scaler.data_max_[colMaturityIndex]
    minColFunctionMat = scaler.data_min_[colMaturityIndex] #0
    scFMat = (maxColFunctionMat - minColFunctionMat) #1
    scaleTensorMaturity = tf.constant(scFMat, dtype=tf.float32)
    maturityMinTensor = tf.constant(minColFunctionMat, dtype=tf.float32)
    
    # Get scaling for price
    colPriceIndex = dataSet.columns.get_loc("Price")
    maxColFunctionPrice = scaler.data_max_[colPriceIndex]
    minColFunctionPrice = scaler.data_min_[colPriceIndex] #0
    scFPrice = (maxColFunctionPrice - minColFunctionPrice) #1
    scaleTensorPrice = tf.constant(scFPrice, dtype=tf.float32)
    priceMinTensor = tf.constant(minColFunctionPrice, dtype=tf.float32)
    
    tensorDict = {}
    tensorDict["Maturity"] = Maturity
    tensorDict["Strike"] = Strike
    tensorDict["scaleTensorStrike"] = scaleTensor
    tensorDict["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDict["StrikeMinTensor"] = StrikeMinTensor
    tensorDict["maturityMinTensor"] = maturityMinTensor
    tensorDict["PriceMinTensor"] = priceMinTensor
    tensorDict["scaleTensorPrice"] = scaleTensorPrice
    tensorDict["lossWeighting"] = weighting
    
    tensorDictPenalization = {}
    tensorDictPenalization["Maturity"] = MaturityPenalization
    tensorDictPenalization["Strike"] = StrikePenalization
    tensorDictPenalization["scaleTensorStrike"] = scaleTensor
    tensorDictPenalization["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDictPenalization["StrikeMinTensor"] = StrikeMinTensor
    tensorDictPenalization["maturityMinTensor"] = maturityMinTensor
    tensorDictPenalization["PriceMinTensor"] = priceMinTensor
    tensorDictPenalization["scaleTensorPrice"] = scaleTensorPrice
    tensorDictPenalization["lossWeighting"] = weightingPenalization
        
    #Grid on which is applied Penalization [0, 2 * maxMaturity] and [minMoneyness, 2 * maxMoneyness]
    
    k = np.linspace(-0.5 * minColFunction / scF,
                    (2.0 * maxColFunction - minColFunction) / scF,
                    num=50)
    
    if useLogMaturity :
        #t = np.linspace(minColFunctionMat, np.log(2) + maxColFunctionMat, num=100)
        t = np.linspace(0, (np.log(4) + maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    else :
        #t = np.linspace(0, 4, num=100)
        #t = np.linspace(minColFunctionMat, 2 * maxColFunctionMat, num=100)
        t = np.linspace(0, (4 * maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    with tf.device("/gpu:0"):
        if activateRegularization:  # Add pseudo local volatility regularisation
            price_pred_tensor, TensorList, penalizationList, formattingFunction = addDupireRegularisation(
                  *NNFactory(hidden_nodes,
                             tensorDict,
                             hyperparameters),
                  weighting,
                  hyperparameters)
            price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = addDupireRegularisation(
                  *NNFactory(hidden_nodes,
                             tensorDictPenalization,
                             hyperparameters),
                  weightingPenalization,
                  hyperparameters)
        else:
            price_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                            tensorDict,
                                                                                            hyperparameters)
            price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                                tensorDictPenalization,
                                                                                                hyperparameters)

        price_pred_tensor_sc = tf.multiply(factorPrice, price_pred_tensor * scaleTensorPrice + priceMinTensor)
        unscaledMaturityTensor = (Maturity * scaleTensorMaturity + maturityMinTensor)
        maturityOriginal = tf.exp(unscaledMaturityTensor) if useLogMaturity else unscaledMaturityTensor
        TensorList[0] = price_pred_tensor_sc

        # Define a loss function
        #pointwiseError = tf.pow(tf.reduce_mean(tf.pow((price_pred_tensor_sc - y)/y , holderExponent) * weighting), 1.0 / holderExponent)
        pointwiseError = tf.pow(tf.reduce_mean(tf.pow((price_pred_tensor_sc - y) , holderExponent) * weighting), 1.0 / holderExponent)
        errors = tf.add_n([pointwiseError] + penalizationList1) #tf.add_n([pointwiseError] + penalizationList)
        loss = tf.log(tf.reduce_mean(errors))
        
        forkPenalization = hyperparameters["lambdaFork"] * tf.reduce_mean(tf.square(tf.nn.relu(yBid - price_pred_tensor_sc) / yBid) + tf.square(tf.nn.relu(price_pred_tensor_sc - yAsk) / yAsk))
        errorFork = tf.add_n([pointwiseError, forkPenalization] + penalizationList1)
        lossFork = tf.log(tf.reduce_mean(errorFork))

        # Define a train operation to minimize the loss
        lr = learningRate

        optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
        train = optimizer.minimize(loss)
        trainFork = optimizer.minimize(lossFork)

        # Initialize variables and run session
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)
    n = dataSet.shape[0]
    loss_serie = []
    
    #Create weighting to focus learning on isolated points


    def createFeedDict(batch):
        batchSize = batch.shape[0]
        scaledBatch = dataSetConstruction.transformCustomMinMax(batch, scaler)
        weighthingBatch = computePriceWeighting(batch, scaler)
        feedDict = {Strike: scaledBatch["ChangedStrike"].values.reshape(batchSize, 1),  
                    Maturity:  scaledBatch["logMaturity"].values.reshape(batchSize, 1) if useLogMaturity else scaledBatch["Maturity"].values.reshape(batchSize, 1),  
                    y: batch["Price"].values.reshape(batchSize, 1),
                    yBid : batch["Bid"].values.reshape(batchSize, 1),
                    yAsk : batch["Ask"].values.reshape(batchSize, 1),
                    StrikePenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization, 1),
                    factorPrice: batch["DividendFactor"].values.reshape(batchSize, 1),
                    learningRateTensor: learningRate,
                    weighting: np.ones_like(batch["ChangedStrike"].values.reshape(batchSize, 1)),
                    weightingPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict

    # Learning rate is divided by 10 if no imporvement is observed for training loss after "patience" epochs
    def updateLearningRate(iterNumber, lr, lrEpoch):
        if not activateLearningDecrease:
            print("Constant learning rate, stop training")
            return False, lr, lrEpoch
        if learningRate > finalLearningRate:
            lr *= 0.1
            lrEpoch = iterNumber
            saver.restore(sess, modelFolder + modelName)
            print("Iteration : ", lrEpoch, "new learning rate : ", lr)
        else:
            print("Last Iteration : ", lrEpoch, "final learning rate : ", lr)
            return False, lr, lrEpoch
        return True, lr, lrEpoch

    epochFeedDict = createFeedDict(dataSet)

    def evalBestModel():
        if not activateLearningDecrease:
            print("Learning rate : ", learningRate, " final loss : ", min(loss_serie))
        currentBestLoss = sess.run(loss, feed_dict=epochFeedDict)
        currentBestPenalizations = sess.run([pointwiseError, forkPenalization, penalizationList], feed_dict=epochFeedDict)
        currentBestPenalizations1 = sess.run([penalizationList1], feed_dict=epochFeedDict)
        print("Best loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, len(loss_serie), currentBestLoss))
        print("Best Penalization : ", currentBestPenalizations)
        print("Best Penalization (Refined Grid) : ", currentBestPenalizations1)
        return
    
    print("Training w.r.t. price RMSE and arbitrage constraints")
    debug = True
    for i in range(nbEpoch):
        miniBatchList = [dataSet]
        
        if len(miniBatchList) > 1 :
            for k in range(len(miniBatchList)):
                batchFeedDict = createFeedDict(miniBatchList[k])
                sess.run(train, feed_dict=batchFeedDict)
        else : 
            sess.run(train, feed_dict=epochFeedDict)
        loss_serie.append(sess.run(loss, feed_dict=epochFeedDict))

        if (len(loss_serie) < 2) or (loss_serie[-1] <= min(loss_serie)):
            # Save model as model is improved
            saver.save(sess, modelFolder + modelName)
        if debug and (np.isnan(loss_serie[-1])):
            print("Epoch : ", i)
            print(np.isnan(y.eval(session=sess, feed_dict=epochFeedDict)).any())
            print(np.isnan(price_pred_tensor_sc.eval(session=sess, feed_dict=epochFeedDict)).any())
            print(np.isnan(price_pred_tensor.eval(session=sess, feed_dict=epochFeedDict)).any())
            print(np.isnan(factorPrice.eval(session=sess, feed_dict=epochFeedDict)).any())
            print(np.isnan(Strike.eval(session=sess, feed_dict=epochFeedDict)).any())
            print(np.isnan(Maturity.eval(session=sess, feed_dict=epochFeedDict)).any())
            print(np.isnan(pointwiseError.eval(session=sess, feed_dict=epochFeedDict)).any())
            print(np.isnan(tf.add_n(penalizationList1).eval(session=sess, feed_dict=epochFeedDict)).any())
        if (np.isnan(loss_serie[-1]) or  # Unstable model
                ((i - learningRateEpoch >= patience) and (min(loss_serie[-patience:]) > min(
                    loss_serie)))):  # No improvement for training loss during the latest 100 iterations
            continueTraining, learningRate, learningRateEpoch = updateLearningRate(i, learningRate, learningRateEpoch)
            if continueTraining:
                evalBestModel()
            else:
                break
    saver.restore(sess, modelFolder + modelName)
    
    print("Training w.r.t. price RMSE, arbitrage constraints and bid-ask fork violation")
    learningRate = hyperparameters["LearningRateStart"]
    loss_serie_fork = []
    learningRateEpoch = 0
    
    for i in range(nbEpochFork):
        miniBatchList = [dataSet]
        
        for k in range(len(miniBatchList)):
            batchFeedDict = createFeedDict(miniBatchList[k])
            sess.run(trainFork, feed_dict=batchFeedDict)

        loss_serie_fork.append(sess.run(lossFork, feed_dict=epochFeedDict))

        if (len(loss_serie_fork) > 1) and (loss_serie_fork[-1] <= min(loss_serie_fork)):
            # Save model as error is improved
            saver.save(sess, modelFolder + modelName)
        if (np.isnan(loss_serie_fork[-1]) or  # Unstable training
                ((i - learningRateEpoch >= patience) and (min(loss_serie[-patience:]) > min(
                    loss_serie)))):  # No improvement for training loss during the latest 100 iterations
            continueTraining, learningRate, learningRateEpoch = updateLearningRate(i, learningRate, learningRateEpoch)
            if continueTraining:
                evalBestModel()
            else:
                break
    
    saver.restore(sess, modelFolder + modelName)
    evalBestModel()
    
    #Count arbitrage violations on refined grid (grid on which are applied penalizations)
    print("Refined Grid evaluation :")
    refinedEpochDict = {Strike: np.expand_dims(kPenalization, 1),
                        Maturity:  np.expand_dims(tPenalization, 1),  
                        y: np.ones_like(np.expand_dims(kPenalization, 1)),
                        yBid : np.ones_like(np.expand_dims(kPenalization, 1)),
                        yAsk : np.ones_like(np.expand_dims(kPenalization, 1)),
                        StrikePenalization : np.expand_dims(kPenalization, 1),
                        MaturityPenalization : np.expand_dims(tPenalization, 1),
                        factorPrice: np.expand_dims(tPenalization, 1),
                        learningRateTensor: learningRate,
                        weighting: np.ones_like(np.expand_dims(kPenalization, 1)),
                        weightingPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
    evalRefinedList = sess.run(TensorList, feed_dict=refinedEpochDict)
    emptyDf = pd.DataFrame(np.ones((kPenalization.size, dataSet.shape[1])), 
                           index = pd.MultiIndex.from_tuples( list(zip(kPenalization, tPenalization)) ),
                           columns = dataSet.columns)
    formattingFunction(*evalRefinedList, loss_serie, emptyDf, scaler)
    
    print("Dataset Grid evaluation :")
    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()
    end = time.time()
    print("Training Time : ", end - start)

    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    lossEpochSerie = pd.Series(loss_serie + loss_serie_fork)
    lossEpochSerie.to_csv(modelFolder +  "loss" + modelName + ".csv", header = True)

    return formattingFunction(*evalList, loss_serie, dataSet, scaler)



# Train neural network with a decreasing rule for learning rate
# NNFactory :  function creating the architecture
# dataSet : training data
# activateRegularization : boolean, if true add bound penalization to dupire variance
# hyperparameters : dictionnary containing various hyperparameters
# modelName : name under which tensorflow model is saved
def create_eval_model(NNFactory,
                      dataSet,
                      activateRegularization,
                      hyperparameters,
                      scaler,
                      modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]
    nbEpoch = hyperparameters["maxEpoch"]
    nbEpochFork = hyperparameters["nbEpochFork"]
    fixedLearningRate = (None if (("FixedLearningRate" in hyperparameters) and hyperparameters["FixedLearningRate"]) else hyperparameters["LearningRateStart"])
    patience = hyperparameters["Patience"]
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    holderExponent = (hyperparameters["HolderExponent"] if ("HolderExponent" in hyperparameters) else 2.0)

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]
    learningRateEpoch = 0
    finalLearningRate = hyperparameters["FinalLearningRate"]

    batch_size = hyperparameters["batchSize"]

    start = time.time()
    # Reset the graph
    resetTensorflow()
    # Placeholders for input and output data
    Strike = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    factorPrice = tf.placeholder(tf.float32, [None, 1])
    StrikePenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    yBid = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yBid')
    yAsk = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yAsk')
    
    weighting = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weighting')
    weightingPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weightingPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])


    # Get scaling for strike
    colStrikeIndex = dataSet.columns.get_loc("ChangedStrike")
    maxColFunction = scaler.data_max_[colStrikeIndex]
    minColFunction = scaler.data_min_[colStrikeIndex]
    scF = (maxColFunction - minColFunction) #0
    scaleTensor = tf.constant(scF, dtype=tf.float32) #1
    StrikeMinTensor = tf.constant(minColFunction, dtype=tf.float32)
    
    
    # Get scaling for maturity
    colMaturityIndex = dataSet.columns.get_loc("logMaturity") if useLogMaturity else dataSet.columns.get_loc("Maturity")
    maxColFunctionMat = scaler.data_max_[colMaturityIndex]
    minColFunctionMat = scaler.data_min_[colMaturityIndex] #0
    scFMat = (maxColFunctionMat - minColFunctionMat) #1
    scaleTensorMaturity = tf.constant(scFMat, dtype=tf.float32)
    maturityMinTensor = tf.constant(minColFunctionMat, dtype=tf.float32)
    
    # Get scaling for price
    colPriceIndex = dataSet.columns.get_loc("Price")
    maxColFunctionPrice = scaler.data_max_[colPriceIndex]
    minColFunctionPrice = scaler.data_min_[colPriceIndex] #0
    scFPrice = (maxColFunctionPrice - minColFunctionPrice) #1
    scaleTensorPrice = tf.constant(scFPrice, dtype=tf.float32)
    priceMinTensor = tf.constant(minColFunctionPrice, dtype=tf.float32)
    
    tensorDict = {}
    tensorDict["Maturity"] = Maturity
    tensorDict["Strike"] = Strike
    tensorDict["scaleTensorStrike"] = scaleTensor
    tensorDict["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDict["StrikeMinTensor"] = StrikeMinTensor
    tensorDict["maturityMinTensor"] = maturityMinTensor
    tensorDict["PriceMinTensor"] = priceMinTensor
    tensorDict["scaleTensorPrice"] = scaleTensorPrice
    tensorDict["lossWeighting"] = weighting
    
    tensorDictPenalization = {}
    tensorDictPenalization["Maturity"] = MaturityPenalization
    tensorDictPenalization["Strike"] = StrikePenalization
    tensorDictPenalization["scaleTensorStrike"] = scaleTensor
    tensorDictPenalization["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDictPenalization["StrikeMinTensor"] = StrikeMinTensor
    tensorDictPenalization["maturityMinTensor"] = maturityMinTensor
    tensorDictPenalization["PriceMinTensor"] = priceMinTensor
    tensorDictPenalization["scaleTensorPrice"] = scaleTensorPrice
    tensorDictPenalization["lossWeighting"] = weightingPenalization
        
    #Grid on which is applied Penalization [0, 2 * maxMaturity] and [minMoneyness, 2 * maxMoneyness]
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                2.0 * scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    
    #k = np.linspace((np.log(0.5) + minColFunction) / scF,
    #                (np.log(2.0) + maxColFunction) / scF,
    #                num=50)
    
    k = np.linspace(-0.5 * minColFunction / scF,
                    (2.0 * maxColFunction - minColFunction) / scF,
                    num=50)
    
    if useLogMaturity :
        #t = np.linspace(minColFunctionMat, np.log(2) + maxColFunctionMat, num=100)
        t = np.linspace(0, (np.log(4) + maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    else :
        #t = np.linspace(0, 4, num=100)
        #t = np.linspace(minColFunctionMat, 2 * maxColFunctionMat, num=100)
        t = np.linspace(0, (4 * maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    with tf.device("/gpu:0"):
        if activateRegularization:  # Add pseudo local volatility regularisation
            price_pred_tensor, TensorList, penalizationList, formattingFunction = addDupireRegularisation(
                  *NNFactory(hidden_nodes,
                             tensorDict,
                             hyperparameters),
                  weighting,
                  hyperparameters)
            price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = addDupireRegularisation(
                  *NNFactory(hidden_nodes,
                             tensorDictPenalization,
                             hyperparameters),
                  weightingPenalization,
                  hyperparameters)
        else:
            price_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                            tensorDict,
                                                                                            hyperparameters)
            price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                                tensorDictPenalization,
                                                                                                hyperparameters)

        price_pred_tensor_sc = tf.multiply(factorPrice, price_pred_tensor * scaleTensorPrice + priceMinTensor)
        unscaledMaturityTensor = (Maturity * scaleTensorMaturity + maturityMinTensor)
        maturityOriginal = tf.exp(unscaledMaturityTensor) if useLogMaturity else unscaledMaturityTensor
        TensorList[0] = price_pred_tensor_sc

        # Define a loss function
        #pointwiseError = tf.pow(tf.reduce_mean(tf.pow((price_pred_tensor_sc - y)/y , holderExponent) * weighting), 1.0 / holderExponent)
        pointwiseError = tf.pow(tf.reduce_mean(tf.pow((price_pred_tensor_sc - y) , holderExponent) * weighting), 1.0 / holderExponent)
        errors = tf.add_n([pointwiseError] + penalizationList1) #tf.add_n([pointwiseError] + penalizationList)
        loss = tf.log(tf.reduce_mean(errors))
        
        forkPenalization = hyperparameters["lambdaFork"] * tf.reduce_mean(tf.square(tf.nn.relu(yBid - price_pred_tensor_sc) / yBid) + tf.square(tf.nn.relu(price_pred_tensor_sc - yAsk) / yAsk))
        errorFork = tf.add_n([pointwiseError, forkPenalization] + penalizationList1)
        lossFork = tf.log(tf.reduce_mean(errorFork))

        # Define a train operation to minimize the loss
        lr = learningRate

        optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
        train = optimizer.minimize(loss)
        trainFork = optimizer.minimize(lossFork)

        # Initialize variables and run session
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)
    n = dataSet.shape[0]
    loss_serie = []
    
    #Create weighting to focus learning on isolated points


    def createFeedDict(batch):
        batchSize = batch.shape[0]
        scaledBatch = dataSetConstruction.transformCustomMinMax(batch, scaler)
        weighthingBatch = computePriceWeighting(batch, scaler)
        feedDict = {Strike: scaledBatch["ChangedStrike"].values.reshape(batchSize, 1),  
                    Maturity:  scaledBatch["logMaturity"].values.reshape(batchSize, 1) if useLogMaturity else scaledBatch["Maturity"].values.reshape(batchSize, 1),  
                    y: batch["Price"].values.reshape(batchSize, 1),
                    yBid : batch["Bid"].values.reshape(batchSize, 1),
                    yAsk : batch["Ask"].values.reshape(batchSize, 1),
                    StrikePenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization, 1),
                    factorPrice: batch["DividendFactor"].values.reshape(batchSize, 1),
                    learningRateTensor: learningRate,
                    weighting: np.ones_like(batch["ChangedStrike"].values.reshape(batchSize, 1)),
                    weightingPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict

    # Learning rate is divided by 10 if no imporvement is observed for training loss after "patience" epochs
    def updateLearningRate(iterNumber, lr, lrEpoch):
        if not activateLearningDecrease:
            print("Constant learning rate, stop training")
            return False, lr, lrEpoch
        if learningRate > finalLearningRate:
            lr *= 0.1
            lrEpoch = iterNumber
            saver.restore(sess, modelFolder + modelName)
            print("Iteration : ", lrEpoch, "new learning rate : ", lr)
        else:
            print("Last Iteration : ", lrEpoch, "final learning rate : ", lr)
            return False, lr, lrEpoch
        return True, lr, lrEpoch

    epochFeedDict = createFeedDict(dataSet)

    def evalBestModel():
        if not activateLearningDecrease:
            print("Learning rate : ", learningRate, " final loss : ", min(loss_serie))
        currentBestLoss = sess.run(loss, feed_dict=epochFeedDict)
        currentBestPenalizations = sess.run([pointwiseError, forkPenalization, penalizationList], feed_dict=epochFeedDict)
        currentBestPenalizations1 = sess.run([penalizationList1], feed_dict=epochFeedDict)
        print("Best loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, len(loss_serie), currentBestLoss))
        print("Best Penalization : ", currentBestPenalizations)
        print("Best Penalization (Refined Grid) : ", currentBestPenalizations1)
        return
    
    
    saver.restore(sess, modelFolder + modelName)
    evalBestModel()
    
    #Count arbitrage violations on refined grid (grid on which are applied penalizations)
    print("Refined Grid evaluation :")
    refinedEpochDict = {Strike: np.expand_dims(kPenalization, 1),
                        Maturity:  np.expand_dims(tPenalization, 1),  
                        y: np.ones_like(np.expand_dims(kPenalization, 1)),
                        yBid : np.ones_like(np.expand_dims(kPenalization, 1)),
                        yAsk : np.ones_like(np.expand_dims(kPenalization, 1)),
                        StrikePenalization : np.expand_dims(kPenalization, 1),
                        MaturityPenalization : np.expand_dims(tPenalization, 1),
                        factorPrice: np.expand_dims(tPenalization, 1),
                        learningRateTensor: learningRate,
                        weighting: np.ones_like(np.expand_dims(kPenalization, 1)),
                        weightingPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
    evalRefinedList = sess.run(TensorList, feed_dict=refinedEpochDict)
    emptyDf = pd.DataFrame(np.ones((kPenalization.size, dataSet.shape[1])), 
                           index = pd.MultiIndex.from_tuples( list(zip(kPenalization, tPenalization)) ),
                           columns = dataSet.columns)
    formattingFunction(*evalRefinedList, loss_serie, emptyDf, scaler)
    
    print("Dataset Grid evaluation :")
    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()

    return formattingFunction(*evalList, loss_serie, dataSet, scaler)





def evalVolLocale(NNFactory,
                  strikes,
                  maturities,
                  dataSet,
                  hyperparameters,
                  scaler,
                  bootstrap,
                  S0,
                  modelName="bestModel"):
    
    hidden_nodes = hyperparameters["nbUnits"]
    nbEpoch = hyperparameters["maxEpoch"]
    nbEpochFork = hyperparameters["nbEpochFork"]
    fixedLearningRate = (None if (("FixedLearningRate" in hyperparameters) and hyperparameters["FixedLearningRate"]) else hyperparameters["LearningRateStart"])
    patience = hyperparameters["Patience"]
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    holderExponent = (hyperparameters["HolderExponent"] if ("HolderExponent" in hyperparameters) else 2.0)

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]
    learningRateEpoch = 0
    finalLearningRate = hyperparameters["FinalLearningRate"]

    batch_size = hyperparameters["batchSize"]

    start = time.time()
    # Reset the graph
    resetTensorflow()
    # Placeholders for input and output data
    Strike = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    factorPrice = tf.placeholder(tf.float32, [None, 1])
    StrikePenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    yBid = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yBid')
    yAsk = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yAsk')
    
    weighting = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weighting')
    weightingPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weightingPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])


    # Get scaling for strike
    colStrikeIndex = dataSet.columns.get_loc("ChangedStrike")
    maxColFunction = scaler.data_max_[colStrikeIndex]
    minColFunction = scaler.data_min_[colStrikeIndex]
    scF = (maxColFunction - minColFunction) #0
    scaleTensor = tf.constant(scF, dtype=tf.float32) #1
    StrikeMinTensor = tf.constant(minColFunction, dtype=tf.float32)
    
    
    # Get scaling for maturity
    colMaturityIndex = dataSet.columns.get_loc("logMaturity") if useLogMaturity else dataSet.columns.get_loc("Maturity")
    maxColFunctionMat = scaler.data_max_[colMaturityIndex]
    minColFunctionMat = scaler.data_min_[colMaturityIndex] #0
    scFMat = (maxColFunctionMat - minColFunctionMat) #1
    scaleTensorMaturity = tf.constant(scFMat, dtype=tf.float32)
    maturityMinTensor = tf.constant(minColFunctionMat, dtype=tf.float32)
    
    # Get scaling for price
    colPriceIndex = dataSet.columns.get_loc("Price")
    maxColFunctionPrice = scaler.data_max_[colPriceIndex]
    minColFunctionPrice = scaler.data_min_[colPriceIndex] #0
    scFPrice = (maxColFunctionPrice - minColFunctionPrice) #1
    scaleTensorPrice = tf.constant(scFPrice, dtype=tf.float32)
    priceMinTensor = tf.constant(minColFunctionPrice, dtype=tf.float32)
    
    tensorDict = {}
    tensorDict["Maturity"] = Maturity
    tensorDict["Strike"] = Strike
    tensorDict["scaleTensorStrike"] = scaleTensor
    tensorDict["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDict["StrikeMinTensor"] = StrikeMinTensor
    tensorDict["maturityMinTensor"] = maturityMinTensor
    tensorDict["PriceMinTensor"] = priceMinTensor
    tensorDict["scaleTensorPrice"] = scaleTensorPrice
    tensorDict["lossWeighting"] = weighting
    
    tensorDictPenalization = {}
    tensorDictPenalization["Maturity"] = MaturityPenalization
    tensorDictPenalization["Strike"] = StrikePenalization
    tensorDictPenalization["scaleTensorStrike"] = scaleTensor
    tensorDictPenalization["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDictPenalization["StrikeMinTensor"] = StrikeMinTensor
    tensorDictPenalization["maturityMinTensor"] = maturityMinTensor
    tensorDictPenalization["PriceMinTensor"] = priceMinTensor
    tensorDictPenalization["scaleTensorPrice"] = scaleTensorPrice
    tensorDictPenalization["lossWeighting"] = weightingPenalization
        
    #Grid on which is applied Penalization [0, 2 * maxMaturity] and [minMoneyness, 2 * maxMoneyness]
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                2.0 * scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    
    #k = np.linspace((np.log(0.5) + minColFunction) / scF,
    #                (np.log(2.0) + maxColFunction) / scF,
    #                num=50)
    
    k = np.linspace(-0.5 * minColFunction / scF,
                    (2.0 * maxColFunction - minColFunction) / scF,
                    num=50)
    
    if useLogMaturity :
        #t = np.linspace(minColFunctionMat, np.log(2) + maxColFunctionMat, num=100)
        t = np.linspace((np.log(0.00001) - minColFunctionMat) / scFMat, 
                        (np.log(4) + maxColFunctionMat - minColFunctionMat) / scFMat, 
                        num=100)
    else :
        #t = np.linspace(0, 4, num=100)
        #t = np.linspace(minColFunctionMat, 2 * maxColFunctionMat, num=100)
        t = np.linspace(0, (4 * maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])
    
    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    with tf.device("/gpu:0"):
        price_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                        tensorDict,
                                                                                        hyperparameters)
        price_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                            tensorDictPenalization,
                                                                                            hyperparameters)

        price_pred_tensor_sc = tf.multiply(factorPrice, price_pred_tensor * scaleTensorPrice + priceMinTensor)
        unscaledMaturityTensor = (Maturity * scaleTensorMaturity + maturityMinTensor)
        maturityOriginal = tf.exp(unscaledMaturityTensor) if useLogMaturity else unscaledMaturityTensor
        TensorList[0] = price_pred_tensor_sc

        # Define a loss function
        #pointwiseError = tf.pow(tf.reduce_mean(tf.pow((price_pred_tensor_sc - y)/y , holderExponent) * weighting), 1.0 / holderExponent)
        pointwiseError = tf.pow(tf.reduce_mean(tf.pow((price_pred_tensor_sc - y) , holderExponent) * weighting), 1.0 / holderExponent)
        errors = tf.add_n([pointwiseError] + penalizationList1) #tf.add_n([pointwiseError] + penalizationList)
        loss = tf.log(tf.reduce_mean(errors))
        
        forkPenalization = hyperparameters["lambdaFork"] * tf.reduce_mean(tf.square(tf.nn.relu(yBid - price_pred_tensor_sc) / yBid) + tf.square(tf.nn.relu(price_pred_tensor_sc - yAsk) / yAsk))
        errorFork = tf.add_n([pointwiseError, forkPenalization] + penalizationList1)
        lossFork = tf.log(tf.reduce_mean(errorFork))

        # Define a train operation to minimize the loss
        lr = learningRate

        optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
        train = optimizer.minimize(loss)
        trainFork = optimizer.minimize(lossFork)

        # Initialize variables and run session
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)
    n = dataSet.shape[0]
    loss_serie = []
    changedVar = bootstrap.changeOfVariable(strikes, maturities)
    scaledStrike = (changedVar[0] - minColFunction) / scF
    dividendFactor = changedVar[1]

    def createFeedDict(m, t, d):
        batchSize = m.shape[0]
        feedDict = {Strike: np.reshape(m, (batchSize, 1)),
                    Maturity: np.reshape(t, (batchSize, 1)),
                    StrikePenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization,1),
                    factorPrice: np.reshape(d, (batchSize, 1)),
                    weighting: np.ones((batchSize, 1)),
                    weightingPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict
    
    scaledMaturities = ((np.log(maturities) if useLogMaturity else maturities) - minColFunctionMat) / scFMat
    epochFeedDict = createFeedDict(scaledStrike, scaledMaturities, dividendFactor)

    saver.restore(sess, modelFolder + modelName)

    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()

    return pd.Series(evalList[1].flatten(),
                     index=pd.MultiIndex.from_arrays([strikes, maturities], names=('Strike', 'Maturity')))

################################################################################################################## Local volatility functions

#Dupire formula from exact derivative computation
def dupireFormula(priceTensor, 
                  tensorDict,
                  weighting,
                  hyperparameters):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    priceMinTensor = tensorDict["PriceMinTensor"]
    scaleTensorPrice = tensorDict["scaleTensorPrice"]
    
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    maturity = scaledMaturityTensor * scaleTensorMaturity + maturityMinTensor
    
    dK = tf.gradients(priceTensor, scaledStrikeTensor, name="dK") / scaleTensor * scaleTensorPrice
    hK = tf.gradients(dK[0], scaledStrikeTensor, name="hK") / scaleTensor
    if useLogMaturity :
        theta = tf.gradients(priceTensor, scaledMaturityTensor, name="dT") / scaleTensorMaturity * scaleTensorPrice
    else :
        theta = tf.gradients(priceTensor, scaledMaturityTensor, name="dT") / scaleTensorMaturity / tf.exp(maturity) * scaleTensorPrice
    
    dupireDenominator = tf.square(scaledStrikeTensor * scaleTensor + StrikeMinTensor) * hK
    
    #Initial weights of neural network can be random which lead to negative dupireVar
    dupireVar = 2 * theta / dupireDenominator
    dupireVol = tf.sqrt(dupireVar)
    return dupireVol, dupireVar

#Dupire formula with derivative obtained from native tensorflow algorithmic differentiation
def rawDupireFormula(priceTensor, 
                     tensorDict,
                     weighting,
                     hyperparameters):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    priceMinTensor = tensorDict["PriceMinTensor"]
    scaleTensorPrice = tensorDict["scaleTensorPrice"]
    
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    maturity = scaledMaturityTensor * scaleTensorMaturity + maturityMinTensor
    
    dK = tf.gradients(priceTensor, scaledStrikeTensor, name="dK") / scaleTensor * scaleTensorPrice
    hK = tf.gradients(dK[0], scaledStrikeTensor, name="hK") / scaleTensor
    if useLogMaturity :
        theta = tf.gradients(priceTensor, scaledMaturityTensor, name="dT") / scaleTensorMaturity / tf.exp(maturity) * scaleTensorPrice
    else :
        theta = tf.gradients(priceTensor, scaledMaturityTensor, name="dT") / scaleTensorMaturity * scaleTensorPrice
    
    dupireDenominator = tf.square(scaledStrikeTensor * scaleTensor + StrikeMinTensor) * hK
    
    #Initial weights of neural network can be random which lead to negative dupireVar
    dupireVar = 2 * theta / dupireDenominator
    dupireVol = tf.sqrt(dupireVar)
    return  dupireVol, theta, hK, dupireVar


################################################################################################################## Soft constraints penalties

# Soft constraints for strike convexity and strike/maturity monotonicity
def arbitragePenaltiesPrice(theta, 
                            hK, 
                            tensorDict,
                            weighting,
                            hyperparameters):

    lambdas = tf.reduce_mean(weighting)
    lowerBoundTheta = tf.constant(hyperparameters["lowerBoundTheta"])
    lowerBoundGamma = tf.constant(hyperparameters["lowerBoundGamma"])
    grad_penalty = lambdas * hyperparameters["lambdaSoft"] * tf.reduce_mean(tf.nn.relu(-theta[0] + lowerBoundTheta))
    hessian_penalty = lambdas * hyperparameters["lambdaGamma"] * tf.reduce_mean( tf.nn.relu(-hK[0] + lowerBoundGamma) )

    return [grad_penalty, hessian_penalty]


############################################################################# Tools function for Neural network architecture
############################################################################# Hard constraints architecture

# Initilize weights as positive
def positiveKernelInitializer(shape,
                              dtype=None,
                              partition_info=None):
    return tf.abs(tf.keras.initializers.normal()(shape, dtype=dtype, partition_info=partition_info))


# Soft convex layer
def convexLayer(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.softplus(layer)


# Soft monotonic layer
def monotonicLayer(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor,
                                units=n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.sigmoid(layer)


# Soft convex layer followed by output layer for regression
def convexOutputLayer(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=2 * n_units,
                                kernel_initializer=tf.keras.initializers.glorot_normal(),
                                activation='softplus')

        layer = tf.layers.dense(layer,
                                units=1,
                                kernel_initializer=positiveKernelInitializer,
                                activation='softplus')

        return layer


# Neural network factory for Hybrid approach : splitted network with soft contraints
def NNArchitectureConstrained(n_units,
                              tensorDict,
                              hyperparameters,
                              IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    
    # First splitted layer
    hidden1S = convexLayer(n_units=n_units,
                           tensor=scaledStrikeTensor,
                           isTraining=IsTraining,
                           name="Hidden1S")

    hidden1M = monotonicLayer(n_units=n_units,
                              tensor=scaledMaturityTensor,
                              isTraining=IsTraining,
                              name="Hidden1M")

    hidden1 = tf.concat([hidden1S, hidden1M], axis=-1)

    # Second and output layer
    out = convexOutputLayer(n_units=n_units,
                            tensor=hidden1,
                            isTraining=IsTraining,
                            name="Output")
    # Soft constraints
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, 
                                                       tensorDict,
                                                       weighting,
                                                       hyperparameters)
    penaltyList = arbitragePenaltiesPrice(theta, 
                                          hK,
                                          tensorDict,
                                          weighting,
                                          hyperparameters)  

    return out, [out], penaltyList, evalAndFormatResult


################################################################### Dugas neural network architecture
def convexDugasLayer(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        nbInputFeatures = tensor.get_shape().as_list()[1]
        bias = tf.Variable(initial_value=tf.zeros_initializer()([n_units], dtype=tf.float32),
                           trainable=True,
                           shape=[n_units],
                           dtype=tf.float32,
                           name=name + "Bias")
        weights = tf.exp(tf.Variable(
            initial_value=tf.keras.initializers.glorot_normal()([nbInputFeatures, n_units], dtype=tf.float32),
            trainable=True,
            shape=[nbInputFeatures, n_units],
            dtype=tf.float32,
            name=name + "Weights"))
        layer = tf.matmul(tensor, weights) + bias
        return K.softplus(layer)


def monotonicDugasLayer(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        nbInputFeatures = tensor.get_shape().as_list()[1]
        bias = tf.Variable(initial_value=tf.zeros_initializer()([n_units], dtype=tf.float32),
                           trainable=True,
                           shape=[n_units],
                           dtype=tf.float32,
                           name=name + "Bias")
        weights = tf.exp(tf.Variable(
            initial_value=tf.keras.initializers.glorot_normal()([nbInputFeatures, n_units], dtype=tf.float32),
            trainable=True,
            shape=[nbInputFeatures, n_units],
            dtype=tf.float32,
            name=name + "Weights"))
        layer = tf.matmul(tensor, weights) + bias
        return K.sigmoid(layer)


def convexDugasOutputLayer(tensor, isTraining, name):
    with tf.name_scope(name):
        nbInputFeatures = tensor.get_shape().as_list()[1]
        bias = tf.exp(tf.Variable(initial_value=tf.zeros_initializer()([], dtype=tf.float32),
                                  shape=[],
                                  trainable=True,
                                  dtype=tf.float32,
                                  name=name + "Bias"))
        weights = tf.exp(
            tf.Variable(initial_value=tf.keras.initializers.glorot_normal()([nbInputFeatures, 1], dtype=tf.float32),
                        shape=[nbInputFeatures, 1],
                        trainable=True,
                        dtype=tf.float32,
                        name=name + "Weights"))
        layer = tf.matmul(tensor, weights) + bias
        return layer


def NNArchitectureHardConstrainedDugas(n_units,
                                       tensorDict,
                                       hyperparameters,
                                       IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    # First layer
    hidden1S = convexDugasLayer(n_units=n_units,
                                tensor=scaledStrikeTensor,
                                isTraining=IsTraining,
                                name="Hidden1S")

    hidden1M = monotonicDugasLayer(n_units=n_units,
                                   tensor=scaleTensorMaturity,
                                   isTraining=IsTraining,
                                   name="Hidden1M")

    hidden1 = hidden1S * hidden1M

    # Second layer and output layer
    out = convexDugasOutputLayer(tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Output")
    # Local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, 
                                                       tensorDict,
                                                       weighting,
                                                       hyperparameters)

    return out, [out, dupireVol, theta, hK, dupireVar], [], evalAndFormatDupireResult

############################################################################# Dense Unconstrained architecture


# Unconstrained dense layer
def unconstrainedLayer(n_units, tensor, isTraining, name, activation=K.softplus):
    with tf.name_scope(name):
        if name not in layerFactory :
            layerFactory[name] = tf.keras.layers.Dense(n_units,
                                                       activation=activation,
                                                       kernel_initializer=tf.keras.initializers.he_normal())
        #layer = tf.layers.dense(tensor,
        #                        units=n_units,
        #                        activation=activation,
        #                        kernel_initializer=tf.keras.initializers.he_normal())
        return layerFactory[name](tensor)


# Factory for unconstrained network
def NNArchitectureUnconstrained(n_units,
                                tensorDict,
                                hyperparameters,
                                IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    inputLayer = tf.concat([scaledStrikeTensor, scaledMaturityTensor], axis=-1)

    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")

    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Output layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden2,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)

    return out, [out], [], evalAndFormatResult
    
def NNArchitectureUnconstrainedDupire(n_units,
                                      tensorDict,
                                      hyperparameters,
                                      IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    inputLayer = tf.concat([scaledStrikeTensor, scaledMaturityTensor], axis=-1)

    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")
    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Ouput layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden2,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)
    # Local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, 
                                                       tensorDict,
                                                       weighting,
                                                       hyperparameters)

    return out, [out, dupireVol, theta, hK, dupireVar], [], evalAndFormatDupireResult

############################################################################# Dense Soft constraint architecture


def NNArchitectureVanillaSoftDupire(n_units,
                                    tensorDict,
                                    hyperparameters,
                                    IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    inputLayer = tf.concat([scaledStrikeTensor, scaledMaturityTensor], axis=-1)
    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")
    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Output layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden2,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)
    # Local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, 
                                                       tensorDict,
                                                       weighting,
                                                       hyperparameters)
    # Soft constraints for no arbitrage
    
    penaltyList = arbitragePenaltiesPrice(theta, 
                                          hK,
                                          tensorDict,
                                          weighting,
                                          hyperparameters)  
    
    return out, [out, dupireVol, theta, hK, dupireVar], penaltyList, evalAndFormatDupireResult


############################################################################# Hard constraint Splitted Architecture

# Tools functions for hard constrained neural architecture

def convexLayerHard(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=n_units,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.softplus(layer), layer


def monotonicLayerHard(n_units, tensor, isTraining, name):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor,
                                units=n_units,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=tf.keras.initializers.glorot_normal())

        return tf.nn.sigmoid(layer), layer


def convexOutputLayerHard(n_units, tensor, isTraining, name, isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=2 * n_units,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=tf.keras.initializers.glorot_normal(),
                                activation='softplus')

        layer = tf.layers.dense(layer,
                                units=1,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=positiveKernelInitializer,
                                activation='softplus')

        return layer, layer


def convexLayerHybridHard(n_units,
                          tensor,
                          isTraining,
                          name,
                          activationFunction2=Act.softplus,
                          activationFunction1=Act.exponential,
                          isNonDecreasing=True):
    with tf.name_scope(name):
        layer = tf.layers.dense(tensor if isNonDecreasing else (- tensor),
                                units=n_units,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer=positiveKernelInitializer)
        l1, l2 = tf.split(layer, 2, 1)
        output = tf.concat([activationFunction1(l1), activationFunction2(l2)], axis=-1)
        return output, layer


def sigmoidGradientHard(inputTensor):
    return tf.nn.sigmoid(inputTensor) * (1 - tf.nn.sigmoid(inputTensor))


def sigmoidHessianHard(inputTensor):
    return (tf.square(1 - tf.nn.sigmoid(inputTensor)) -
            tf.nn.sigmoid(inputTensor) * (1 - tf.nn.sigmoid(inputTensor)))


def NNArchitectureHardConstrainedDupire(n_units,
                                        tensorDict,
                                        hyperparameters,
                                        IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    # Splitted First layer
    hidden1S, layer1S = convexLayerHard(n_units=n_units,
                                        tensor=scaledStrikeTensor,
                                        isTraining=IsTraining,
                                        name="Hidden1S")

    hidden1M, layer1M = monotonicLayerHard(n_units=n_units,
                                           tensor=scaledMaturityTensor,
                                           isTraining=IsTraining,
                                           name="Hidden1M")

    hidden1 = tf.concat([hidden1S, hidden1M], axis=-1)

    # Second layer and output layer
    out, layer = convexOutputLayerHard(n_units=n_units,
                                       tensor=hidden1,
                                       isTraining=IsTraining,
                                       name="Output")
    # Local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, 
                                                       tensorDict,
                                                       weighting,
                                                       hyperparameters)

    return out, [out, dupireVol, theta, hK, dupireVar], [], evalAndFormatDupireResult

############################################################################# Soft constraint Splitted Architecture


def NNArchitectureConstrainedRawDupire(n_units,
                                       tensorDict,
                                       hyperparameters,
                                       IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledStrikeTensor = tensorDict["Strike"]
    scaleTensor = tensorDict["scaleTensorStrike"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    StrikeMinTensor = tensorDict["StrikeMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    # Splitted First splitted layer
    hidden1S = convexLayer(n_units=n_units,
                           tensor=scaledStrikeTensor,
                           isTraining=IsTraining,
                           name="Hidden1S")

    hidden1M = monotonicLayer(n_units=n_units,
                              tensor=scaledMaturityTensor,
                              isTraining=IsTraining,
                              name="Hidden1M")

    hidden1 = tf.concat([hidden1S, hidden1M], axis=-1)

    # Second hidden layer and output layer
    out = convexOutputLayer(n_units=n_units,
                            tensor=hidden1,
                            isTraining=IsTraining,
                            name="Output")

    # Compute local volatility
    dupireVol, theta, hK, dupireVar = rawDupireFormula(out, 
                                                       tensorDict,
                                                       weighting,
                                                       hyperparameters)
    # Soft constraints for no arbitrage
    
    penaltyList = arbitragePenaltiesPrice(theta, 
                                          hK,
                                          tensorDict,
                                          weighting,
                                          hyperparameters)   

    return out, [out, dupireVol, theta, hK, dupireVar], penaltyList, evalAndFormatDupireResult




























####################################################################################################################### Implied volatility Neural network API
#######################################################################################################################

def computeWeighting(batch, scaler):
    def secondMin(row):
        return row.sort_values().iloc[1]
    coordinates = dataSetConstruction.transformCustomMinMax(batch, scaler)[["logMoneyness", "logMaturity"]]
    distanceToClosestPoint = pd.DataFrame(scipy.spatial.distance_matrix(coordinates,
                                                                        coordinates), 
                                          index = coordinates.index,
                                          columns = coordinates.index).apply(secondMin)
    return distanceToClosestPoint

# Train neural network with a decreasing rule for learning rate
# NNFactory :  function creating the architecture
# dataSet : training data
# activateRegularization : boolean, if true add bound penalization to dupire variance
# hyperparameters : dictionnary containing various hyperparameters
# modelName : name under which tensorflow model is saved
def create_train_model_gatheral(NNFactory,
                                dataSet,
                                activateRegularization,
                                hyperparameters,
                                scaler,
                                modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]
    nbEpoch = hyperparameters["maxEpoch"]
    nbEpochFork = hyperparameters["nbEpochFork"]
    fixedLearningRate = (None if (("FixedLearningRate" in hyperparameters) and hyperparameters["FixedLearningRate"]) else hyperparameters["LearningRateStart"])
    patience = hyperparameters["Patience"]
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    holderExponent = (hyperparameters["HolderExponent"] if ("HolderExponent" in hyperparameters) else 2.0)

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]
    learningRateEpoch = 0
    finalLearningRate = hyperparameters["FinalLearningRate"]

    batch_size = hyperparameters["batchSize"]

    start = time.time()
    # Reset the graph
    resetTensorflow()
    # Placeholders for input and output data
    Moneyness = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    MoneynessPenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    yBid = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yBid')
    yAsk = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yAsk')
    
    weighting = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weighting')
    weightingPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weightingPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])


    # Get scaling for strike
    colMoneynessIndex = dataSet.columns.get_loc("logMoneyness")
    maxColFunction = scaler.data_max_[colMoneynessIndex]
    minColFunction = scaler.data_min_[colMoneynessIndex]
    scF = (maxColFunction - minColFunction) #0
    scaleTensor = tf.constant(scF, dtype=tf.float32) #1
    moneynessMinTensor = tf.constant(minColFunction, dtype=tf.float32)
    
    
    # Get scaling for maturity
    colMaturityIndex = dataSet.columns.get_loc("logMaturity") if useLogMaturity else dataSet.columns.get_loc("Maturity")
    maxColFunctionMat = scaler.data_max_[colMaturityIndex]
    minColFunctionMat = scaler.data_min_[colMaturityIndex] #0
    scFMat = (maxColFunctionMat - minColFunctionMat) #1
    scaleTensorMaturity = tf.constant(scFMat, dtype=tf.float32)
    maturityMinTensor = tf.constant(minColFunctionMat, dtype=tf.float32)
    
    tensorDict = {}
    tensorDict["Maturity"] = Maturity
    tensorDict["Moneyness"] = Moneyness
    tensorDict["scaleTensorMoneyness"] = scaleTensor
    tensorDict["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDict["moneynessMinTensor"] = moneynessMinTensor
    tensorDict["maturityMinTensor"] = maturityMinTensor
    tensorDict["lossWeighting"] = weighting
    
    tensorDictPenalization = {}
    tensorDictPenalization["Maturity"] = MaturityPenalization
    tensorDictPenalization["Moneyness"] = MoneynessPenalization
    tensorDictPenalization["scaleTensorMoneyness"] = scaleTensor
    tensorDictPenalization["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDictPenalization["moneynessMinTensor"] = moneynessMinTensor
    tensorDictPenalization["maturityMinTensor"] = maturityMinTensor
    tensorDictPenalization["lossWeighting"] = weightingPenalization
        
    #Grid on which is applied Penalization [0, 2 * maxMaturity] and [minMoneyness, 2 * maxMoneyness]
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                2.0 * scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    
    #k = np.linspace((np.log(0.5) + minColFunction) / scF,
    #                (np.log(2.0) + maxColFunction) / scF,
    #                num=50)
    
    k = np.linspace(np.log(0.5) / scF,
                    (np.log(2.0) + maxColFunction - minColFunction) / scF,
                    num=50)
    
    if useLogMaturity :
        #t = np.linspace(minColFunctionMat, np.log(2) + maxColFunctionMat, num=100)
        t = np.linspace(0, (np.log(4) + maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    else :
        #t = np.linspace(0, 4, num=100)
        #t = np.linspace(minColFunctionMat, 2 * maxColFunctionMat, num=100)
        t = np.linspace(0, (4 * maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    with tf.device("/gpu:0"):
        if activateRegularization:  # Add pseudo local volatility regularisation
            vol_pred_tensor, TensorList, penalizationList, formattingFunction = addDupireRegularisation(
                *NNFactory(hidden_nodes,
                           tensorDict,
                           hyperparameters),
                weighting,
                hyperparameters)
            vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = addDupireRegularisation(
                *NNFactory(hidden_nodes,
                           tensorDictPenalization,
                           hyperparameters),
                weightingPenalization,
                hyperparameters)
        else:
            vol_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                          tensorDict,
                                                                                          hyperparameters)
            vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                              tensorDictPenalization,
                                                                                              hyperparameters)

        vol_pred_tensor_sc = vol_pred_tensor
        unscaledMaturityTensor = (Maturity * scaleTensorMaturity + maturityMinTensor)
        maturityOriginal = tf.exp(unscaledMaturityTensor) if useLogMaturity else unscaledMaturityTensor
        TensorList[0] = tf.square(vol_pred_tensor_sc) * maturityOriginal

        # Define a loss function
        pointwiseError = tf.pow(tf.reduce_mean(tf.pow((vol_pred_tensor_sc - y)/y , holderExponent) * weighting), 1.0 / holderExponent)
        #pointwiseError = tf.pow(tf.reduce_mean(tf.pow((vol_pred_tensor_sc - y) , holderExponent) ), 1.0 / holderExponent) * tf.reduce_mean(weighting)
        errors = tf.add_n([pointwiseError] + penalizationList1) #tf.add_n([pointwiseError] + penalizationList)
        loss = tf.log(tf.reduce_mean(errors))
        
        forkPenalization = hyperparameters["lambdaFork"] * tf.reduce_mean(tf.square(tf.nn.relu(yBid - vol_pred_tensor_sc) / yBid) + tf.square(tf.nn.relu(vol_pred_tensor_sc - yAsk) / yAsk))
        errorFork = tf.add_n([pointwiseError, forkPenalization] + penalizationList1)
        lossFork = tf.log(tf.reduce_mean(errorFork))

        # Define a train operation to minimize the loss
        lr = learningRate

        optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
        train = optimizer.minimize(loss)
        trainFork = optimizer.minimize(lossFork)

        # Initialize variables and run session
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)
    n = dataSet.shape[0]
    loss_serie = []
    
    #Create weighting to focus learning on isolated points


    def createFeedDict(batch):
        batchSize = batch.shape[0]
        scaledBatch = dataSetConstruction.transformCustomMinMax(batch, scaler)
        weighthingBatch = computeWeighting(batch, scaler)
        feedDict = {Moneyness: scaledBatch["logMoneyness"].values.reshape(batchSize, 1),  
                    Maturity:  scaledBatch["logMaturity"].values.reshape(batchSize, 1) if useLogMaturity else scaledBatch["Maturity"].values.reshape(batchSize, 1),  
                    y: batch[impliedVolColumn].values.reshape(batchSize, 1),
                    yBid : batch["ImpVolBid"].values.reshape(batchSize, 1),
                    yAsk : batch["ImpVolAsk"].values.reshape(batchSize, 1),
                    MoneynessPenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization, 1),
                    learningRateTensor: learningRate,
                    weighting: np.expand_dims(weighthingBatch.values,1),#np.ones_like(batch["logMoneyness"].values.reshape(batchSize, 1)),
                    weightingPenalization : np.mean(weighthingBatch) * np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict

    # Learning rate is divided by 10 if no imporvement is observed for training loss after "patience" epochs
    def updateLearningRate(iterNumber, lr, lrEpoch):
        if not activateLearningDecrease:
            print("Constant learning rate, stop training")
            return False, lr, lrEpoch
        if learningRate > finalLearningRate:
            lr *= 0.1
            lrEpoch = iterNumber
            saver.restore(sess, modelFolder + modelName)
            print("Iteration : ", lrEpoch, "new learning rate : ", lr)
        else:
            print("Last Iteration : ", lrEpoch, "final learning rate : ", lr)
            return False, lr, lrEpoch
        return True, lr, lrEpoch

    epochFeedDict = createFeedDict(dataSet)

    def evalBestModel():
        if not activateLearningDecrease:
            print("Learning rate : ", learningRate, " final loss : ", min(loss_serie))
        currentBestLoss = sess.run(loss, feed_dict=epochFeedDict)
        currentBestPenalizations = sess.run([pointwiseError, forkPenalization, penalizationList], feed_dict=epochFeedDict)
        currentBestPenalizations1 = sess.run([penalizationList1], feed_dict=epochFeedDict)
        print("Best loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, len(loss_serie), currentBestLoss))
        print("Best Penalization : ", currentBestPenalizations)
        print("Best Penalization (Refined Grid) : ", currentBestPenalizations1)
        return
    
    print("Training w.r.t. implied vol RMSE and arbitrage constraints")
    for i in range(nbEpoch):
        miniBatchList = [dataSet]
        #penalizationResult = sess.run(penalizationList, feed_dict=epochFeedDict)
        #lossResult = sess.run(pointwiseError, feed_dict=epochFeedDict)

        # miniBatchList = selectMiniBatchWithoutReplacement(dataSet, batch_size)
        if len(miniBatchList) > 1 :
            for k in range(len(miniBatchList)):
                batchFeedDict = createFeedDict(miniBatchList[k])
                sess.run(train, feed_dict=batchFeedDict)
        else : 
            sess.run(train, feed_dict=epochFeedDict)
        loss_serie.append(sess.run(loss, feed_dict=epochFeedDict))

        if (len(loss_serie) < 2) or (loss_serie[-1] <= min(loss_serie)):
            # Save model as model is improved
            saver.save(sess, modelFolder + modelName)
        if (np.isnan(loss_serie[-1]) or  # Unstable model
                ((i - learningRateEpoch >= patience) and (min(loss_serie[-patience:]) > min(
                    loss_serie)))):  # No improvement for training loss during the latest 100 iterations
            continueTraining, learningRate, learningRateEpoch = updateLearningRate(i, learningRate, learningRateEpoch)
            if continueTraining:
                evalBestModel()
            else:
                break
    saver.restore(sess, modelFolder + modelName)

    print("Training w.r.t. implied vol RMSE, arbitrage constraints and bid-ask fork violation")
    learningRate = hyperparameters["LearningRateStart"]
    loss_serie_fork = []
    learningRateEpoch = 0
    for i in range(nbEpochFork):
        miniBatchList = [dataSet]
        #penalizationResult = sess.run(penalizationList, feed_dict=epochFeedDict)
        #lossResult = sess.run(pointwiseError, feed_dict=epochFeedDict)

        # miniBatchList = selectMiniBatchWithoutReplacement(dataSet, batch_size)
        for k in range(len(miniBatchList)):
            batchFeedDict = createFeedDict(miniBatchList[k])
            sess.run(trainFork, feed_dict=batchFeedDict)

        loss_serie_fork.append(sess.run(lossFork, feed_dict=epochFeedDict))

        if (len(loss_serie_fork) > 1) and (loss_serie_fork[-1] <= min(loss_serie_fork)):
            # Save model as error is improved
            saver.save(sess, modelName)
        if (np.isnan(loss_serie_fork[-1]) or  # Unstable training
                ((i - learningRateEpoch >= patience) and (min(loss_serie[-patience:]) > min(
                    loss_serie)))):  # No improvement for training loss during the latest 100 iterations
            continueTraining, learningRate, learningRateEpoch = updateLearningRate(i, learningRate, learningRateEpoch)
            if continueTraining:
                evalBestModel()
            else:
                break
    
    saver.restore(sess, modelFolder + modelName)
    evalBestModel()
    
    #Count arbitrage violations on refined grid (grid on which are applied penalizations)
    print("Refined Grid evaluation :")
    refinedEpochDict = {Moneyness: np.expand_dims(kPenalization, 1),
                        Maturity:  np.expand_dims(tPenalization, 1),  
                        y: np.ones_like(np.expand_dims(kPenalization, 1)),
                        yBid : np.ones_like(np.expand_dims(kPenalization, 1)),
                        yAsk : np.ones_like(np.expand_dims(kPenalization, 1)),
                        MoneynessPenalization : np.expand_dims(kPenalization, 1),
                        MaturityPenalization : np.expand_dims(tPenalization, 1),
                        learningRateTensor: learningRate,
                        weighting: np.ones_like(np.expand_dims(kPenalization, 1)),
                        weightingPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
    evalRefinedList = sess.run(TensorList, feed_dict=refinedEpochDict)
    emptyDf = pd.DataFrame(np.ones((kPenalization.size, dataSet.shape[1])), 
                           index = pd.MultiIndex.from_tuples( list(zip(kPenalization, tPenalization)) ),
                           columns = dataSet.columns)
    formattingFunction(*evalRefinedList, loss_serie, emptyDf, scaler)
    
    print("Dataset Grid evaluation :")
    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()
    end = time.time()
    print("Training Time : ", end - start)

    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    lossEpochSerie = pd.Series(loss_serie + loss_serie_fork)
    lossEpochSerie.to_csv(modelFolder + "loss" + modelName + ".csv", header = True)

    return formattingFunction(*evalList, loss_serie, dataSet, scaler)


# Evaluate neural network without training, it restores parameters obtained from a pretrained model
# NNFactory :  function creating the neural architecture
# dataSet : dataset on which neural network is evaluated
# activateRegularization : boolean, if true add bound penalization for dupire variance
# hyperparameters : dictionnary containing various hyperparameters
# modelName : name of tensorflow model to restore
def create_eval_model_gatheral(NNFactory,
                               dataSet,
                               activateRegularization,
                               hyperparameters,
                               scaler,
                               modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]
    nbEpoch = hyperparameters["maxEpoch"]
    fixedLearningRate = (None if (("FixedLearningRate" in hyperparameters) and hyperparameters["FixedLearningRate"]) else hyperparameters["LearningRateStart"])
    patience = hyperparameters["Patience"]
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    holderExponent = (hyperparameters["HolderExponent"] if ("HolderExponent" in hyperparameters) else 2.0)

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]
    learningRateEpoch = 0
    finalLearningRate = hyperparameters["FinalLearningRate"]

    batch_size = hyperparameters["batchSize"]

    start = time.time()
    # Reset the graph
    resetTensorflow()

    # Placeholders for input and output data
    Moneyness = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    MoneynessPenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    yBid = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yBid')
    yAsk = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yAsk')
    weighting = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weighting')
    weightingPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weightingPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])


    # Get scaling for strike
    colMoneynessIndex = dataSet.columns.get_loc("logMoneyness")
    maxColFunction = scaler.data_max_[colMoneynessIndex]
    minColFunction = scaler.data_min_[colMoneynessIndex]
    scF = (maxColFunction - minColFunction) #0
    scaleTensor = tf.constant(scF, dtype=tf.float32) #1
    moneynessMinTensor = tf.constant(minColFunction, dtype=tf.float32)
    
    
    # Get scaling for maturity
    colMaturityIndex = dataSet.columns.get_loc("logMaturity") if useLogMaturity else dataSet.columns.get_loc("Maturity")
    maxColFunctionMat = scaler.data_max_[colMaturityIndex]
    minColFunctionMat = scaler.data_min_[colMaturityIndex] #0
    scFMat = (maxColFunctionMat - minColFunctionMat) #1
    scaleTensorMaturity = tf.constant(scFMat, dtype=tf.float32)
    maturityMinTensor = tf.constant(minColFunctionMat, dtype=tf.float32)
    
    tensorDict = {}
    tensorDict["Maturity"] = Maturity
    tensorDict["Moneyness"] = Moneyness
    tensorDict["scaleTensorMoneyness"] = scaleTensor
    tensorDict["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDict["moneynessMinTensor"] = moneynessMinTensor
    tensorDict["maturityMinTensor"] = maturityMinTensor
    tensorDict["lossWeighting"] = weighting
    
    tensorDictPenalization = {}
    tensorDictPenalization["Maturity"] = MaturityPenalization
    tensorDictPenalization["Moneyness"] = MoneynessPenalization
    tensorDictPenalization["scaleTensorMoneyness"] = scaleTensor
    tensorDictPenalization["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDictPenalization["moneynessMinTensor"] = moneynessMinTensor
    tensorDictPenalization["maturityMinTensor"] = maturityMinTensor
    tensorDictPenalization["lossWeighting"] = weightingPenalization
        
    #Grid on which is applied Penalization [0, 2 * maxMaturity] and [minMoneyness, 2 * maxMoneyness]
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                2.0 * scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    
    #k = np.linspace((np.log(0.5) + minColFunction) / scF,
    #                (np.log(2.0) + maxColFunction) / scF,
    #                num=50)
    
    k = np.linspace(np.log(0.5) / scF,
                    (np.log(2.0) + maxColFunction - minColFunction) / scF,
                    num=50)
    
    if useLogMaturity :
        #t = np.linspace(minColFunctionMat, np.log(2) + maxColFunctionMat, num=100)
        t = np.linspace(0, (np.log(4) + maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    else :
        #t = np.linspace(0, 4, num=100)
        #t = np.linspace(minColFunctionMat, 2 * maxColFunctionMat, num=100)
        t = np.linspace(0, (4 * maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    if activateRegularization:  # Add pseudo local volatility regularisation
        vol_pred_tensor, TensorList, penalizationList, formattingFunction = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       tensorDict,
                       hyperparameters),
            weighting,
            hyperparameters)
        vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = addDupireRegularisation(
            *NNFactory(hidden_nodes,
                       tensorDictPenalization,
                       hyperparameters),
            weightingPenalization,
            hyperparameters)
    else:
        vol_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                      tensorDict,
                                                                                      hyperparameters)
        vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                          tensorDictPenalization,
                                                                                          hyperparameters)

    vol_pred_tensor_sc = vol_pred_tensor
    unscaledMaturityTensor = (Maturity * scaleTensorMaturity + maturityMinTensor)
    maturityOriginal = tf.exp(unscaledMaturityTensor) if useLogMaturity else  unscaledMaturityTensor
    TensorList[0] = tf.square(vol_pred_tensor_sc) * maturityOriginal

    # Define a loss function
    pointwiseError = tf.pow(tf.reduce_mean(tf.pow((vol_pred_tensor_sc - y)/y , holderExponent) * weighting), 1.0 / holderExponent)
    #pointwiseError = tf.pow(tf.reduce_mean(tf.pow((vol_pred_tensor_sc - y) , holderExponent) ), 1.0 / holderExponent) * tf.reduce_mean(weighting)
    errors = tf.add_n([pointwiseError] + penalizationList1) #tf.add_n([pointwiseError] + penalizationList)
    loss = tf.log(tf.reduce_mean(errors))
    
    forkPenalization = hyperparameters["lambdaFork"] * tf.reduce_mean(tf.square(tf.nn.relu(yBid - vol_pred_tensor_sc) / yBid) + tf.square(tf.nn.relu(vol_pred_tensor_sc - yAsk) / yAsk))
    errorFork = tf.add_n([pointwiseError, forkPenalization] + penalizationList1)
    lossFork = tf.log(tf.reduce_mean(errorFork))

    # Define a train operation to minimize the loss
    lr = learningRate

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
    train = optimizer.minimize(loss)
    trainFork = optimizer.minimize(lossFork)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n = dataSet.shape[0]
    loss_serie = []

    #Create weighting to focus learning on isolated points


    def createFeedDict(batch):
        batchSize = batch.shape[0]
        scaledBatch = dataSetConstruction.transformCustomMinMax(batch, scaler)
        weighthingBatch = computeWeighting(batch, scaler)
        feedDict = {Moneyness: scaledBatch["logMoneyness"].values.reshape(batchSize, 1),  
                    Maturity:  scaledBatch["logMaturity"].values.reshape(batchSize, 1) if useLogMaturity else scaledBatch["Maturity"].values.reshape(batchSize, 1),  
                    y: batch[impliedVolColumn].values.reshape(batchSize, 1),
                    yBid : batch["ImpVolBid"].values.reshape(batchSize, 1),
                    yAsk : batch["ImpVolAsk"].values.reshape(batchSize, 1),
                    MoneynessPenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization, 1),
                    learningRateTensor: learningRate,
                    weighting: np.expand_dims(weighthingBatch.values,1),#np.ones_like(batch["logMoneyness"].values.reshape(batchSize, 1)),
                    weightingPenalization : np.mean(weighthingBatch) * np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict

    epochFeedDict = createFeedDict(dataSet)

    def evalBestModel():
        if not activateLearningDecrease:
            print("Learning rate : ", learningRate, " final loss : ", min(loss_serie))
        currentBestLoss = sess.run(loss, feed_dict=epochFeedDict)
        currentBestPenalizations = sess.run([pointwiseError, forkPenalization, penalizationList], feed_dict=epochFeedDict)
        currentBestPenalizations1 = sess.run([penalizationList1], feed_dict=epochFeedDict)
        print("Best loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, len(loss_serie), currentBestLoss))
        print("Best Penalization : ", currentBestPenalizations)
        print("Best Penalization (Refined Grid): ", currentBestPenalizations1)
        return

    saver.restore(sess, modelFolder + modelName)

    evalBestModel()
    
    #Count arbitrage violations on refined grid (grid on which are applied penalizations)
    print("Refined Grid evaluation :")
    refinedEpochDict = {Moneyness: np.expand_dims(kPenalization, 1),
                        Maturity:  np.expand_dims(tPenalization, 1),  
                        y: np.ones_like(np.expand_dims(kPenalization, 1)),
                        MoneynessPenalization : np.expand_dims(kPenalization, 1),
                        yBid : np.ones_like(np.expand_dims(kPenalization, 1)),
                        yAsk : np.ones_like(np.expand_dims(kPenalization, 1)),
                        MaturityPenalization : np.expand_dims(tPenalization, 1),
                        learningRateTensor: learningRate,
                        weighting: np.ones_like(np.expand_dims(kPenalization, 1)),
                        weightingPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
    evalRefinedList = sess.run(TensorList, feed_dict=refinedEpochDict)
    emptyDf = pd.DataFrame(np.ones((kPenalization.size, dataSet.shape[1])), 
                           index = pd.MultiIndex.from_tuples( list(zip(kPenalization, tPenalization)) ),
                           columns = dataSet.columns)
    formattingFunction(*evalRefinedList, [0], emptyDf, scaler)
    
    print("Dataset Grid evaluation :")
    evalList = sess.run(TensorList, feed_dict=epochFeedDict)
    
    sess.close()

    return formattingFunction(*evalList, [0], dataSet, scaler)


def evalVolLocaleGatheral(NNFactory,
                          strikes,
                          maturities,
                          dataSet,
                          hyperparameters,
                          scaler,
                          bootstrap,
                          S0,
                          modelName="bestModel"):
    hidden_nodes = hyperparameters["nbUnits"]
    nbEpoch = hyperparameters["maxEpoch"]
    fixedLearningRate = (None if (("FixedLearningRate" in hyperparameters) and hyperparameters["FixedLearningRate"]) else hyperparameters["LearningRateStart"])
    patience = hyperparameters["Patience"]
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    holderExponent = (hyperparameters["HolderExponent"] if ("HolderExponent" in hyperparameters) else 2.0)

    # Go through num_iters iterations (ignoring mini-batching)
    activateLearningDecrease = (~ hyperparameters["FixedLearningRate"])
    learningRate = hyperparameters["LearningRateStart"]
    learningRateEpoch = 0
    finalLearningRate = hyperparameters["FinalLearningRate"]

    batch_size = hyperparameters["batchSize"]

    start = time.time()
    # Reset the graph
    resetTensorflow()

    # Placeholders for input and output data
    Moneyness = tf.placeholder(tf.float32, [None, 1])
    Maturity = tf.placeholder(tf.float32, [None, 1])
    MoneynessPenalization = tf.placeholder(tf.float32, [None, 1])
    MaturityPenalization = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')
    yBid = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yBid')
    yAsk = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='yAsk')
    weighting = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weighting')
    weightingPenalization = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='weightingPenalization')
    learningRateTensor = tf.placeholder(tf.float32, [])


    # Get scaling for strike
    colMoneynessIndex = dataSet.columns.get_loc("logMoneyness")
    maxColFunction = scaler.data_max_[colMoneynessIndex]
    minColFunction = scaler.data_min_[colMoneynessIndex]
    scF = (maxColFunction - minColFunction) #0
    scaleTensor = tf.constant(scF, dtype=tf.float32) #1
    moneynessMinTensor = tf.constant(minColFunction, dtype=tf.float32)
    
    
    # Get scaling for maturity
    colMaturityIndex = dataSet.columns.get_loc("logMaturity") if useLogMaturity else dataSet.columns.get_loc("Maturity")
    maxColFunctionMat = scaler.data_max_[colMaturityIndex]
    minColFunctionMat = scaler.data_min_[colMaturityIndex] #0
    scFMat = (maxColFunctionMat - minColFunctionMat) #1
    scaleTensorMaturity = tf.constant(scFMat, dtype=tf.float32)
    maturityMinTensor = tf.constant(minColFunctionMat, dtype=tf.float32)
    
    tensorDict = {}
    tensorDict["Maturity"] = Maturity
    tensorDict["Moneyness"] = Moneyness
    tensorDict["scaleTensorMoneyness"] = scaleTensor
    tensorDict["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDict["moneynessMinTensor"] = moneynessMinTensor
    tensorDict["maturityMinTensor"] = maturityMinTensor
    tensorDict["lossWeighting"] = weighting
    
    tensorDictPenalization = {}
    tensorDictPenalization["Maturity"] = MaturityPenalization
    tensorDictPenalization["Moneyness"] = MoneynessPenalization
    tensorDictPenalization["scaleTensorMoneyness"] = scaleTensor
    tensorDictPenalization["scaleTensorMaturity"] = scaleTensorMaturity
    tensorDictPenalization["moneynessMinTensor"] = moneynessMinTensor
    tensorDictPenalization["maturityMinTensor"] = maturityMinTensor
    tensorDictPenalization["lossWeighting"] = weightingPenalization
        
    #Grid on which is applied Penalization [0, 2 * maxMaturity] and [minMoneyness, 2 * maxMoneyness]
    #k = np.linspace(scaler.data_min_[dataSet.columns.get_loc("logMoneyness")],
    #                2.0 * scaler.data_max_[dataSet.columns.get_loc("logMoneyness")],
    #                num=50)
    
    #k = np.linspace((np.log(0.5) + minColFunction) / scF,
    #                (np.log(2.0) + maxColFunction) / scF,
    #                num=50)
    
    k = np.linspace(np.log(0.5) / scF,
                    (np.log(2.0) + maxColFunction - minColFunction) / scF,
                    num=50)
    
    if useLogMaturity :
        #t = np.linspace(minColFunctionMat, np.log(2) + maxColFunctionMat, num=100)
        t = np.linspace((np.log(0.00001) - minColFunctionMat) / scFMat, 
                        (np.log(4) + maxColFunctionMat - minColFunctionMat) / scFMat, 
                        num=100)
    else :
        #t = np.linspace(0, 4, num=100)
        #t = np.linspace(minColFunctionMat, 2 * maxColFunctionMat, num=100)
        t = np.linspace(0, (4 * maxColFunctionMat - minColFunctionMat) / scFMat, num=100)
    
    penalizationGrid = np.meshgrid(k, t)
    tPenalization = np.ravel(penalizationGrid[1])
    kPenalization = np.ravel(penalizationGrid[0])

    price_pred_tensor = None
    TensorList = None
    penalizationList = None
    formattingFunction = None
    vol_pred_tensor, TensorList, penalizationList, formattingFunction = NNFactory(hidden_nodes,
                                                                                  tensorDict,
                                                                                  hyperparameters)
    vol_pred_tensor1, TensorList1, penalizationList1, formattingFunction1 = NNFactory(hidden_nodes,
                                                                                      tensorDictPenalization,
                                                                                      hyperparameters)

    vol_pred_tensor_sc = vol_pred_tensor
    unscaledMaturityTensor = (Maturity * scaleTensorMaturity + maturityMinTensor)
    maturityOriginal = tf.exp(unscaledMaturityTensor) if useLogMaturity else  unscaledMaturityTensor
    TensorList[0] = tf.square(vol_pred_tensor_sc) * maturityOriginal

    # Define a loss function
    pointwiseError = tf.pow(tf.reduce_mean(tf.pow((vol_pred_tensor_sc - y)/y , holderExponent) * weighting), 1.0 / holderExponent)
    #pointwiseError = tf.pow(tf.reduce_mean(tf.pow((vol_pred_tensor_sc - y) , holderExponent) ), 1.0 / holderExponent) * tf.reduce_mean(weighting)
    errors = tf.add_n([pointwiseError] + penalizationList1)
    loss = tf.log(tf.reduce_mean(errors))
    
    forkPenalization = hyperparameters["lambdaFork"] * tf.reduce_mean(tf.square(tf.nn.relu(yBid - vol_pred_tensor_sc) / yBid) + tf.square(tf.nn.relu(vol_pred_tensor_sc - yAsk) / yAsk))
    errorFork = tf.add_n([pointwiseError, forkPenalization] + penalizationList1)
    lossFork = tf.log(tf.reduce_mean(errorFork))

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRateTensor)
    train = optimizer.minimize(loss)
    trainFork = optimizer.minimize(lossFork)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    n = strikes.shape[0]
    changedVar = bootstrap.changeOfVariable(strikes, maturities)

    moneyness = np.log(changedVar[0] / S0)
    scaledMoneyness = (moneyness - minColFunction) / scF

    def createFeedDict(m, t):
        batchSize = m.shape[0]
        feedDict = {Moneyness: np.reshape(m, (batchSize, 1)),
                    Maturity: np.reshape(t, (batchSize, 1)),
                    MoneynessPenalization : np.expand_dims(kPenalization, 1),
                    MaturityPenalization : np.expand_dims(tPenalization,1),
                    weighting: np.ones((batchSize, 1)),
                    weightingPenalization : np.ones_like(np.expand_dims(kPenalization, 1))}
        return feedDict
    
    scaledMaturities = ((np.log(maturities) if useLogMaturity else maturities) - minColFunctionMat) / scFMat
    epochFeedDict = createFeedDict(scaledMoneyness, scaledMaturities)

    saver.restore(sess, modelFolder + modelName)

    evalList = sess.run(TensorList, feed_dict=epochFeedDict)

    sess.close()

    return pd.Series(evalList[1].flatten(),
                     index=pd.MultiIndex.from_arrays([strikes, maturities], names=('Strike', 'Maturity')))

################################################################################################################ Local volatility function
# Dupire formula from exact derivative computation
def dupireFormulaGatheral(HessianMoneyness,
                          GradMoneyness,
                          GradMaturity,
                          totalVariance,
                          ScaledMoneyness,
                          scaleTensor,
                          MoneynessMinTensor,
                          IsTraining=True):
    twoConstant = tf.constant(2.0)
    oneConstant = tf.constant(1.0)
    quarterConstant = tf.constant(0.25)
    halfConstant = tf.constant(0.5)

    moneyness = ScaledMoneyness * scaleTensor + MoneynessMinTensor

    dT = GradMaturity

    dMoneyness = GradMoneyness / scaleTensor
    dMoneynessFactor = (moneyness / totalVariance)
    dMoneynessSquaredFactor = quarterConstant * (
                -quarterConstant - oneConstant / totalVariance + tf.square(dMoneynessFactor))

    gMoneyness = HessianMoneyness / tf.square(scaleTensor)
    gMoneynessFactor = halfConstant
    denominator = oneConstant - dMoneynessFactor * (dMoneyness) + dMoneynessSquaredFactor * tf.square(
        dMoneyness) + gMoneynessFactor * gMoneyness

    gatheralVar = dT / denominator
    # Initial weights of neural network can be random which lead to negative dupireVar
    gatheralVolTensor = tf.sqrt(gatheralVar)
    return gatheralVolTensor, gatheralVar, gatheralDenominator

#Dupire formula with derivative obtained from native tensorflow algorithmic differentiation
def rawDupireFormulaGatheral(totalVarianceTensor,
                             tensorDict,
                             IsTraining=True,
                             useLogMaturity = False):
  scaledMaturityTensor = tensorDict["Maturity"] 
  scaledMoneynessTensor = tensorDict["Moneyness"]
  scaleTensor = tensorDict["scaleTensorMoneyness"] 
  scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
  moneynessMinTensor = tensorDict["moneynessMinTensor"] 
  maturityMinTensor = tensorDict["maturityMinTensor"]
  weighting = tensorDict["lossWeighting"]
    
  batchSize = tf.shape(scaledMoneynessTensor)[0]
  twoConstant = tf.constant(2.0)
  oneConstant = tf.constant(1.0)
  quarterConstant = tf.constant(0.25)
  halfConstant = tf.constant(0.5)

  moneyness = scaledMoneynessTensor * scaleTensor + moneynessMinTensor
  maturity = scaledMaturityTensor * scaleTensorMaturity + maturityMinTensor

  dMoneyness = tf.reshape(tf.gradients(totalVarianceTensor, scaledMoneynessTensor, name="dK")[0], shape=[batchSize,-1]) / scaleTensor
  dMoneynessFactor = (moneyness/totalVarianceTensor)
  dMoneynessSquaredFactor = quarterConstant * (-quarterConstant - oneConstant/totalVarianceTensor + tf.square(dMoneynessFactor))

  gMoneyness = tf.reshape(tf.gradients(dMoneyness, scaledMoneynessTensor, name="hK")[0], shape=[batchSize,-1]) / scaleTensor
  gMoneynessFactor = halfConstant


  gatheralDenominator = oneConstant - dMoneynessFactor * (dMoneyness) + dMoneynessSquaredFactor * tf.square(dMoneyness) + gMoneynessFactor *  gMoneyness
  
  if useLogMaturity :
    dT = tf.reshape(tf.gradients(totalVarianceTensor,scaledMaturityTensor,name="dT")[0], shape=[batchSize,-1]) / scaleTensorMaturity / tf.exp(maturity)
  else :
    dT = tf.reshape(tf.gradients(totalVarianceTensor,scaledMaturityTensor,name="dT")[0], shape=[batchSize,-1]) / scaleTensorMaturity

  #Initial weights of neural network can be random which lead to negative dupireVar
  gatheralVar = dT / gatheralDenominator
  gatheralVol = tf.sqrt(gatheralVar)
  return  gatheralVol, dT, gMoneyness, gatheralVar, gatheralDenominator


################################################################################################################## Soft constraints penalties

# Soft constraints for strike convexity and strike/maturity monotonicity
def arbitragePenalties(dT, gatheralDenominator, weighting, hyperparameters):
    lambdas = tf.reduce_mean(weighting)
    lowerBoundTheta = tf.constant(hyperparameters["lowerBoundTheta"])
    lowerBoundGamma = tf.constant(hyperparameters["lowerBoundGamma"])
    calendar_penalty = lambdas * hyperparameters["lambdaSoft"] * tf.reduce_mean(tf.nn.relu(-dT + lowerBoundTheta))
    butterfly_penalty = lambdas * hyperparameters["lambdaGamma"] * tf.reduce_mean( tf.nn.relu(-gatheralDenominator + lowerBoundGamma) )

    return [calendar_penalty, butterfly_penalty]


################################################################################################################## Ackerer neural network
def NNArchitectureVanillaSoftGatheralAckerer(n_units,
                                             tensorDict,
                                             hyperparameters,
                                             IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledMoneynessTensor = tensorDict["Moneyness"]
    scaleTensor = tensorDict["scaleTensorMoneyness"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    moneynessMinTensor = tensorDict["moneynessMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    
    inputLayer = tf.concat([scaledMoneynessTensor, scaledMaturityTensor], axis=-1)
    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")
    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Third layer
    hidden3 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden2,
                                 isTraining=IsTraining,
                                 name="Hidden3")
    # Output layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden3,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)
    # Local volatility
    unscaledMaturity = (scaledMaturityTensor * scaleTensorMaturity + maturityMinTensor)
    gatheralVol, theta, hK, gatheralVar, gatheralDenominator = rawDupireFormulaGatheral(tf.square(out) * (tf.exp(unscaledMaturity) if useLogMaturity else unscaledMaturity),
                                                                                        tensorDict,
                                                                                        IsTraining=IsTraining, 
                                                                                        useLogMaturity = useLogMaturity)
    # Soft constraints for no arbitrage
    penalties = arbitragePenalties(theta, gatheralDenominator, weighting, hyperparameters)
    grad_penalty = penalties[0]
    hessian_penalty = penalties[1]

    return out, [out, gatheralVol, theta, gatheralDenominator, gatheralVar], [grad_penalty, hessian_penalty], evalAndFormatDupireResult


################################################################################################################## Dense soft constraints architecture
def NNArchitectureVanillaSoftGatheral(n_units,
                                      tensorDict,
                                      hyperparameters,
                                      IsTraining=True):
    scaledMaturityTensor = tensorDict["Maturity"] 
    scaledMoneynessTensor = tensorDict["Moneyness"]
    scaleTensor = tensorDict["scaleTensorMoneyness"] 
    scaleTensorMaturity = tensorDict["scaleTensorMaturity"] 
    moneynessMinTensor = tensorDict["moneynessMinTensor"] 
    maturityMinTensor = tensorDict["maturityMinTensor"]
    weighting = tensorDict["lossWeighting"]
    
    useLogMaturity = (hyperparameters["UseLogMaturity"] if ("UseLogMaturity" in hyperparameters) else False)
    inputLayer = tf.concat([scaledMoneynessTensor, scaledMaturityTensor], axis=-1)
    #inputLayer = tf.concat([scaledMoneynessTensor, tf.log(scaledMaturityTensor)], axis=-1)
    
    # First layer
    hidden1 = unconstrainedLayer(n_units=n_units,
                                 tensor=inputLayer,
                                 isTraining=IsTraining,
                                 name="Hidden1")
    # Second layer
    hidden2 = unconstrainedLayer(n_units=n_units,
                                 tensor=hidden1,
                                 isTraining=IsTraining,
                                 name="Hidden2")
    # Output layer
    out = unconstrainedLayer(n_units=1,
                             tensor=hidden2,
                             isTraining=IsTraining,
                             name="Output",
                             activation=None)
    # Local volatility
    unscaledMaturity = (scaledMaturityTensor * scaleTensorMaturity + maturityMinTensor)
    gatheralVol, theta, hK, gatheralVar, gatheralDenominator = rawDupireFormulaGatheral(tf.square(out) * (tf.exp(unscaledMaturity) if useLogMaturity else unscaledMaturity),
                                                                                        tensorDict,
                                                                                        IsTraining=IsTraining, 
                                                                                        useLogMaturity = useLogMaturity)
    # Soft constraints for no arbitrage
    penalties = arbitragePenalties(theta, gatheralDenominator, weighting, hyperparameters)
    grad_penalty = penalties[0]
    hessian_penalty = penalties[1]

    return out, [out, gatheralVol, theta, gatheralDenominator, gatheralVar], [grad_penalty, hessian_penalty], evalAndFormatDupireResult



####################################################################################### Automatic Hyperparameter selection

def selectHyperparameters(hyperparameters, parameterOfInterest, modelFactory,
                          modelName, activateDupireReg, scaledDataSet,
                          scaler,
                          trainedOnPrice = True,
                          logGrid=True):
    oldValue = hyperparameters[parameterOfInterest]
    gridValue = oldValue * (
        np.exp(np.log(10) * np.array([-2, -1, 0, 1, 2])) if logGrid else np.array([0.2, 0.5, 1, 2, 5]))

    oldNbEpochs = hyperparameters["maxEpoch"]
    hyperparameters["maxEpoch"] = int(oldNbEpochs / 10)
    trainLoss = {}
    arbitrageViolation = {}
    for v in gridValue:
        hyperparameters[parameterOfInterest] = int(v)
        if trainedOnPrice :
            pred, volLoc, theta, gammaK, loss = create_train_model(modelFactory,
                                                                   scaledDataSet,
                                                                   activateDupireReg,
                                                                   hyperparameters,
                                                                   scaler,
                                                                   modelName=modelName)
        else :
            pred, volLoc, theta, gammaK, loss = create_train_model_gatheral(modelFactory,
                                                                            scaledDataSet,
                                                                            activateDupireReg,
                                                                            hyperparameters,
                                                                            scaler,
                                                                            modelName=modelName)

        nbArbitrageViolation = np.sum((theta <= 0)) + np.sum((gammaK <= 0))
        trainLoss[v] = min(loss)
        arbitrageViolation[v] = nbArbitrageViolation
        print()
        print()

    hyperparameters["maxEpoch"] = oldNbEpochs
    hyperparameters[parameterOfInterest] = oldValue
    # Plot curves

    fig, ax1 = plt.subplots()
    if logGrid:
        plt.xscale('symlog')

    color = 'tab:red'
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(pd.Series(trainLoss), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Arbitrage violation', color=color)  # we already handled the x-label with ax1
    ax2.plot(pd.Series(arbitrageViolation), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    return


def selectHyperparametersRandom(hyperparameters,
                                parametersOfInterest,
                                modelFactory,
                                modelName,
                                activateDupireReg,
                                nbAttempts,
                                scaledDataSet,
                                scaler,
                                trainedOnPrice = True,
                                logGrid=True):
    oldValue = {}
    for k in parametersOfInterest:
        oldValue[k] = hyperparameters[k]

    gridValue = np.exp(np.log(10) * np.array([-2, -1, 0, 1, 2])) if logGrid else np.array([0.2, 0.5, 1, 2, 5])

    oldNbEpochs = hyperparameters["maxEpoch"]
    hyperparameters["maxEpoch"] = int(oldNbEpochs / 10)
    trainLoss = {}
    arbitrageViolation = {}
    nbTry = nbAttempts
    for v in range(nbTry):
        combination = np.random.randint(5, size=len(parametersOfInterest))
        for p in range(len(parametersOfInterest)):
            hyperparameters[parametersOfInterest[p]] = oldValue[parametersOfInterest[p]] * gridValue[
                int(combination[p])]
            print(parametersOfInterest[p], " : ", hyperparameters[parametersOfInterest[p]])
        if trainedOnPrice :
            pred, volLoc, theta, gammaK, loss = create_train_model(modelFactory,
                                                                   scaledDataSet,
                                                                   activateDupireReg,
                                                                   hyperparameters,
                                                                   scaler,
                                                                   modelName=modelName)
        else :
            pred, volLoc, theta, gammaK, loss = create_train_model_gatheral(modelFactory,
                                                                            scaledDataSet,
                                                                            activateDupireReg,
                                                                            hyperparameters,
                                                                            scaler,
                                                                            modelName=modelName)

        nbArbitrageViolation = np.sum((theta <= 0)) + np.sum((gammaK <= 0))
        print("loss : ", min(loss))
        print("nbArbitrageViolation : ", nbArbitrageViolation)
        print()
        print()
        print()

    hyperparameters["maxEpoch"] = oldNbEpochs
    for k in parametersOfInterest:
        hyperparameters[k] = oldValue[k]

    return
