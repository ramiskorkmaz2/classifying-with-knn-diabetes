library(mlr)
library(tidyverse)

data(diabetes, package = "mclust")
diabetesTib <- as_tibble(diabetes)

summary(diabetesTib)

ggplot(diabetesTib, aes(glucose, insulin, shape = class,
                        col = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, insulin, shape = class, col = class)) +
  geom_point() +
  theme_bw()

ggplot(diabetesTib, aes(sspg, glucose, shape = class, col = class)) +
  geom_point() +
  theme_bw()


### Never eveluate model this way

diabetesTask <- makeClassifTask(data = diabetesTib, target = "class")
diabetesTask

knn <- makeLearner("classif.knn", par.vals = list("k"= 3))
knn


knnModel <- train(knn, diabetesTask)
knnPred <- predict(knnModel, newdata = diabetesTib)

performance(knnPred, measures = list(mmce, acc))

###

diabetesTask <- makeClassifTask(data = diabetesTib, target = "class")
knn <- makeLearner("classif.knn", par.vals = list("k"= 3))


# Holdout cross-validation

holdout <- makeResampleDesc(method = "Holdout",
                            split = 2/3,
                            stratify = TRUE)

holdoutCV <- resample(learner = knn, task = diabetesTask,
                      resampling = holdout , measures = list(mmce, acc))


holdoutCV$aggr

calculateConfusionMatrix(holdoutCV$pred, relative = TRUE)


## Exercise 2
holdout_2 <- makeResampleDesc(method = "Holdout",
                              split = 0.9,
                              stratify = TRUE)
holdoutCV_2 <- resample(learner = knn,
         task = diabetesTask,
         resampling = holdout_2,
         measures = list(mmce,acc))

calculateConfusionMatrix(holdoutCV_2$pred, relative = TRUE)
####


## K-fold cross validation

kFold <- makeResampleDesc(method = "RepCV",
                          folds = 10,
                          reps = 50,
                          stratify = TRUE)

kFoldCV <- resample(learner = knn,
                    task = diabetesTask,
                    resampling = kFold, 
                    measures = list(mmce, acc))
kFoldCV$aggr


calculateConfusionMatrix(kFoldCV$pred, relative = TRUE)



## Exercise 3 

kfold_2 <- makeResampleDesc(method = "RepCV",
                            folds = 3,
                            reps = 5,
                            stratify = TRUE)


kfoldCV_2 <- resample(learner = knn,
                      task = diabetesTask,
                      resampling = kfold_2,
                      measures = list(mmce, acc))
kfoldCV_2$aggr


kfold_3 <- makeResampleDesc(method = "RepCV",
                            folds = 3,
                            reps = 500,
                            stratify = TRUE)
kfoldCV_3 <- resample(learner = knn,
                      task = diabetesTask,
                      resampling = kfold_3,
                      measures = list(mmce, acc))
kfoldCV_3$aggr

####


## Leave-one-out Cross Validation
LOO <- makeResampleDesc(method = "LOO")
LOOCV <- resample(learner = knn,
                  task = diabetesTask,
                  resampling = LOO,
                  measures = list(mmce, acc))
LOOCV$aggr
calculateConfusionMatrix(LOOCV$pred, relative = TRUE)

## Exercise 4

loo_2 <- makeResampleDesc(method = "LOO",
                          stratify = TRUE)

loo_3 <- makeResampleDesc(method = "LOO",
                          reps = 5)


###


knnParamSpace <- makeParamSet(makeDiscreteParam("k", values = 1:10))
gridSearch <- makeTuneControlGrid()
cvforTuning <- makeResampleDesc("RepCV",
                                folds = 10,
                                reps = 20)
tuneDK <- tuneParams("classif.knn",
                     task = diabetesTask,
                     resampling = cvforTuning,
                     par.set = knnParamSpace,
                     control = gridSearch)

knnTuningData <- generateHyperParsEffectData(tuneDK)
plotHyperParsEffect(knnTuningData, x = "k",
                    y = "mmce.test.mean",
                    plot.type = "line") +
  theme_bw()

## Train the model

tunedKnn <- setHyperPars(makeLearner("classif.knn"),
                         par.vals = tuneDK$x)

tunedKnnModel <- train(tunedKnn, diabetesTask)

##

inner <- makeResampleDesc("CV")
outer <- makeResampleDesc("RepCV", folds = 10, reps = 5)
knnWrapper <- makeTuneWrapper("classif.knn",
                              resampling = inner,
                              control = gridSearch,
                              par.set = knnParamSpace)

cvWithTuning <- resample(knnWrapper, diabetesTask, resampling = outer)
cvWithTuning

### Using model to make prediction

newDiabetespatients <- tibble(glucose = c(90, 105, 295),
                              insulin = c(360, 280, 1050),
                              sspg = c(190, 200, 130))

newpred <- predict(tunedKnnModel, newdata = newDiabetespatients)
getPredictionResponse(newpred)
