## +---------------------------------------------------------------------------------------------------+
## | AudioSimilarity_Q3.py: DATA420 20S2 :: Assignment 2 :: Audio Similarity :: Question 3             |
## |                                                                                                   |
## | __author__ = "Gurpreet Singh"                                                                     |
## | __ID__     = "24701854"                                                                           |
## | __email__  = "gsi58@uclive.ac.nz"                                                                 |
## | __date__   = "29 ‎September ‎2020"                                                                  |
## +---------------------------------------------------------------------------------------------------+


import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 3b: Use cross-validation to tune some of the hyperparameters of your best        |
## |                      performing binary classification model.                                      |
## |                      How has this changed your performance metrics?                               |
## +---------------------------------------------------------------------------------------------------+

rf_cv = RandomForestClassifier(labelCol = 'genre', featuresCol = 'Features')
pipeline = Pipeline(stages = [assembler, rf_cv])

paramGrid = ParamGridBuilder() \
    .addGrid(rf_cv.numTrees, [10, 50, 100]) \
    .addGrid(rf_cv.maxDepth, [3, 6, 9]) \
    .build()

cv = CrossValidator(estimator = rf_cv,
                          estimatorParamMaps = paramGrid,
                          evaluator = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction", labelCol = 'genre', metricName = "areaUnderROC"),
                          numFolds = 5)
                          
cv_model = cv.fit(training_updownsampled)

best_cv_model = cv_model.bestModel

print("Best Depth: ", best_cv_model._java_obj.getMaxDepth())
print("Good Number of Trees: ", best_cv_model._java_obj.getNumTrees())

''' output
Best Depth:  9
Good Number of Trees:  100
'''

rf_best = RandomForestClassifier(featuresCol = 'Features', labelCol = 'genre', numTrees = 100, maxDepth = 9)
updownsampled_rf_best_model = rf_best.fit(training_updownsampled)
updownsampled_rf_best_predictions = updownsampled_rf_best_model.transform(test)
updownsampled_rf_best_predictions.cache()

print_binary_metrics(updownsampled_rf_best_predictions, 'updownsampled_rf_best_predictions', labelCol = 'genre')

''' Output
updownsampled_rf_best_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              20366
nN:              63759
TP:              3288
FP:              17078
FN:              892
TN:              62867
precision:       0.16144554649906706
recall:          0.7866028708133971
accuracy:        0.7863893016344725
auroc:           0.8639041045264123
'''
