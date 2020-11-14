## +---------------------------------------------------------------------------------------------------+
## | AudioSimilarity_Q2.py: DATA420 20S2 :: Assignment 2 :: Audio Similarity :: Question 2             |
## |                                                                                                   |
## | __author__ = "Gurpreet Singh"                                                                     |
## | __ID__     = "24701854"                                                                           |
## | __email__  = "gsi58@uclive.ac.nz"                                                                 |
## | __date__   = "03 October ‎2020"                                                                    |
## +---------------------------------------------------------------------------------------------------+

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 2a: Based on the descriptive statistics from Q1 part (a), decide what processing |
## |                      you should apply to the audio features before using them to train each model.|
## +---------------------------------------------------------------------------------------------------+

# remove the 3rd, 7th and 9th column as they are highly correlated with 4, 6, and 8
audio_features_genre = (
	audio_features_genre.drop("feature_0003")
	.drop("feature_0007")
	.drop("feature_0009")
)
audio_features_genre.show()

'''output
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+
|          track_id|feature_0000|feature_0001|feature_0002|feature_0004|feature_0005|feature_0006|feature_0008|   genre_type|
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+
|TRAAABD128F429CF47|      0.1308|       9.587|       459.9|   4303000.0|      0.2474|       26.02|     67790.0|     Pop_Rock|
|TRAABPK128F424CFDB|      0.1208|       6.738|       215.1|   2278000.0|      0.4882|       41.76|    220400.0|     Pop_Rock|
|TRAACER128F4290F96|      0.2838|       8.995|       429.5|   5272000.0|      0.5388|       28.29|    185100.0|     Pop_Rock|
|TRAADYB128F92D7E73|      0.1346|       7.321|       499.6|   5877000.0|      0.2839|       15.75|    116500.0|         Jazz|
|TRAAGHM128EF35CF8E|      0.1563|       9.959|       502.8|   3660000.0|      0.3835|       28.24|    180800.0|   Electronic|
|TRAAGRV128F93526C0|      0.1076|       7.401|       389.7|   2739000.0|      0.4221|       30.99|    191700.0|     Pop_Rock|
|TRAAGTO128F1497E3C|      0.1069|       8.987|       562.6|   7057000.0|      0.1007|        22.9|    157700.0|     Pop_Rock|
|TRAAHAU128F9313A3D|     0.08485|       9.031|       445.9|   3274000.0|      0.2583|       35.59|    198400.0|     Pop_Rock|
|TRAAHEG128E07861C3|      0.1699|       17.22|       741.3|   8275000.0|      0.2812|       28.83|    160800.0|          Rap|
|TRAAHZP12903CA25F4|      0.1654|       12.31|       565.1|   5273000.0|      0.1861|       38.38|    196600.0|          Rap|
|TRAAICW128F1496C68|      0.1104|       7.123|       398.2|   3240000.0|      0.2871|       28.53|    189400.0|International|
|TRAAJJW12903CBDDCB|      0.2267|       14.88|       592.7|   4569000.0|      0.4219|       36.17|    179400.0|International|
|TRAAJKJ128F92FB44F|     0.03861|        6.87|       407.8|   7299000.0|      0.0466|       15.79|    121700.0|         Folk|
|TRAAKLX128F934CEE4|      0.1647|       16.77|       850.0|     1.011E7|      0.2823|       26.52|    152000.0|   Electronic|
|TRAAKWR128F931B29F|     0.04881|       9.331|       564.0|   4920000.0|     0.08647|        18.1|     57700.0|     Pop_Rock|
|TRAALQN128E07931A4|      0.1989|       12.83|       578.7|   4921000.0|      0.5452|       33.37|    188700.0|   Electronic|
|TRAAMFF12903CE8107|      0.1385|       9.699|       581.6|   4569000.0|      0.3706|       23.63|    163800.0|     Pop_Rock|
|TRAAMHG128F92ED7B2|      0.1799|       10.52|       551.4|   4396000.0|      0.4046|       30.78|    183200.0|International|
|TRAAROH128F42604B0|      0.1192|        16.4|       737.3|   6295000.0|      0.2284|       31.04|    169100.0|   Electronic|
|TRAARQN128E07894DF|      0.2559|       15.23|       757.1|     1.065E7|      0.5417|       40.96|    189000.0|     Pop_Rock|
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+
only showing top 20 rows
'''


## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 2b: Convert the genre column into a column representing if the song is ”Rap” or  |
## |                      some other genre as a binary label.                                          |
## |                      What is the class balance of the binary label?                               |
## +---------------------------------------------------------------------------------------------------+

audio_features_genre_class = audio_features_genre.withColumn('genre', when(F.col('genre_type') == 'Rap', 1).otherwise(0))
audio_features_genre_class.show()

'''Output
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+-----+
|          track_id|feature_0000|feature_0001|feature_0002|feature_0004|feature_0005|feature_0006|feature_0008|   genre_type|genre|
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+-----+
|TRAAABD128F429CF47|      0.1308|       9.587|       459.9|   4303000.0|      0.2474|       26.02|     67790.0|     Pop_Rock|    0|
|TRAABPK128F424CFDB|      0.1208|       6.738|       215.1|   2278000.0|      0.4882|       41.76|    220400.0|     Pop_Rock|    0|
|TRAACER128F4290F96|      0.2838|       8.995|       429.5|   5272000.0|      0.5388|       28.29|    185100.0|     Pop_Rock|    0|
|TRAADYB128F92D7E73|      0.1346|       7.321|       499.6|   5877000.0|      0.2839|       15.75|    116500.0|         Jazz|    0|
|TRAAGHM128EF35CF8E|      0.1563|       9.959|       502.8|   3660000.0|      0.3835|       28.24|    180800.0|   Electronic|    0|
|TRAAGRV128F93526C0|      0.1076|       7.401|       389.7|   2739000.0|      0.4221|       30.99|    191700.0|     Pop_Rock|    0|
|TRAAGTO128F1497E3C|      0.1069|       8.987|       562.6|   7057000.0|      0.1007|        22.9|    157700.0|     Pop_Rock|    0|
|TRAAHAU128F9313A3D|     0.08485|       9.031|       445.9|   3274000.0|      0.2583|       35.59|    198400.0|     Pop_Rock|    0|
|TRAAHEG128E07861C3|      0.1699|       17.22|       741.3|   8275000.0|      0.2812|       28.83|    160800.0|          Rap|    1|
|TRAAHZP12903CA25F4|      0.1654|       12.31|       565.1|   5273000.0|      0.1861|       38.38|    196600.0|          Rap|    1|
|TRAAICW128F1496C68|      0.1104|       7.123|       398.2|   3240000.0|      0.2871|       28.53|    189400.0|International|    0|
|TRAAJJW12903CBDDCB|      0.2267|       14.88|       592.7|   4569000.0|      0.4219|       36.17|    179400.0|International|    0|
|TRAAJKJ128F92FB44F|     0.03861|        6.87|       407.8|   7299000.0|      0.0466|       15.79|    121700.0|         Folk|    0|
|TRAAKLX128F934CEE4|      0.1647|       16.77|       850.0|     1.011E7|      0.2823|       26.52|    152000.0|   Electronic|    0|
|TRAAKWR128F931B29F|     0.04881|       9.331|       564.0|   4920000.0|     0.08647|        18.1|     57700.0|     Pop_Rock|    0|
|TRAALQN128E07931A4|      0.1989|       12.83|       578.7|   4921000.0|      0.5452|       33.37|    188700.0|   Electronic|    0|
|TRAAMFF12903CE8107|      0.1385|       9.699|       581.6|   4569000.0|      0.3706|       23.63|    163800.0|     Pop_Rock|    0|
|TRAAMHG128F92ED7B2|      0.1799|       10.52|       551.4|   4396000.0|      0.4046|       30.78|    183200.0|International|    0|
|TRAAROH128F42604B0|      0.1192|        16.4|       737.3|   6295000.0|      0.2284|       31.04|    169100.0|   Electronic|    0|
|TRAARQN128E07894DF|      0.2559|       15.23|       757.1|     1.065E7|      0.5417|       40.96|    189000.0|     Pop_Rock|    0|
+------------------+------------+------------+------------+------------+------------+------------+------------+-------------+-----+
only showing top 20 rows
'''

audio_features_genre_class.groupby('genre').count().show()


''' output
+-----+------+
|genre| count|  
+-----+------+
|    1| 20899|
|    0|399721|
+-----+------+
'''

## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 2c: Split the dataset into training and test sets. Note that you may need to take|
## |                      class balance into account using a sampling method such as stratification,   |
## |                      subsampling, or oversampling. Justify your choice of sampling method.        |
## +---------------------------------------------------------------------------------------------------+


from pyspark.sql.window import *
from pyspark.ml.feature import VectorAssembler
import numpy as np


def print_class_balance(data, name):
    N = data.count()
    counts = data.groupBy("genre").count().toPandas()
    counts["ratio"] = counts["count"] / N
    print(name)
    print(N)
    print(counts)
    print("")

# Assemble features

assembler = VectorAssembler(
    inputCols=[col for col in audio_features_genre_class.columns if col.startswith("feature_")],
    outputCol="Features"
)
features = assembler.transform(audio_features_genre_class).select(["Features", "genre"])
features.cache()
features.count()

''' output
420620
'''

# randomSplit (not stratified)

temp = (
    features
    .withColumn("id", monotonically_increasing_id())
    .withColumn("Random", rand())
    .withColumn(
        "Row",
        row_number()
        .over(
            Window
            .partitionBy("genre")
            .orderBy("Random")
        )
    )
)

class0 = audio_features_genre_class.groupby('genre').count().select(F.col('count')).where(F.col('genre') == 0).first()[0]
class1 = audio_features_genre_class.groupby('genre').count().select(F.col('count')).where(F.col('genre') == 1).first()[0]

training = temp.where(
    ((col("genre") == 0) & (col("Row") <  class0 * 0.8)) |
    ((col("genre") == 1) & (col("Row") < class1 * 0.8))
)
training.cache()
training.show()

''' output
+-----------------------------------------------------+-----+------------+---------------------+---+
|Features                                             |genre|id          |Random               |Row|
+-----------------------------------------------------+-----+------------+---------------------+---+
|[0.2378,14.91,741.1,5588000.0,0.4145,27.04,160600.0] |1    |77309423578 |1.7138632536162213E-5|1  |
|[0.2311,15.78,587.9,5865000.0,0.4351,32.64,177800.0] |1    |128849019924|5.517053790682347E-5 |2  |
|[0.2515,11.05,546.3,5319000.0,0.5545,26.85,175100.0] |1    |85899357825 |1.1312725853174221E-4|3  |
|[0.164,11.64,562.8,6475000.0,0.3013,28.35,176700.0]  |1    |42949673210 |1.6545153554536896E-4|4  |
|[0.243,16.85,764.3,5645000.0,0.4726,28.97,158000.0]  |1    |60129550453 |2.1364649208022168E-4|5  |
|[0.2742,14.38,721.5,6774000.0,0.4333,27.88,162200.0] |1    |103079236266|2.1931072350211966E-4|6  |
|[0.2855,15.42,630.7,5868000.0,0.5198,36.57,181000.0] |1    |103079237264|2.2057056635360617E-4|7  |
|[0.1528,14.31,753.6,9754000.0,0.145,30.05,176200.0]  |1    |68719479934 |2.764449295205029E-4 |8  |
|[0.2128,14.69,669.1,5050000.0,0.3653,32.46,175600.0] |1    |51539616837 |3.531065412082368E-4 |9  |
|[0.3628,10.97,578.2,6313000.0,0.5161,26.91,170100.0] |1    |68719485588 |3.6204475095524824E-4|10 |
|[0.1077,4.876,315.4,7126000.0,0.1095,17.37,113200.0] |1    |60129546118 |3.7798886953066546E-4|11 |
|[0.08598,15.3,713.9,8848000.0,0.1003,28.11,168100.0] |1    |42949698759 |4.3299993702172745E-4|12 |
|[0.1398,13.22,569.8,5464000.0,0.1428,31.14,178300.0] |1    |128849019581|6.714913235630338E-4 |13 |
|[0.2388,10.1,492.7,4513000.0,0.4872,30.46,186600.0]  |1    |120259107762|6.827944206018177E-4 |14 |
|[0.2691,14.39,661.3,5013000.0,0.4102,31.29,172600.0] |1    |17179886095 |7.631165100570048E-4 |15 |
|[0.2026,16.02,698.5,5438000.0,0.3083,31.22,165100.0] |1    |120259107303|8.050338279314007E-4 |16 |
|[0.2219,19.13,869.1,6241000.0,0.3118,37.95,133300.0] |1    |34359746326 |8.21224799098963E-4  |17 |
|[0.1308,14.79,580.3,4819000.0,0.3262,34.91,181800.0] |1    |17179873265 |9.037059725380825E-4 |18 |
|[0.06211,15.82,785.5,8214000.0,0.1356,22.87,141300.0]|1    |128849027806|9.212331245755934E-4 |19 |
|[0.2084,20.49,823.8,6274000.0,0.3278,36.26,124200.0] |1    |94489286872 |9.278568886339489E-4 |20 |
+-----------------------------------------------------+-----+------------+---------------------+---+
only showing top 20 rows
'''

test = temp.join(training, on = "id", how = "left_anti")
test.cache()
test.show(truncate = False)

''' output
+------------+-----------------------------------------------------+-----+------------------+-----+
|id          |Features                                             |genre|Random            |Row  |
+------------+-----------------------------------------------------+-----+------------------+-----+
|111669168856|[0.2014,13.57,654.2,8010000.0,0.3123,23.38,149500.0] |1    |0.8032054373073088|16720|
|103079217551|[0.2943,15.26,692.3,5683000.0,0.564,28.55,164500.0]  |1    |0.8032145267651812|16721|
|120259106904|[0.245,12.36,672.3,5076000.0,0.4624,27.36,169000.0]  |1    |0.8032156525397153|16722|
|128849037879|[0.1776,8.736,394.7,5208000.0,0.2978,27.95,187700.0] |1    |0.8032229311721685|16723|
|42949674153 |[0.1035,19.98,573.7,4330000.0,0.09343,41.69,168100.0]|1    |0.80333102484558  |16724|
|85899354782 |[0.08865,10.41,540.2,4207000.0,0.2342,26.91,169000.0]|1    |0.8033430336668064|16725|
|34359759156 |[0.1944,11.56,698.0,8938000.0,0.318,17.9,129400.0]   |1    |0.8036430631181076|16726|
|8589952864  |[0.09753,13.03,720.4,6408000.0,0.1965,30.44,109400.0]|1    |0.8037507552894546|16727|
|34359739284 |[0.196,15.85,802.0,1.033E7,0.2239,30.08,174800.0]    |1    |0.8037635818730505|16728|
|103079231149|[0.1154,16.12,624.8,4689000.0,0.2079,36.43,176400.0] |1    |0.8037973725147828|16729|
|85899352182 |[0.1147,13.38,711.7,6593000.0,0.296,25.0,154900.0]   |1    |0.8038167305582303|16730|
|128849032215|[0.211,14.29,773.4,8673000.0,0.3871,21.1,135400.0]   |1    |0.8038470167472311|16731|
|94489295301 |[0.2123,14.6,704.8,5223000.0,0.422,27.99,158900.0]   |1    |0.803950960217415 |16732|
|17179874939 |[0.2058,13.83,624.2,5577000.0,0.3414,27.68,169700.0] |1    |0.804098757662977 |16733|
|103079237998|[0.2156,12.56,723.0,5622000.0,0.1943,27.07,158600.0] |1    |0.8041020054996662|16734|
|103079222832|[0.2991,16.26,682.4,6171000.0,0.5033,34.99,176700.0] |1    |0.8042194140389988|16735|
|128849032512|[0.2116,24.17,1051.0,7364000.0,0.2294,30.71,72120.0] |1    |0.8042341647136527|16736|
|60129544671 |[0.2776,18.58,866.6,9291000.0,0.3237,26.54,148800.0] |1    |0.8042471035900055|16737|
|103079237652|[0.2169,17.18,769.4,7704000.0,0.3126,29.5,161700.0]  |1    |0.8043351088411925|16738|
|85899356052 |[0.3305,14.56,571.7,3857000.0,0.5968,31.08,80420.0]  |1    |0.8043629340690961|16739|
+------------+-----------------------------------------------------+-----+------------------+-----+
only showing top 20 rows
'''

training = training.drop("id", "Random", "Row")
test = test.drop("id", "Random", "Row")

print_class_balance(features, "features")
print_class_balance(training, "training")
print_class_balance(test, "test")

''' output
features
420620
   genre   count     ratio
0      1   20899  0.049686
1      0  399721  0.950314

training
336495
   genre   count     ratio
0      1   16719  0.049686
1      0  319776  0.950314

test
84125
   genre  count     ratio
0      1   4180  0.049688
1      0  79945  0.950312
'''

# ----------
# Up sampling
# ----------
# Randomly upsample by exploding a vector of length betwen 0 and n for each row

ratio = 10
n = 20
p = ratio / n  # ratio < n such that probability < 1

def random_resample(x, n, p):
    # Can implement custom sampling logic per class,
    if x == 0:
        return [0]  # no sampling
    if x == 1:
        return list(range((np.sum(np.random.random(n) > p))))  # upsampling
    return []  # drop

random_resample_udf = udf(lambda x: random_resample(x, n, p), ArrayType(IntegerType()))

training_upsampled = (
    training
    .withColumn("Sample", random_resample_udf(col("genre")))
    .select(
        col("Features"),
        col("genre"),
        explode(col("Sample")).alias("Sample")
    )
    .drop("Sample")
)
print_class_balance(features, "features")
print_class_balance(training_upsampled, "training_upsampled")

''' output
features
420620
   genre   count     ratio
0      1   20899  0.049686
1      0  399721  0.950314

training_upsampled
487004
   genre   count     ratio
0      1  167228  0.343381
1      0  319776  0.656619
'''

# ------------
# Downsampling
# ------------

training_downsampled = (
    training
    .withColumn("Random", rand())
    .where((col("genre") != 0) | ((col("genre") == 0) & (col("Random") < 2 * (class1 / class0))))
)
training_downsampled.cache()

print_class_balance(features, "features")
print_class_balance(training_downsampled, "training_downsampled")


''' output
features
420620
   genre   count     ratio
0      1   20899  0.049686
1      0  399721  0.950314

training_downsampled
50021
   genre  count    ratio
0      1  16719  0.33424
1      0  33302  0.66576
'''

# ------------------------------
# UpSample and then Downsampling
# ------------------------------

training_updownsampled = (
    training_upsampled
    .withColumn("Random", rand())
    .where((col("genre") != 0) | ((col("genre") == 0) & (col("Random") < 10 * (class1 / class0))))
)
training_updownsampled.cache()

print_class_balance(features, "features")
print_class_balance(training_updownsampled, "training_updownsampled")


''' output
features
420620
   genre   count     ratio
0      1   20899  0.049686
1      0  399721  0.950314

training_updownsampled
335159
   genre   count     ratio
0      1  167483  0.499712
1      0  167676  0.500288
'''


# -----------------------
# Observation reweighting
# -----------------------

training_weighted = (
    training
    .withColumn(
        "Weight",
        when(col("genre") == 0, 1.0)
        .when(col("genre") == 1, 10.0)
        .otherwise(1.0)
    )
)

weights = (
    training_weighted
    .groupBy("genre")
    .agg(
        collect_set(col("Weight")).alias("Weights")
    )
    .toPandas()
)
print(weights)

''' output
   genre Weights
0      1  [10.0]
1      0   [1.0]
'''


## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 2d: Train each of the three classification algorithms that you chose in part (a).|
## +---------------------------------------------------------------------------------------------------+

def print_binary_metrics(predictions, predictionsName, labelCol, predictionCol = "prediction", rawPredictionCol = "rawPrediction"):

    total = predictions.count()
    positive = predictions.filter((col(labelCol) == 1)).count()
    negative = predictions.filter((col(labelCol) == 0)).count()
    nP = predictions.filter((col(predictionCol) == 1)).count()
    nN = predictions.filter((col(predictionCol) == 0)).count()
    TP = predictions.filter((col(predictionCol) == 1) & (col(labelCol) == 1)).count()
    FP = predictions.filter((col(predictionCol) == 1) & (col(labelCol) == 0)).count()
    FN = predictions.filter((col(predictionCol) == 0) & (col(labelCol) == 1)).count()
    TN = predictions.filter((col(predictionCol) == 0) & (col(labelCol) == 0)).count()

    binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol = rawPredictionCol, labelCol = labelCol, metricName = "areaUnderROC")
    auroc = binary_evaluator.evaluate(predictions)
    print(predictionsName)
    print('----------------------------')
    print('actual total:    {}'.format(total))
    print('actual positive: {}'.format(positive))
    print('actual negative: {}'.format(negative))
    print('nP:              {}'.format(nP))
    print('nN:              {}'.format(nN))
    print('TP:              {}'.format(TP))
    print('FP:              {}'.format(FP))
    print('FN:              {}'.format(FN))
    print('TN:              {}'.format(TN))
    print('precision:       {}'.format(TP / (TP + FP)))
    print('recall:          {}'.format(TP / (TP + FN)))
    print('accuracy:        {}'.format((TP + TN) / total))
    print('auroc:           {}'.format(auroc))

def with_custom_prediction(predictions, threshold, probabilityCol = "probability", customPredictionCol = "customPrediction"):

    def apply_custom_threshold(probability, threshold):
        return int(probability[1] > threshold)

    apply_custom_threshold_udf = udf(lambda x: apply_custom_threshold(x, threshold), IntegerType())

    return predictions.withColumn(customPredictionCol, apply_custom_threshold_udf(col(probabilityCol)))


# --------------------
# Logistoic Regression
# --------------------


# -----------
# No sampling
# -----------
lr = LogisticRegression(featuresCol = 'Features', labelCol = 'genre')
nosampling_lr_model = lr.fit(training)

# --------------------------------
# upsampling
# --------------------------------

lr = LogisticRegression(featuresCol = 'Features', labelCol = 'genre')
upsampling_lr_model = lr.fit(training_upsampled)

# --------------------------------
# downsampling
# --------------------------------

lr = LogisticRegression(featuresCol = 'Features', labelCol = 'genre')
downsampling_lr_model = lr.fit(training_downsampled)

# --------------------------------
# upsampling and then downsampling
# --------------------------------

lr = LogisticRegression(featuresCol = 'Features', labelCol = 'genre')
updownsampling_lr_model = lr.fit(training_updownsampled)

# -----------------------
# Observation reweighting
# -----------------------
lr = LogisticRegression(featuresCol = 'Features', labelCol = 'genre', weightCol = "Weight")
reweighting_lr_model = lr.fit(training_weighted)


# ------------------------
# Random Forest Classifier
# ------------------------
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol = 'Features', labelCol = 'genre')
upsampling_rf_model = rf.fit(training_upsampled)

rf = RandomForestClassifier(featuresCol = 'Features', labelCol = 'genre')
downsampling_rf_model = rf.fit(training_downsampled)

rf = RandomForestClassifier(featuresCol = 'Features', labelCol = 'genre')
updownsampling_rf_model = rf.fit(training_updownsampled)


# ------------------------
# Decision Tree Classifier
# ------------------------

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol = 'Features', labelCol = 'genre', maxDepth = 5)
upsampling_dt_model = dt.fit(training_upsampled)

dt = DecisionTreeClassifier(featuresCol = 'Features', labelCol = 'genre', maxDepth = 5)
downsampling_dt_model = dt.fit(training_downsampled)


dt = DecisionTreeClassifier(featuresCol = 'Features', labelCol = 'genre', maxDepth = 5)
updownsampling_dt_model = dt.fit(training_updownsampled)




## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 2e: Use the test set to compute the compute a range of performance metrics for   |
## |                      each model, such as precision, accuracy, and recall.                         |
## +---------------------------------------------------------------------------------------------------+

# --------------------
# Logistoic Regression
# --------------------
# -----------
# No sampling
# -----------
nosampling_lr_predictions = nosampling_lr_model.transform(test)
nosampling_lr_predictions.cache()
print_binary_metrics(nosampling_lr_predictions, 'nosampling_lr_predictions', labelCol = 'genre')

'''output
nosampling_lr_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              548
nN:              83577
TP:              161
FP:              387
FN:              4019
TN:              79558
precision:       0.2937956204379562
recall:          0.038516746411483255
accuracy:        0.9476255572065378
auroc:           0.8513347364111903
'''


# --------------------------------
# upsampling
# --------------------------------
upsampling_lr_predictions = upsampling_lr_model.transform(test)
upsampling_lr_predictions.cache()
print_binary_metrics(upsampling_lr_predictions, 'upsampling_lr_predictions', labelCol = 'genre')

''' Output
upsampling_lr_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              11905
nN:              72220
TP:              2608
FP:              9297
FN:              1572
TN:              70648
precision:       0.21906761864762705
recall:          0.6239234449760765
accuracy:        0.8707994056463596
auroc:           0.8528140144794596
'''

# --------------------------------
# downsampling
# --------------------------------
downsampling_lr_predictions = downsampling_lr_model.transform(test)
downsampling_lr_predictions.cache()
print_binary_metrics(downsampling_lr_predictions, 'downsampling_lr_predictions', labelCol = 'genre')

''' Output
downsampling_lr_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              11437
nN:              72688
TP:              2555
FP:              8882
FN:              1625
TN:              71063
precision:       0.22339774416367927
recall:          0.611244019138756
accuracy:        0.8751025260029718
auroc:           0.8528168214331782
'''


# --------------------------------
# upsampling and then downsampling
# --------------------------------
updownsampling_lr_predictions = updownsampling_lr_model.transform(test)
updownsampling_lr_predictions.cache()
print_binary_metrics(updownsampling_lr_predictions, 'updownsampling_lr_predictions', labelCol = 'genre')

'''output
updownsampling_lr_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              19934
nN:              64191
TP:              3236
FP:              16698
FN:              944
TN:              63247
precision:       0.16233570783585832
recall:          0.7741626794258373
accuracy:        0.7902882615156018
auroc:           0.8534162377184447
'''

# -----------------------
# Observation Reweighting
# -----------------------
reweighting_lr_predictions = reweighting_lr_model.transform(test)
reweighting_lr_predictions.cache()

print_binary_metrics(reweighting_lr_predictions, 'reweighting_lr_predictions', labelCol = 'genre')


''' output
reweighting_lr_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              11905
nN:              72220
TP:              2604
FP:              9301
FN:              1576
TN:              70644
precision:       0.21873162536749266
recall:          0.6229665071770335
accuracy:        0.8707043090638931
auroc:           0.8528578514355408

'''


# --------------------
# Random Forest Classifier
# --------------------

upsampling_rf_predictions = upsampling_rf_model.transform(test)
upsampling_rf_predictions.cache()
print_binary_metrics(upsampling_rf_predictions, 'upsampling_rf_predictions', labelCol = 'genre')

''' output
upsampling_rf_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              13562
nN:              70563
TP:              2746
FP:              10816
FN:              1434
TN:              69129
precision:       0.2024775106916384
recall:          0.6569377990430622
accuracy:        0.8543833580980683
auroc:           0.846539458497334
'''

downsampling_rf_predictions = downsampling_rf_model.transform(test)
downsampling_rf_predictions.cache()
print_binary_metrics(downsampling_rf_predictions, 'downsampling_rf_predictions', labelCol = 'genre')

''' output
downsampling_rf_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              13597
nN:              70528
TP:              2670
FP:              10927
FN:              1510
TN:              69018
precision:       0.196366845627712
recall:          0.638755980861244
accuracy:        0.8521604754829123
auroc:           0.8454271312723668
'''


updownsampling_rf_predictions = updownsampling_rf_model.transform(test)
updownsampling_rf_predictions.cache()
print_binary_metrics(updownsampling_rf_predictions, 'updownsampling_rf_predictions', labelCol = 'genre')

''' Output
updownsampling_rf_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              23729
nN:              60396
TP:              3394
FP:              20335
FN:              786
TN:              59610
precision:       0.14303173332209532
recall:          0.8119617224880383
accuracy:        0.7489331352154532
auroc:           0.8506102999640006
'''

# ------------------------
# Decision Tree Classifier
# ------------------------

upsampling_dt_predictions = upsampling_dt_model.transform(test)
upsampling_dt_predictions.cache()
print_binary_metrics(upsampling_dt_predictions, 'upsampling_dt_predictions', labelCol = 'genre')

''' output
upsampling_dt_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              14870
nN:              69255
TP:              2834
FP:              12036
FN:              1346
TN:              67909
precision:       0.19058507061197041
recall:          0.6779904306220096
accuracy:        0.8409271916790491
auroc:           0.5248683439960667
'''

downsampling_dt_predictions = downsampling_dt_model.transform(test)
downsampling_dt_predictions.cache()
print_binary_metrics(downsampling_dt_predictions, 'downsampling_dt_predictions', labelCol = 'genre')

''' output
downsampling_dt_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              15398
nN:              68727
TP:              2842
FP:              12556
FN:              1338
TN:              67389
precision:       0.18456942460059747
recall:          0.6799043062200957
accuracy:        0.8348410104011887
auroc:           0.5259307954242466
'''

updownsampling_dt_predictions = updownsampling_dt_model.transform(test)
updownsampling_dt_predictions.cache()
print_binary_metrics(updownsampling_dt_predictions, 'updownsampling_dt_predictions', labelCol = 'genre')

''' output
updownsampling_dt_predictions
----------------------------
actual total:    84125
actual positive: 4180
actual negative: 79945
nP:              23745
nN:              60380
TP:              3373
FP:              20372
FN:              807
TN:              59573
precision:       0.14205095809644136
recall:          0.8069377990430622
accuracy:        0.7482436849925705
auroc:           0.7318534647474444
'''


from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Helpers

def evaluate_metrics_and_cost_local(predictions_local, threshold, FP_cost, FN_cost):

    predictions_local["customPrediction"] = (predictions_local["probability"].apply(lambda x: x[1]) > threshold).astype(int)

    total = predictions_local.shape[0]
    positive = (predictions_local["genre"] == 1).sum()
    negative = (predictions_local["genre"] == 0).sum()
    nP = (predictions_local["customPrediction"] == 1).sum()
    nN = (predictions_local["customPrediction"] == 0).sum()
    TP = ((predictions_local["customPrediction"] == 1) & (predictions_local["genre"] == 1)).sum()
    FP = ((predictions_local["customPrediction"] == 1) & (predictions_local["genre"] == 0)).sum()
    FN = ((predictions_local["customPrediction"] == 0) & (predictions_local["genre"] == 1)).sum()
    TN = ((predictions_local["customPrediction"] == 0) & (predictions_local["genre"] == 0)).sum()

    precision = TP / (TP + FP) if TP + FP > 0 else np.NaN
    recall = TP / (TP + FN) if TP + FN > 0 else np.NaN
    accuracy = (TP + TN) / total

    cost = FP_cost * FP + FN_cost * FN

    return precision, recall, accuracy, cost

def show_axes_legend(a, loc = "upper right"):
  """Create legend based on labeled objects in the order they were labeled.

  Args:
    a (matplotlib.axes.Axes): target axes
    loc (str): axes location (default: "upper right")
  """

  handles, labels = a.get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  legend = a.legend(
    by_label.values(), by_label.keys(),
    borderpad = 0.5,
    borderaxespad = 0,
    fancybox = False,
    edgecolor = "black",
    framealpha = 1,
    loc = loc,
    fontsize = "x-small",
    ncol = 1
  )
  frame = legend.get_frame().set_linewidth(0.75)

# --------------------
# Logistoic Regression
# --------------------

# Evaluate for different thresholds
predictions_local = updownsampling_lr_predictions.select(["genre", "probability"]).toPandas()

FP_cost = 1     # e.g. time cost of a human to manually check a transaction and reject / approve 
FN_cost = 1000  # e.g. bank cost of reversing a fraudulent transaction (charged to the company) 

N = 1000
precision_array = np.zeros(N + 1)
recall_array = np.zeros(N + 1)
accuracy_array = np.zeros(N + 1)
cost_array = np.zeros(N + 1)
thresholds = np.linspace(0, 1, N + 1)

import datetime

counter = 0
for threshold in thresholds:
    precision, recall, accuracy, cost = evaluate_metrics_and_cost_local(predictions_local, threshold, FP_cost, FN_cost)
    precision_array[counter] = precision
    recall_array[counter] = recall
    accuracy_array[counter] = accuracy
    cost_array[counter] = cost
    counter += 1
    # check if counter is a power of 2
    if (counter & (counter - 1) == 0) or counter == len(thresholds) - 1:
        print(f"{datetime.datetime.now()}: {counter:04d}")

# Plot over threshold

# accuracy_array = np.array(accuracy_array)
# precision_array = np.array(precision_array)
# recall_array = np.array(recall_array)
# cost_array = np.array(cost_array)

f, a = plt.subplots(dpi = 300, figsize = (10, 5))

a.plot(thresholds, accuracy_array, color = "orange", label = "accuracy")
a.plot(thresholds, precision_array, color = "red", label = "precision")
a.plot(thresholds, recall_array, color = "blue", label = "recall")
a.plot(thresholds, cost_array / cost_array.max(), color = "black", label = "cost")
a.set_xlim([0 - 0.05, 1 + 0.05])
a.set_ylim([0 - 0.05, 1 + 0.05])

show_axes_legend(a)

plt.title("Metrics tradeoff based on threshold for Logistic Regression")
plt.xlabel("Threshold")
plt.ylabel("Metrics (relative)")

# Save

plt.tight_layout()  # reduce whitespace
f.savefig(os.path.join(os.path.expanduser("~/lr_metrics.png")), bbox_inches = "tight")  # save as png and view in windows
plt.close(f)


# --------------------
# Random Forest Classifier
# --------------------
predictions_local = updownsampling_rf_predictions.select(["genre", "probability"]).toPandas()

FP_cost = 1     # e.g. time cost of a human to manually check a transaction and reject / approve 
FN_cost = 1000  # e.g. bank cost of reversing a fraudulent transaction (charged to the company) 

N = 1000
precision_array = np.zeros(N + 1)
recall_array = np.zeros(N + 1)
accuracy_array = np.zeros(N + 1)
cost_array = np.zeros(N + 1)
thresholds = np.linspace(0, 1, N + 1)

import datetime

counter = 0
for threshold in thresholds:
    precision, recall, accuracy, cost = evaluate_metrics_and_cost_local(predictions_local, threshold, FP_cost, FN_cost)
    precision_array[counter] = precision
    recall_array[counter] = recall
    accuracy_array[counter] = accuracy
    cost_array[counter] = cost
    counter += 1
    # check if counter is a power of 2
    if (counter & (counter - 1) == 0) or counter == len(thresholds) - 1:
        print(f"{datetime.datetime.now()}: {counter:04d}")

# Plot over threshold

# accuracy_array = np.array(accuracy_array)
# precision_array = np.array(precision_array)
# recall_array = np.array(recall_array)
# cost_array = np.array(cost_array)

f, a = plt.subplots(dpi = 300, figsize = (10, 5))

a.plot(thresholds, accuracy_array, color = "orange", label = "accuracy")
a.plot(thresholds, precision_array, color = "red", label = "precision")
a.plot(thresholds, recall_array, color = "blue", label = "recall")
a.plot(thresholds, cost_array / cost_array.max(), color = "black", label = "cost")
a.set_xlim([0 - 0.05, 1 + 0.05])
a.set_ylim([0 - 0.05, 1 + 0.05])

show_axes_legend(a)

plt.title("Metrics Tradeoff Based on Threshold for Random Forest")
plt.xlabel("Threshold")
plt.ylabel("Metrics (relative)")

# Save

plt.tight_layout()  # reduce whitespace
f.savefig(os.path.join(os.path.expanduser("~/rf_metrics.png")), bbox_inches = "tight")  # save as png and view in windows
plt.close(f)


# -------------------------
# Decsision Tree Classifier
# -------------------------
predictions_local = updownsampling_dt_predictions.select(["genre", "probability"]).toPandas()

FP_cost = 1     # e.g. time cost of a human to manually check a transaction and reject / approve 
FN_cost = 1000  # e.g. bank cost of reversing a fraudulent transaction (charged to the company) 

N = 1000
precision_array = np.zeros(N + 1)
recall_array = np.zeros(N + 1)
accuracy_array = np.zeros(N + 1)
cost_array = np.zeros(N + 1)
thresholds = np.linspace(0, 1, N + 1)

import datetime

counter = 0
for threshold in thresholds:
    precision, recall, accuracy, cost = evaluate_metrics_and_cost_local(predictions_local, threshold, FP_cost, FN_cost)
    precision_array[counter] = precision
    recall_array[counter] = recall
    accuracy_array[counter] = accuracy
    cost_array[counter] = cost
    counter += 1
    # check if counter is a power of 2
    if (counter & (counter - 1) == 0) or counter == len(thresholds) - 1:
        print(f"{datetime.datetime.now()}: {counter:04d}")

# Plot over threshold

# accuracy_array = np.array(accuracy_array)
# precision_array = np.array(precision_array)
# recall_array = np.array(recall_array)
# cost_array = np.array(cost_array)

f, a = plt.subplots(dpi = 300, figsize = (10, 5))

a.plot(thresholds, accuracy_array, color = "orange", label = "accuracy")
a.plot(thresholds, precision_array, color = "red", label = "precision")
a.plot(thresholds, recall_array, color = "blue", label = "recall")
a.plot(thresholds, cost_array / cost_array.max(), color = "black", label = "cost")
a.set_xlim([0 - 0.05, 1 + 0.05])
a.set_ylim([0 - 0.05, 1 + 0.05])

show_axes_legend(a)

plt.title("Metrics Tradeoff Based on Threshold for Decsision Tree")
plt.xlabel("Threshold")
plt.ylabel("Metrics (relative)")

# Save

plt.tight_layout()  # reduce whitespace
f.savefig(os.path.join(os.path.expanduser("~/dt_metrics.png")), bbox_inches = "tight")  # save as png and view in windows
plt.close(f)
