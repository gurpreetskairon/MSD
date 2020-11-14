## +---------------------------------------------------------------------------------------------------+
## | AudioSimilarity_Q4.py: DATA420 20S2 :: Assignment 2 :: Audio Similarity :: Question 4             |
## |                                                                                                   |
## | __author__ = "Gurpreet Singh"                                                                     |
## | __ID__     = "24701854"                                                                           |
## | __email__  = "gsi58@uclive.ac.nz"                                                                 |
## | __date__   = "29 ‎September ‎2020"                                                                  |
## +---------------------------------------------------------------------------------------------------+


from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier

## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 4b: Convert the genre column into an integer index that encodes each genre       |
## |                      consistently. Find a way to do this that requires the least amount of work by|
## |                      hand.                                                                        |
## +---------------------------------------------------------------------------------------------------+

audio_features_genre = audio_features.join(genre, on = 'track_id', how = 'inner')
audio_features_genre.show()

audio_features_genre = features_not_mismatched.join(msd_not_mismatched, on = "track_id", how = "inner")

indexer = StringIndexer(inputCol = "genre_type", outputCol = "genre").fit(audio_features_genre)

indexed_audio_features_genre = indexer.transform(audio_features_genre)
indexed_audio_features_genre.show()

''' output
+------------------+------------+------------+------------+------------+------------+------------+------------+----------+-----+
|          track_id|feature_0000|feature_0001|feature_0002|feature_0004|feature_0005|feature_0006|feature_0008|genre_type|genre|
+------------------+------------+------------+------------+------------+------------+------------+------------+----------+-----+
|TRAAABD128F429CF47|      0.1308|       9.587|       459.9|   4303000.0|      0.2474|       26.02|     67790.0|  Pop_Rock|  0.0|
|TRAAAGR128F425B14B|      0.2284|       12.99|       730.3|   5683000.0|      0.5962|       39.32|    109800.0|  Pop_Rock|  0.0|
|TRAAAMO128F1481E7F|      0.1087|       9.765|       523.2|   5161000.0|      0.2674|       21.94|    155700.0| Religious|  8.0|
|TRAABNV128F425CEE1|     0.06862|       12.34|       661.5|   5870000.0|      0.1254|       26.78|    164800.0|   New Age| 13.0|
|TRAABOG128F42955B1|      0.2765|       9.438|       433.9|   3661000.0|      0.6153|       32.93|    194100.0|  Pop_Rock|  0.0|
|TRAABPK128F424CFDB|      0.1208|       6.738|       215.1|   2278000.0|      0.4882|       41.76|    220400.0|  Pop_Rock|  0.0|
|TRAACER128F4290F96|      0.2838|       8.995|       429.5|   5272000.0|      0.5388|       28.29|    185100.0|  Pop_Rock|  0.0|
|TRAACUI128EF367904|     0.07019|       6.088|       370.0|   4533000.0|      0.1715|       19.74|    143200.0|  Pop_Rock|  0.0|
|TRAADQV128F930F792|     0.05175|       8.975|       447.1|   3790000.0|     0.08706|       30.56|    187300.0|     Vocal| 11.0|
|TRAADQX128F422B4CF|      0.2084|       24.29|      1067.0|   8923000.0|      0.3136|        30.2|     60590.0|  Pop_Rock|  0.0|
|TRAADSH128F425C38D|      0.2053|       7.718|       444.1|   4001000.0|      0.8181|       33.72|     75170.0|  Pop_Rock|  0.0|
|TRAADYB128F92D7E73|      0.1346|       7.321|       499.6|   5877000.0|      0.2839|       15.75|    116500.0|      Jazz|  3.0|
|TRAAEEO128F4288E88|      0.1586|       9.511|       481.6|   3946000.0|      0.3906|       31.43|    188800.0|  Pop_Rock|  0.0|
|TRAAFTE128F429545F|      0.1812|       11.55|       610.2|   6605000.0|      0.3214|       27.84|    173300.0|  Pop_Rock|  0.0|
|TRAAGCG128F421CC9F|      0.1646|       16.11|       911.7|   8977000.0|      0.2542|       26.53|    149000.0|     Latin|  4.0|
|TRAAGHM128EF35CF8E|      0.1563|       9.959|       502.8|   3660000.0|      0.3835|       28.24|    180800.0|Electronic|  1.0|
|TRAAGRV128F93526C0|      0.1076|       7.401|       389.7|   2739000.0|      0.4221|       30.99|    191700.0|  Pop_Rock|  0.0|
|TRAAGTO128F1497E3C|      0.1069|       8.987|       562.6|   7057000.0|      0.1007|        22.9|    157700.0|  Pop_Rock|  0.0|
|TRAAHAU128F9313A3D|     0.08485|       9.031|       445.9|   3274000.0|      0.2583|       35.59|    198400.0|  Pop_Rock|  0.0|
|TRAAHDC128EF365E45|       0.178|        11.5|       497.2|   4226000.0|      0.3889|       32.48|    192100.0|Electronic|  1.0|
+------------------+------------+------------+------------+------------+------------+------------+------------+----------+-----+
only showing top 20 rows
'''

assembler = VectorAssembler(
    inputCols = [col for col in indexed_audio_features_genre.columns if col.startswith("feature_")],
    outputCol = "Features"
)
features = assembler.transform(indexed_audio_features_genre).select(["Features", "genre"])
features.cache()
features.count()

''' output
420620
'''

print_class_balance(indexed_audio_features_genre, 'indexed_audio_features_genre')

''' output
indexed_audio_features_genre
420620
    genre   count     ratio
0    13.0    4000  0.009510
1     7.0   11691  0.027795
2     6.0   14194  0.033745
3    12.0    5789  0.013763
4     4.0   17504  0.041615
5     1.0   40666  0.096681
6    17.0    1012  0.002406
7    20.0     200  0.000475
8     8.0    8780  0.020874
9     9.0    6931  0.016478
10   14.0    2067  0.004914
11    0.0  237649  0.564997
12   10.0    6801  0.016169
13   18.0     555  0.001319
14    3.0   17775  0.042259
15   19.0     463  0.001101
16   16.0    1535  0.003649
17    2.0   20899  0.049686
18    5.0   14314  0.034031
19   11.0    6182  0.014697
20   15.0    1613  0.003835
'''

counts = features.groupBy("genre").count().toPandas().set_index("genre")["count"].to_dict()

temp = (
    features
    .withColumn("id", monotonically_increasing_id())
    .withColumn("random", rand())
    .withColumn(
        "row",
        row_number()
        .over(
            Window
            .partitionBy("genre")
            .orderBy("random")
        )
    )
)

training = temp
training.show()

''' output
+--------------------+-----+-----------+--------------------+---+
|            Features|genre|         id|              random|row|
+--------------------+-----+-----------+--------------------+---+
|[0.232,13.66,373....|  6.0|51539611650|2.393213702494012...|  1|
|[0.135,10.63,301....|  6.0|25769842427|2.720093276387114...|  2|
|[0.2537,10.25,520...|  6.0|25769855039|2.950729160636767E-4|  3|
|[0.07079,7.396,45...|  6.0| 8589943986|2.958954268669211...|  4|
|[0.0892,12.78,599...|  6.0|42949675558| 3.30288942774426E-4|  5|
|[0.1245,10.83,584...|  6.0|60129552134| 3.53524087220225E-4|  6|
|[0.07875,3.442,23...|  6.0| 8589987049|3.560717708224192...|  7|
|[0.173,11.75,611....|  6.0|51539649565|4.418865427765484E-4|  8|
|[0.1131,7.45,410....|  6.0|17179886961|5.355541361974359E-4|  9|
|[0.1658,19.5,729....|  6.0|60129569439|6.391769624555232E-4| 10|
|[0.1571,9.318,505...|  6.0|      46907|6.518818043699248E-4| 11|
|[0.0913,7.662,506...|  6.0|17179908601|6.869616226736763E-4| 12|
|[0.09796,9.166,54...|  6.0|60129560573|0.001131957590352...| 13|
|[0.1626,8.984,462...|  6.0|      10332|0.001281804387997...| 14|
|[0.2097,11.64,594...|  6.0|51539640493|0.001283946578605...| 15|
|[0.2263,11.88,630...|  6.0|25769852493|0.001290688293414...| 16|
|[0.1058,7.168,355...|  6.0| 8589944356|0.001295895592863...| 17|
|[0.1176,8.842,498...|  6.0|51539616925|0.001353506907534996| 18|
|[0.2631,10.73,435...|  6.0| 8589950434|0.001381958876001...| 19|
|[0.2182,8.878,498...|  6.0|17179896000|0.001500739818470...| 20|
+--------------------+-----+-----------+--------------------+---+
only showing top 20 rows
'''

for k, v in counts.items():
    training = training.where((col("genre") != k) | (col("row") < v * 0.8))
    
test = temp.join(training, on = "id", how = "left_anti")
test.cache()

training = training.drop("id", "Random", "Row")
test = test.drop("id", "Random", "Row")

print_class_balance(features, "Features:")
print_class_balance(training, "Training:")
print_class_balance(test, "Test:")

''' output
Features:
420620
    genre   count     ratio
0    12.0    5789  0.013763
1     7.0   11691  0.027795
2     6.0   14194  0.033745
3    13.0    4000  0.009510
4     4.0   17504  0.041615
5     1.0   40666  0.096681
6    17.0    1012  0.002406
7    20.0     200  0.000475
8     8.0    8780  0.020874
9     9.0    6931  0.016478
10   14.0    2067  0.004914
11    0.0  237649  0.564997
12   10.0    6801  0.016169
13   18.0     555  0.001319
14    3.0   17775  0.042259
15   19.0     463  0.001101
16   16.0    1535  0.003649
17    2.0   20899  0.049686
18    5.0   14314  0.034031
19   11.0    6182  0.014697
20   15.0    1613  0.003835

Training:
336483
    genre   count     ratio
0     6.0   11355  0.033746
1     7.0    9352  0.027793
2    12.0    4631  0.013763
3    13.0    3199  0.009507
4     1.0   32532  0.096682
5     4.0   14003  0.041616
6    17.0     809  0.002404
7    20.0     159  0.000473
8     8.0    7023  0.020872
9     9.0    5544  0.016476
10   14.0    1653  0.004913
11    0.0  190119  0.565018
12   10.0    5440  0.016167
13   18.0     443  0.001317
14    3.0   14219  0.042258
15   16.0    1227  0.003647
16   19.0     370  0.001100
17    2.0   16719  0.049688
18    5.0   11451  0.034031
19   11.0    4945  0.014696
20   15.0    1290  0.003834

Test:
84137
    genre  count     ratio
0     6.0   2839  0.033743
1     7.0   2339  0.027800
2    12.0   1158  0.013763
3    13.0    801  0.009520
4     1.0   8134  0.096676
5     4.0   3501  0.041611
6    17.0    203  0.002413
7    20.0     41  0.000487
8     9.0   1387  0.016485
9     8.0   1757  0.020883
10   14.0    414  0.004921
11    0.0  47530  0.564912
12   10.0   1361  0.016176
13   18.0    112  0.001331
14    3.0   3556  0.042264
15   16.0    308  0.003661
16   19.0     93  0.001105
17    2.0   4180  0.049681
18    5.0   2863  0.034028
19   15.0    323  0.003839
20   11.0   1237  0.014702
'''

rf = RandomForestClassifier(featuresCol = 'Features', labelCol = 'genre')
multiclass_rf_model = rf.fit(training)
multiclass_rf_predictions = multiclass_rf_model.transform(test)
multiclass_rf_predictions.cache()

print_binary_metrics(multiclass_rf_predictions, "multiclass_rf_predictions", labelCol = 'genre')

''' Output
multiclass_rf_predictions
----------------------------
actual total:    84137
actual positive: 8134
actual negative: 47530
nP:              1211
nN:              82741
TP:              673
FP:              258
FN:              7436
TN:              47238
precision:       0.7228786251342643
recall:          0.08299420397089653
accuracy:        0.5694403175772846
auroc:           0.6827430136571357
'''