## +---------------------------------------------------------------------------------------------------+
## | Recommendations_Q2.py: DATA420 20S2 :: Assignment 2 :: Song Recommendations :: Question 2         |
## |                                                                                                   |
## | __author__ = "Gurpreet Singh"                                                                     |
## | __ID__     = "24701854"                                                                           |
## | __email__  = "gsi58@uclive.ac.nz"                                                                 |
## | __date__   = "22 October ‎2020"                                                                    |
## +---------------------------------------------------------------------------------------------------+

from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

## +-----------------------------------------------------------------------------------------------------+
## | Recommendations 2a: Use the spark.ml library to train an implicit matrix factorization model using  |
## |                     Alternating Least Squares (ALS).                                                |
## +-----------------------------------------------------------------------------------------------------+

# Modeling

als = ALS(maxIter = 5, regParam = 0.01, userCol = "user_id_encoded", itemCol = "song_id_encoded", ratingCol = "plays", implicitPrefs = True)
als_model = als.fit(training)
predictions = als_model.transform(test)

predictions = predictions.orderBy(col("user_id"), col("song_id"), col("prediction").desc())
predictions.cache()

predictions.show(50, False)

''' Output
+------------------+----------------------------------------+-----+---------------+---------------+------------+
|song_id           |user_id                                 |plays|user_id_encoded|song_id_encoded|prediction  |
+------------------+----------------------------------------+-----+---------------+---------------+------------+
|SODHJHX12A58A7D24C|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |309974.0       |462.0          |0.10157579  |
|SOFFWDQ12A8C13B433|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|3    |309974.0       |1953.0         |0.0542689   |
|SOGVKXX12A67ADA0B8|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |309974.0       |321.0          |0.13334659  |
|SOKEYJQ12A6D4F6132|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |309974.0       |209.0          |0.15914854  |
|SOPHBRE12A8C142825|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |309974.0       |917.0          |0.07637477  |
|SOUGLUN12A8C14282A|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |309974.0       |4527.0         |0.021333747 |
|SOUZBUD12A8C13FD8E|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |309974.0       |995.0          |0.03288278  |
|SOWNIUS12A8C142815|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |309974.0       |920.0          |0.07256627  |
|SOWQLXP12AF72A08A2|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |309974.0       |1811.0         |0.053873114 |
|SOYEQLD12AB017C713|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |309974.0       |1210.0         |0.07155317  |
|SOZORGY12A8C140382|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |309974.0       |1062.0         |0.04681293  |
|SOZVCRW12A67ADA0B7|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |309974.0       |27.0           |0.24377096  |
|SOABUZM12A6D4FB8C9|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |26360.0        |2.0695198E-4|
|SOANVMB12AB017F1DD|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |81930.0        |3.1672676E-5|
|SOBJYFB12AB018372D|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |6077.0         |8.7296247E-4|
|SOBWGGV12A6D4FD72E|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |22502.0        |2.4168851E-4|
|SOCRKNT12AB018940D|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |53496.0        |1.0846279E-4|
|SODESWY12AB0182F2E|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |52126.0        |8.9652196E-5|
|SODTGOI12A8C13EBE8|00007ed2509128dcdd74ea3aac2363e24e9dc06b|4    |226363.0       |10227.0        |0.0010492842|
|SOGRNDU12A3F1EB51F|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |226363.0       |90052.0        |4.2267995E-5|
|SOINVHR12AB0189418|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |226363.0       |52830.0        |1.0845581E-4|
|SOJUDEO12A8C13F8E5|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |51134.0        |8.607171E-5 |
|SOKHEEY12A8C1418FE|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |226363.0       |5223.0         |0.0017780694|
|SOMEQZY12A8C1362D4|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |30796.0        |5.1202695E-4|
|SOOGCBL12A8C13FA4E|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |226363.0       |61472.0        |4.8450154E-5|
|SOOHIDI12AB0182EFC|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |53723.0        |5.1874304E-5|
|SOPFVLV12AB0185C5D|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |226363.0       |47760.0        |1.1139272E-4|
|SOQPGMT12AF72A0865|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |47504.0        |9.2068585E-5|
|SORPSOF12AB0188C39|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |226363.0       |47840.0        |1.0401918E-4|
|SOTOAAN12AB0185C68|00007ed2509128dcdd74ea3aac2363e24e9dc06b|3    |226363.0       |29288.0        |2.6114535E-4|
|SOWXPFM12A8C13B2EC|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |46053.0        |9.413181E-5 |
|SOWYFRZ12A6D4FD507|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226363.0       |61931.0        |4.040433E-5 |
|SOYZNPE12A58A79CAD|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |226363.0       |44441.0        |1.2231247E-4|
|SOBXLOE12AF72A43FA|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |15667.0        |6.885141E-4 |
|SOEXJOD12AC9071953|00009d93dc719d1dbaf13507725a03b9fdeebebb|5    |306205.0       |88686.0        |3.597044E-5 |
|SOFRGVQ12A8C1428AA|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |6928.0         |0.0012118306|
|SOJAXPH12AB017FC6F|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |16137.0        |5.765056E-4 |
|SOQEKID12A8AE45CF5|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |84087.0        |3.1035866E-5|
|SORITNY12AB017D909|00009d93dc719d1dbaf13507725a03b9fdeebebb|11   |306205.0       |93766.0        |1.0150099E-4|
|SOTQYXE12A8C13C028|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |18389.0        |4.3835505E-4|
|SOVIYDJ12A8C13BFE2|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |29901.0        |2.6302395E-4|
|SOVTBQI12A8C142ABA|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |65391.0        |3.0274552E-5|
|SOXQGZZ12AB0187A96|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |42278.0        |6.1495026E-5|
|SOYBKUE12A8C13BFEA|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |306205.0       |41232.0        |9.051462E-5 |
|SOZGYIQ12AB01834BF|00009d93dc719d1dbaf13507725a03b9fdeebebb|5    |306205.0       |79592.0        |5.484643E-5 |
|SOCHDNX12A67ADA90D|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|6    |73972.0        |24201.0        |0.002936832 |
|SOCMEJM12AF72A48D0|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |73972.0        |1684.0         |0.01278111  |
|SOCRUVF12A6D4F5906|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73972.0        |3401.0         |0.0051378333|
|SODEZKY12A8C134C4D|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |73972.0        |16953.0        |0.0029738566|
|SODREUL12AB018D6C3|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |73972.0        |5190.0         |0.00476302  |
+------------------+----------------------------------------+-----+---------------+---------------+------------+
only showing top 50 rows
'''




## +-----------------------------------------------------------------------------------------------------+
## | Recommendations 2b: Select a few of the users from the test set by hand and use the model to        |
## |                     generate some recommendations. Compare these recommendations to the songs the   |
## |                     user has actually played. Comment on the effectiveness of the collaborative     |
## |                     filtering model.                                                                |
## +-----------------------------------------------------------------------------------------------------+

# Metrics

k = 100

def extract_songs_top_k(x, k):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x][0:k]

extract_songs_top_k_udf = udf(lambda x: extract_songs_top_k(x, k), ArrayType(IntegerType()))

def extract_songs(x):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x]

extract_songs_udf = udf(lambda x: extract_songs(x), ArrayType(IntegerType()))

users = test.select(als.getUserCol()).distinct().limit(10)
users.cache()
userSubsetRecs = als_model.recommendForUserSubset(users, k)

recommended_songs = (
    userSubsetRecs
    .withColumn("recommended_songs", extract_songs_top_k_udf(col("recommendations")))
    .select("user_id_encoded", "recommended_songs")
)
recommended_songs.cache()
recommended_songs.count()

''' Output
10
'''

recommended_songs.show(10, 100)

''' Output
+---------------+----------------------------------------------------------------------------------------------------+
|user_id_encoded|                                                                                   recommended_songs|
+---------------+----------------------------------------------------------------------------------------------------+
|          72000|[15, 190, 6, 334, 13, 75, 10, 165, 63, 30, 277, 214, 21, 122, 194, 252, 8, 201, 41, 36, 59, 92, 3...|
|          82831|[72, 154, 237, 233, 221, 241, 207, 261, 245, 37, 249, 140, 319, 255, 318, 336, 145, 247, 340, 357...|
|         156994|[1, 4, 3, 19, 14, 0, 8, 6, 10, 31, 24, 13, 107, 29, 2, 44, 23, 55, 35, 90, 40, 43, 972, 57, 22, 9...|
|          10649|[51, 30, 15, 38, 10, 6, 39, 36, 21, 63, 57, 4, 9, 22, 41, 13, 50, 8, 102, 26, 28, 43, 1, 76, 20, ...|
|          27048|[72, 37, 237, 233, 221, 145, 207, 241, 245, 261, 249, 319, 255, 318, 336, 340, 357, 280, 307, 56,...|
|          70275|[16, 95, 172, 21, 52, 43, 20, 107, 124, 45, 478, 9, 14, 19, 50, 74, 33, 40, 4, 246, 88, 892, 111,...|
|          20148|[334, 16, 190, 140, 35, 201, 214, 497, 124, 165, 308, 766, 473, 517, 252, 23, 194, 1025, 376, 578...|
|          39253|[71, 144, 229, 309, 16, 478, 323, 253, 460, 517, 52, 334, 198, 568, 217, 165, 188, 171, 260, 136,...|
|          33849|[37, 145, 72, 25, 147, 187, 56, 7, 73, 265, 32, 11, 136, 71, 100, 112, 5, 135, 207, 683, 48, 75, ...|
|         123599|[190, 334, 277, 201, 252, 165, 189, 214, 194, 278, 63, 229, 153, 254, 200, 15, 262, 193, 366, 131...|
+---------------+----------------------------------------------------------------------------------------------------+
'''

relevant_songs = (
    test
    .select(
        col("user_id_encoded").cast(IntegerType()),
        col("song_id_encoded").cast(IntegerType()),
        col("plays").cast(IntegerType())
    )
    .groupBy('user_id_encoded')
    .agg(
        collect_list(
            array(
                col("song_id_encoded"),
                col("plays")
            )
        ).alias('relevance')
    )
    .withColumn("relevant_songs", extract_songs_udf(col("relevance")))
    .select("user_id_encoded", "relevant_songs")
    .join(users, ['user_id_encoded'], 'inner')
)

relevant_songs = relevant_songs.join(users, ['user_id_encoded'], 'inner')
relevant_songs.cache()
relevant_songs.count()

''' Output
10
'''

relevant_songs.show(10, 100)

''' output
+---------------+----------------------------------------------------------------------------------------------------+
|user_id_encoded|                                                                                      relevant_songs|
+---------------+----------------------------------------------------------------------------------------------------+
|          72000|[83163, 44823, 4357, 39877, 105637, 44589, 12472, 58330, 557, 200, 84319, 62846, 7815, 24821, 883...|
|          82831|[2758, 77976, 10839, 18664, 16590, 24692, 55037, 2297, 4510, 37889, 6246, 48953, 27273, 6794, 315...|
|         156994|[62911, 20942, 63755, 58763, 23695, 21150, 62846, 491, 85752, 60711, 108215, 1012, 6248, 12695, 5...|
|          10649|[1164, 4194, 29463, 13161, 8011, 25954, 62846, 2109, 334, 19502, 88247, 47457, 174, 88840, 14590,...|
|          27048|[20522, 142, 22562, 8704, 47673, 52108, 6497, 31695, 50726, 6183, 24692, 57176, 3875, 90005, 1470...|
|          70275|[1730, 2298, 58186, 21, 14431, 62846, 45828, 2036, 16520, 41534, 15777, 356, 10993, 1301, 28, 0, ...|
|          20148|[109777, 28534, 109291, 46512, 7452, 5278, 1015, 24269, 6701, 17250, 20862, 25343, 22600, 4359, 7...|
|          33849|[25464, 1841, 229, 54, 18479, 112, 3659, 20776, 1422, 31615, 10487, 15926, 19031, 9179, 24565, 14...|
|          39253|[24080, 20116, 33527, 84652, 101735, 93310, 112856, 38769, 22930, 7422, 78484, 24692, 111567, 104...|
|         123599|[38052, 17438, 11078, 2979, 85990, 28560, 1966, 104480, 3344, 1896, 26953, 840, 1751, 66337, 504,...|
+---------------+----------------------------------------------------------------------------------------------------+
'''

combined = (
    recommended_songs.join(relevant_songs, on = 'user_id_encoded', how = 'inner')
    .rdd
    .map(lambda row: (row[1], row[2]))
)
combined.cache()
combined.count()

''' Output
10
'''

combined.take(1)

''' Output
[([15,
   190,
   6,
   334,
   13,
   75,
   10,
   165,
   63,
   30,
   277,
   214,
   21,
   122,
   194,
   252,
   8,
   201,
   41,
   36,
   59,
   92,
   308,
   27,
   38,
   28,
   171,
   198,
   50,
   131,
   173,
   497,
   246,
   473,
   304,
   278,
   309,
   769,
   46,
   376,
   20,
   517,
   258,
   1025,
   262,
   19,
   366,
   43,
   175,
   153],
  [83163,
   44823,
   4357,
   39877,
   105637,
   44589,
   12472,
   58330,
   557,
   200,
   84319,
   62846,
   7815,
   24821,
   88368,
   415,
   20680,
   12893,
   8095,
   7820,
   808,
   3213,
   48738,
   4161,
   3778,
   1420,
   3854,
   2138,
   90932])]
'''


## +-----------------------------------------------------------------------------------------------------+
## | Recommendations 2c: Use the test set of user-song plays and recommendations from the collaborative  |
## |                     filtering model to compute the following metrics                                |
## |                         => Precision @ 5                                                            |
## |                         => NDCG @ 10                                                                |
## |                         => Mean Average Precision (MAP)                                             |
## |                     Look up these metrics and explain why they are useful in evaluating the         |
## |                     collaborate filtering model. Explore the limitations of these metrics in        |
## |                     evaluating a recommendation system in general. Suggest an alternative method for|
## |                     comparing two recommendation systems in the real world.                         |
## |                     Assuming that you could measure future user-song plays based on your            |
## |                     recommendations, what other metrics could be useful?                            |
## +-----------------------------------------------------------------------------------------------------+

ranking_metrics = RankingMetrics(combined)
precision_at_5 = ranking_metrics.precisionAt(5)
print(precision_at_5)

''' output
0.02
'''

ndcg_at_10 = ranking_metrics.ndcgAt(10)
print(ndcg_at_10)

''' Ouput
0.009478836436955077
'''

mean_average_precision = ranking_metrics.meanAveragePrecision
print(mean_average_precision)

''' Output
0.0012113816568791826
'''

precision_at_10 = ranking_metrics.precisionAt(10)
print(precision_at_10)

''' output
0.01
'''