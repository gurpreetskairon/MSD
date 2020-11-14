## +---------------------------------------------------------------------------------------------------+
## | AudioSimilarity_Q1.py: DATA420 20S2 :: Assignment 2 :: Audio Similarity :: Question 1             |
## |                                                                                                   |
## | __author__ = "Gurpreet Singh"                                                                     |
## | __ID__     = "24701854"                                                                           |
## | __email__  = "gsi58@uclive.ac.nz"                                                                 |
## | __date__   = "29 ‎September ‎2020"                                                                  |
## +---------------------------------------------------------------------------------------------------+

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 1 : There are multiple audio feature datasets, with different levels of detail.  |
## |                      Pick one of the small datasets to use for the following.                     |
## +---------------------------------------------------------------------------------------------------+

# find out the dataset that is the smallest one
# hdfs dfs -du -h -v /data/msd/audio/features

''' Output
SIZE     DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
65.5 M   524.2 M                                /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv
53.1 M   424.6 M                                /data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv
35.8 M   286.5 M                                /data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv
70.8 M   566.1 M                                /data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv
51.1 M   408.9 M                                /data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv
51.1 M   408.9 M                                /data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv
412.2 M  3.2 G                                  /data/msd/audio/features/msd-marsyas-timbral-v1.0.csv
1.3 G    10.3 G                                 /data/msd/audio/features/msd-mvd-v1.0.csv
240.3 M  1.9 G                                  /data/msd/audio/features/msd-rh-v1.0.csv
4.0 G    32.3 G                                 /data/msd/audio/features/msd-rp-v1.0.csv
640.6 M  5.0 G                                  /data/msd/audio/features/msd-ssd-v1.0.csv
1.4 G    11.5 G                                 /data/msd/audio/features/msd-trh-v1.0.csv
3.9 G    31.0 G                                 /data/msd/audio/features/msd-tssd-v1.0.csv
'''

# load the dataset# load the filesdata from the 'msd-jmir-methods-of-moments-all-v1.0.csv.gz' /audio/feature file
audio_features = (
    spark.read.format("csv")
    .option("header", "false")
    .option("codec", "gzip")
    .schema(audio_dataset_schemas['msd-jmir-methods-of-moments-all-v1.0'])
    .load("hdfs:////data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/*.csv.gz")
    .cache()
)

# as the track_id field has additonal quotes on either side, we need to remove them to be able to join the audio features dataset later with other datasets using the track_id
audio_features = audio_features.withColumn('track_id', regexp_replace('track_id', '\'', ''))
audio_features.show()

'''output
+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------------+
|feature_0000|feature_0001|feature_0002|feature_0003|feature_0004|feature_0005|feature_0006|feature_0007|feature_0008|feature_0009|          track_id|
+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------------+
|      0.1545|       13.11|       840.0|     41080.0|   7108000.0|       0.319|       33.41|      1371.0|     64240.0|   8398000.0|TRHFHQZ12903C9E2D5|
|      0.1195|       13.02|       611.9|     43880.0|   7226000.0|      0.2661|       30.26|      1829.0|    183800.0|     3.123E7|TRHFHYX12903CAF953|
|      0.2326|       7.185|       362.2|     19890.0|   3030000.0|      0.8854|       32.68|      1384.0|     79190.0|   9862000.0|TRHFHAU128F9341A0E|
|      0.2283|        10.3|       463.8|     24730.0|   3336000.0|      0.4321|       37.56|      2047.0|    197200.0|     3.293E7|TRHFHLP128F14947A7|
|      0.1841|       8.544|       359.4|     21900.0|   3359000.0|      0.8438|       36.36|      2008.0|    205400.0|     3.539E7|TRHFHFF128F930AC11|
|       0.146|       8.248|       519.4|     42300.0|   6138000.0|      0.2782|       19.08|      1052.0|    130900.0|     2.293E7|TRHFHYJ128F4234782|
|     0.09586|       6.915|       409.9|     29840.0|   4691000.0|       0.213|        23.5|      1140.0|     81350.0|     1.012E7|TRHFHHR128F9339010|
|       0.087|       16.37|       560.4|     36280.0|   4264000.0|       0.111|       21.97|       845.8|     49790.0|   6573000.0|TRHFHMB128F4213BC9|
|     0.09981|       7.871|       550.9|     46880.0|   7261000.0|      0.2457|       14.07|       841.5|    104800.0|     1.847E7|TRHFHWT128F429032D|
|       0.239|       15.11|       729.2|     43200.0|   6646000.0|      0.4067|       32.73|      2045.0|    175200.0|     2.872E7|TRHFHKO12903CBAF09|
|      0.3183|       18.72|       693.6|     45850.0|   5547000.0|      0.5555|       36.64|      2017.0|    167600.0|     2.826E7|TRHFHOB128F425F027|
|      0.1389|       23.14|       644.9|     58570.0|   8454000.0|      0.1433|        34.6|      1642.0|    157500.0|     2.708E7|TRHFHTT128E0789A6E|
|      0.2079|       10.58|       501.6|     34330.0|   5127000.0|      0.5036|       29.79|      1793.0|    184600.0|     3.115E7|TRHFHQQ128EF3601ED|
|      0.1702|       10.25|       607.7|     41390.0|   6037000.0|      0.4307|       23.08|      1364.0|    153900.0|     2.652E7|TRHFHOX128F92E6483|
|      0.1579|       6.821|       354.1|     16240.0|   2427000.0|      0.5549|       31.57|      1890.0|    195400.0|     3.238E7|TRHFHSJ128F92EEFFD|
|     0.07216|       3.973|       175.2|      9063.0|   1755000.0|      0.6749|       40.27|      2238.0|    212900.0|     3.532E7|TRHFHVK128F425EA92|
|      0.1646|       8.187|       416.9|     24610.0|   3659000.0|      0.2886|       31.35|      1707.0|    190300.0|     3.238E7|TRHFHRG128F931A920|
|      0.1622|       10.56|       639.0|     41050.0|   6032000.0|      0.3507|       19.07|       936.6|     71050.0|   9020000.0|TRHFHZI128F42ADA12|
|      0.2638|       10.22|       536.2|     30650.0|   4406000.0|       0.639|       34.01|      1399.0|     74340.0|   9336000.0|TRHFHIC128F149297B|
|     0.06773|       5.914|       396.8|     34080.0|   5154000.0|      0.1491|       19.47|      1070.0|    139500.0|     2.501E7|TRHFHQX128F934415F|
+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------------+
only showing top 20 rows
'''

## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 1a: The audio features are continuous values, obtained using methods such as     |
## |                      Load the MSD All Music Genre Dataset (MAGD).                                 |
## |                      Produce descriptive statistics for each feature column in the dataset you    |
## |                      picked. Are any features strongly correlated?                                |
## +---------------------------------------------------------------------------------------------------+

# 
statistics = (
    audio_features
    .select([col for col in audio_features.columns if col.startswith("feature_")])
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

'''output
               count                 mean               stddev        min       max
feature_0000  994623  0.15498176001746342  0.06646213086143025        0.0     0.959
feature_0001  994623   10.384550576952307    3.868001393874683        0.0     55.42
feature_0002  994623    526.8139724398096    180.4377549977526        0.0    2919.0
feature_0003  994623    35071.97543290272    12806.81627295556        0.0  407100.0
feature_0004  994623    5297870.369577217   2089356.4364558063        0.0   4.657E7
feature_0005  994623   0.3508444432531317  0.18557956834383815        0.0     2.647
feature_0006  994623   27.463867987840707    8.352648595163766        0.0     117.0
feature_0007  994623   1495.8091812075545   505.89376391902306        0.0    5834.0
feature_0008  994623   143165.46163257834   50494.276171032274  -146300.0  452500.0
feature_0009  994623  2.396783048473542E7    9307340.299219666        0.0   9.477E7
'''

# find out the correlations between the features
assembler = VectorAssembler(
    inputCols=[col for col in audio_features.columns if col.startswith("feature_")],
    outputCol="Features"
)
features = assembler.transform(audio_features).select(["Features", "track_id"])
features.cache()
features.show(truncate = False)

'''Output
+----------------------------------------------------------------------------+------------------+
|Features                                                                    |track_id          |
+----------------------------------------------------------------------------+------------------+
|[0.1545,13.11,840.0,41080.0,7108000.0,0.319,33.41,1371.0,64240.0,8398000.0] |TRHFHQZ12903C9E2D5|
|[0.1195,13.02,611.9,43880.0,7226000.0,0.2661,30.26,1829.0,183800.0,3.123E7] |TRHFHYX12903CAF953|
|[0.2326,7.185,362.2,19890.0,3030000.0,0.8854,32.68,1384.0,79190.0,9862000.0]|TRHFHAU128F9341A0E|
|[0.2283,10.3,463.8,24730.0,3336000.0,0.4321,37.56,2047.0,197200.0,3.293E7]  |TRHFHLP128F14947A7|
|[0.1841,8.544,359.4,21900.0,3359000.0,0.8438,36.36,2008.0,205400.0,3.539E7] |TRHFHFF128F930AC11|
|[0.146,8.248,519.4,42300.0,6138000.0,0.2782,19.08,1052.0,130900.0,2.293E7]  |TRHFHYJ128F4234782|
|[0.09586,6.915,409.9,29840.0,4691000.0,0.213,23.5,1140.0,81350.0,1.012E7]   |TRHFHHR128F9339010|
|[0.087,16.37,560.4,36280.0,4264000.0,0.111,21.97,845.8,49790.0,6573000.0]   |TRHFHMB128F4213BC9|
|[0.09981,7.871,550.9,46880.0,7261000.0,0.2457,14.07,841.5,104800.0,1.847E7] |TRHFHWT128F429032D|
|[0.239,15.11,729.2,43200.0,6646000.0,0.4067,32.73,2045.0,175200.0,2.872E7]  |TRHFHKO12903CBAF09|
|[0.3183,18.72,693.6,45850.0,5547000.0,0.5555,36.64,2017.0,167600.0,2.826E7] |TRHFHOB128F425F027|
|[0.1389,23.14,644.9,58570.0,8454000.0,0.1433,34.6,1642.0,157500.0,2.708E7]  |TRHFHTT128E0789A6E|
|[0.2079,10.58,501.6,34330.0,5127000.0,0.5036,29.79,1793.0,184600.0,3.115E7] |TRHFHQQ128EF3601ED|
|[0.1702,10.25,607.7,41390.0,6037000.0,0.4307,23.08,1364.0,153900.0,2.652E7] |TRHFHOX128F92E6483|
|[0.1579,6.821,354.1,16240.0,2427000.0,0.5549,31.57,1890.0,195400.0,3.238E7] |TRHFHSJ128F92EEFFD|
|[0.07216,3.973,175.2,9063.0,1755000.0,0.6749,40.27,2238.0,212900.0,3.532E7] |TRHFHVK128F425EA92|
|[0.1646,8.187,416.9,24610.0,3659000.0,0.2886,31.35,1707.0,190300.0,3.238E7] |TRHFHRG128F931A920|
|[0.1622,10.56,639.0,41050.0,6032000.0,0.3507,19.07,936.6,71050.0,9020000.0] |TRHFHZI128F42ADA12|
|[0.2638,10.22,536.2,30650.0,4406000.0,0.639,34.01,1399.0,74340.0,9336000.0] |TRHFHIC128F149297B|
|[0.06773,5.914,396.8,34080.0,5154000.0,0.1491,19.47,1070.0,139500.0,2.501E7]|TRHFHQX128F934415F|
+----------------------------------------------------------------------------+------------------+
only showing top 20 rows
'''

correlations = Correlation.corr(features, 'Features', 'pearson').collect()[0][0].toArray()
print(correlations)

'''Output
[[ 1.          0.42628035  0.29630589  0.06103865 -0.05533585  0.75420787
   0.49792898  0.44756461  0.16746557  0.10040744]
 [ 0.42628035  1.          0.85754866  0.60952091  0.43379677  0.02522827
   0.40692287  0.39635353  0.01560657 -0.04090215]
 [ 0.29630589  0.85754866  1.          0.80300965  0.68290935 -0.08241507
   0.12591025  0.18496247 -0.08817391 -0.13505636]
 [ 0.06103865  0.60952091  0.80300965  1.          0.94224443 -0.3276915
  -0.22321966 -0.15823074 -0.24503392 -0.22087303]
 [-0.05533585  0.43379677  0.68290935  0.94224443  1.         -0.39255125
  -0.35501874 -0.28596556 -0.26019779 -0.21181281]
 [ 0.75420787  0.02522827 -0.08241507 -0.3276915  -0.39255125  1.
   0.54901522  0.5185027   0.34711201  0.2785128 ]
 [ 0.49792898  0.40692287  0.12591025 -0.22321966 -0.35501874  0.54901522
   1.          0.90336675  0.51649906  0.4225494 ]
 [ 0.44756461  0.39635353  0.18496247 -0.15823074 -0.28596556  0.5185027
   0.90336675  1.          0.7728069   0.68564528]
 [ 0.16746557  0.01560657 -0.08817391 -0.24503392 -0.26019779  0.34711201
   0.51649906  0.7728069   1.          0.9848665 ]
 [ 0.10040744 -0.04090215 -0.13505636 -0.22087303 -0.21181281  0.2785128
   0.4225494   0.68564528  0.9848665   1.        ]]
'''

#List the correlation values as pairs 
for i in range(0, correlations.shape[0]):
    for j in range(i + 1, correlations.shape[1]):
        if correlations[i, j] > 0.5:
            print((i, j), correlations[i, j])

'''Output
(0, 5) 0.7542078681244755
(1, 2) 0.8575486565280668
(1, 3) 0.6095209087098785
(2, 3) 0.8030096521045835
(2, 4) 0.6829093525810729
(3, 4) 0.9422444252761963
(5, 6) 0.5490152221867522
(5, 7) 0.5185026975022888
(6, 7) 0.9033667462435663
(6, 8) 0.5164990583386109
(7, 8) 0.7728068953919849
(7, 9) 0.6856452835773446
(8, 9) 0.9848665037801265

'''


## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 1b: Load the MSD All Music Genre Dataset (MAGD).                                 |
## |                      Visualize the distribution of genres for the songs that were matched.        |
## +---------------------------------------------------------------------------------------------------+

genre_schema = StructType([
    StructField("track_id", StringType(), True),
    StructField("genre_type", StringType(), True)
])

genre = (
    spark.read.format("com.databricks.spark.csv")
    .option("delimiter", "\t")
    .option("header", "false")
    .schema(genre_schema)
    .load("hdfs:////data/msd/genre/msd-MAGD-genreAssignment.tsv")
)

genre.show()

'''output
+------------------+--------------+
|          track_id|    genre_type|
+------------------+--------------+
|TRAAAAK128F9318786|      Pop_Rock|
|TRAAAAV128F421A322|      Pop_Rock|
|TRAAAAW128F429D538|           Rap|
|TRAAABD128F429CF47|      Pop_Rock|
|TRAAACV128F423E09E|      Pop_Rock|
|TRAAADT12903CCC339|Easy_Listening|
|TRAAAED128E0783FAB|         Vocal|
|TRAAAEF128F4273421|      Pop_Rock|
|TRAAAEM128F93347B9|    Electronic|
|TRAAAFD128F92F423A|      Pop_Rock|
|TRAAAFP128F931B4E3|           Rap|
|TRAAAGR128F425B14B|      Pop_Rock|
|TRAAAGW12903CC1049|         Blues|
|TRAAAHD128F42635A5|      Pop_Rock|
|TRAAAHE12903C9669C|      Pop_Rock|
|TRAAAHJ128F931194C|      Pop_Rock|
|TRAAAHZ128E0799171|           Rap|
|TRAAAIR128F1480971|           RnB|
|TRAAAJG128F9308A25|          Folk|
|TRAAAMO128F1481E7F|     Religious|
+------------------+--------------+
only showing top 20 rows
'''

# group the data by genbre_type and look at the count of records under each type
genre.groupBy('genre_type').count().show(21, truncate = False)

''' Output
+--------------+------+
|genre_type    |count |
+--------------+------+
|Stage         |1614  |
|Vocal         |6195  |
|Religious     |8814  |
|Easy_Listening|1545  |
|Jazz          |17836 |
|Electronic    |41075 |
|Blues         |6836  |
|International |14242 |
|Children      |477   |
|Rap           |20939 |
|RnB           |14335 |
|Avant_Garde   |1014  |
|Latin         |17590 |
|Folk          |5865  |
|Pop_Rock      |238786|
|Classical     |556   |
|New Age       |4010  |
|Country       |11772 |
|Comedy_Spoken |2067  |
|Reggae        |6946  |
|Holiday       |200   |
+--------------+------+
'''

genre_fig = genre.groupBy("genre_type").count().toPandas().plot.bar(x = "genre_type", y = "count", color = {"green"}).get_figure()
genre_fig.savefig("genre_type.jpg")

## +---------------------------------------------------------------------------------------------------+
## | Audio Similarity 1c: Merge the genres dataset and the audio features dataset so that every song   |
## |                      has a label.                                                                 |
## +---------------------------------------------------------------------------------------------------+


audio_features_genre = audio_features.join(genre, on = 'track_id', how = 'inner')
audio_features_genre.show()

'''Output
+------------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+-------------+
|          track_id|feature_0000|feature_0001|feature_0002|feature_0003|feature_0004|feature_0005|feature_0006|feature_0007|feature_0008|feature_0009|   genre_type|
+------------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+-------------+
|TRAAABD128F429CF47|      0.1308|       9.587|       459.9|     27280.0|   4303000.0|      0.2474|       26.02|      1067.0|     67790.0|   8281000.0|     Pop_Rock|
|TRAABPK128F424CFDB|      0.1208|       6.738|       215.1|     11890.0|   2278000.0|      0.4882|       41.76|      2164.0|    220400.0|      3.79E7|     Pop_Rock|
|TRAACER128F4290F96|      0.2838|       8.995|       429.5|     31990.0|   5272000.0|      0.5388|       28.29|      1656.0|    185100.0|     3.164E7|     Pop_Rock|
|TRAADYB128F92D7E73|      0.1346|       7.321|       499.6|     38460.0|   5877000.0|      0.2839|       15.75|       929.6|    116500.0|     2.058E7|         Jazz|
|TRAAGHM128EF35CF8E|      0.1563|       9.959|       502.8|     26190.0|   3660000.0|      0.3835|       28.24|      1864.0|    180800.0|     2.892E7|   Electronic|
|TRAAGRV128F93526C0|      0.1076|       7.401|       389.7|     19350.0|   2739000.0|      0.4221|       30.99|      1861.0|    191700.0|     3.166E7|     Pop_Rock|
|TRAAGTO128F1497E3C|      0.1069|       8.987|       562.6|     43100.0|   7057000.0|      0.1007|        22.9|      1346.0|    157700.0|     2.738E7|     Pop_Rock|
|TRAAHAU128F9313A3D|     0.08485|       9.031|       445.9|     23750.0|   3274000.0|      0.2583|       35.59|      2015.0|    198400.0|     3.336E7|     Pop_Rock|
|TRAAHEG128E07861C3|      0.1699|       17.22|       741.3|     52440.0|   8275000.0|      0.2812|       28.83|      1671.0|    160800.0|     2.695E7|          Rap|
|TRAAHZP12903CA25F4|      0.1654|       12.31|       565.1|     33100.0|   5273000.0|      0.1861|       38.38|      1962.0|    196600.0|     3.355E7|          Rap|
|TRAAICW128F1496C68|      0.1104|       7.123|       398.2|     19540.0|   3240000.0|      0.2871|       28.53|      1807.0|    189400.0|     3.156E7|International|
|TRAAJJW12903CBDDCB|      0.2267|       14.88|       592.7|     37980.0|   4569000.0|      0.4219|       36.17|      2111.0|    179400.0|     2.952E7|International|
|TRAAJKJ128F92FB44F|     0.03861|        6.87|       407.8|     41310.0|   7299000.0|      0.0466|       15.79|       955.1|    121700.0|     2.124E7|         Folk|
|TRAAKLX128F934CEE4|      0.1647|       16.77|       850.0|     64420.0|     1.011E7|      0.2823|       26.52|      1600.0|    152000.0|     2.587E7|   Electronic|
|TRAAKWR128F931B29F|     0.04881|       9.331|       564.0|     34410.0|   4920000.0|     0.08647|        18.1|       880.5|     57700.0|   6429000.0|     Pop_Rock|
|TRAALQN128E07931A4|      0.1989|       12.83|       578.7|     30690.0|   4921000.0|      0.5452|       33.37|      2019.0|    188700.0|      3.14E7|   Electronic|
|TRAAMFF12903CE8107|      0.1385|       9.699|       581.6|     31590.0|   4569000.0|      0.3706|       23.63|      1554.0|    163800.0|      2.72E7|     Pop_Rock|
|TRAAMHG128F92ED7B2|      0.1799|       10.52|       551.4|     29170.0|   4396000.0|      0.4046|       30.78|      1806.0|    183200.0|     3.059E7|International|
|TRAAROH128F42604B0|      0.1192|        16.4|       737.3|     41670.0|   6295000.0|      0.2284|       31.04|      1878.0|    169100.0|     2.829E7|   Electronic|
|TRAARQN128E07894DF|      0.2559|       15.23|       757.1|     61750.0|     1.065E7|      0.5417|       40.96|      2215.0|    189000.0|      3.21E7|     Pop_Rock|
+------------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+------------+-------------+
only showing top 20 rows
'''