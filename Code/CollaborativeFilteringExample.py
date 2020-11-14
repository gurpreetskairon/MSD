# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Python and pyspark modules required

import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.sql import functions as F


# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Compute suitable number of partitions

conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------

mismatches_schema = StructType([
    StructField("song_id", StringType(), True),
    StructField("song_artist", StringType(), True),
    StructField("song_title", StringType(), True),
    StructField("track_id", StringType(), True),
    StructField("track_artist", StringType(), True),
    StructField("track_title", StringType(), True)
])

with open("/scratch-network/courses/2020/DATA420-20S2/data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt", "r") as f:
    lines = f.readlines()
    sid_matches_manually_accepted = []
    for line in lines:
        if line.startswith("< ERROR: "):
            a = line[10:28]
            b = line[29:47]
            c, d = line[49:-1].split("  !=  ")
            e, f = c.split("  -  ")
            g, h = d.split("  -  ")
            sid_matches_manually_accepted.append((a, e, f, b, g, h))

matches_manually_accepted = spark.createDataFrame(sc.parallelize(sid_matches_manually_accepted, 8), schema=mismatches_schema)
matches_manually_accepted.cache()
matches_manually_accepted.show(10, 40)

print(matches_manually_accepted.count())  # 488

with open("/scratch-network/courses/2020/DATA420-20S2/data/msd/tasteprofile/mismatches/sid_mismatches.txt", "r") as f:
    lines = f.readlines()
    sid_mismatches = []
    for line in lines:
        if line.startswith("ERROR: "):
            a = line[8:26]
            b = line[27:45]
            c, d = line[47:-1].split("  !=  ")
            e, f = c.split("  -  ")
            g, h = d.split("  -  ")
            sid_mismatches.append((a, e, f, b, g, h))

mismatches = spark.createDataFrame(sc.parallelize(sid_mismatches, 64), schema=mismatches_schema)
mismatches.cache()
mismatches.show(10, 40)

print(mismatches.count())  # 19094

triplets_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("song_id", StringType(), True),
    StructField("plays", IntegerType(), True)
])
triplets = (
    spark.read.format("csv")
    .option("header", "false")
    .option("delimiter", "\t")
    .option("codec", "gzip")
    .schema(triplets_schema)
    .load("hdfs:///data/msd/tasteprofile/triplets.tsv/")
    .cache()
)
triplets.cache()
triplets.show(10, 50)

mismatches_not_accepted = mismatches.join(matches_manually_accepted, on="song_id", how="left_anti")
triplets_not_mismatched = triplets.join(mismatches_not_accepted, on="song_id", how="left_anti")

triplets_not_mismatched = triplets_not_mismatched.repartition(partitions).cache()

print(mismatches_not_accepted.count())  # 19093
print(triplets.count())                 # 48373586
print(triplets_not_mismatched.count())  # 45795111


# -----------------------------------------------------------------------------
# Data analysis
# -----------------------------------------------------------------------------

def get_user_counts(triplets):
    return (
        triplets
        .groupBy("user_id")
        .agg(
            F.count(col("song_id")).alias("song_count"),
            F.sum(col("plays")).alias("play_count"),
        )
        .orderBy(col("play_count").desc())
    )

def get_song_counts(triplets):
    return (
        triplets
        .groupBy("song_id")
        .agg(
            F.count(col("user_id")).alias("user_count"),
            F.sum(col("plays")).alias("play_count"),
        )
        .orderBy(col("play_count").desc())
    )

# User statistics

user_counts = (
    triplets_not_mismatched
    .groupBy("user_id")
    .agg(
        F.count(col("song_id")).alias("song_count"),
        F.sum(col("plays")).alias("play_count"),
    )
    .orderBy(col("play_count").desc())
)
user_counts.cache()
user_counts.count()

# 1019318

user_counts.show(10, False)

# +----------------------------------------+----------+----------+
# |user_id                                 |song_count|play_count|
# +----------------------------------------+----------+----------+
# |093cb74eb3c517c5179ae24caf0ebec51b24d2a2|195       |13074     |
# |119b7c88d58d0c6eb051365c103da5caf817bea6|1362      |9104      |
# |3fa44653315697f42410a30cb766a4eb102080bb|146       |8025      |
# |a2679496cd0af9779a92a13ff7c6af5c81ea8c7b|518       |6506      |
# |d7d2d888ae04d16e994d6964214a1de81392ee04|1257      |6190      |
# |4ae01afa8f2430ea0704d502bc7b57fb52164882|453       |6153      |
# |b7c24f770be6b802805ac0e2106624a517643c17|1364      |5827      |
# |113255a012b2affeab62607563d03fbdf31b08e7|1096      |5471      |
# |99ac3d883681e21ea68071019dba828ce76fe94d|939       |5385      |
# |6d625c6557df84b60d90426c0116138b617b9449|1307      |5362      |
# +----------------------------------------+----------+----------+

statistics = (
    user_counts
    .select("song_count", "play_count")
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

#               count                mean              stddev min    max
# song_count  1019318   44.92720721109605   54.91113199747355   3   4316
# play_count  1019318  128.82423149596102  175.43956510304616   3  13074

user_counts.approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
user_counts.approxQuantile("play_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)

# [3.0, 20.0, 32.0,  58.0,  4316.0]
# [3.0, 35.0, 71.0, 173.0, 13074.0]

# Song statistics

song_counts = (
    triplets_not_mismatched
    .groupBy("song_id")
    .agg(
        F.count(col("user_id")).alias("user_count"),
        F.sum(col("plays")).alias("play_count"),
    )
    .orderBy(col("play_count").desc())
)
song_counts.cache()
song_counts.count()

# 378310

song_counts.show(10, False)

# +------------------+----------+----------+
# |song_id           |song_count|play_count|
# +------------------+----------+----------+
# |SOBONKR12A58A7A7E0|84000     |726885    |
# |SOSXLTC12AF72A7F54|80656     |527893    |
# |SOEGIYH12A6D4FC0E3|69487     |389880    |
# |SOAXGDH12A8C13F8A1|90444     |356533    |
# |SONYKOW12AB01849C9|78353     |292642    |
# |SOPUCYA12A8C13A694|46078     |274627    |
# |SOUFTBI12AB0183F65|37642     |268353    |
# |SOVDSJC12A58A7A271|36976     |244730    |
# |SOOFYTN12A6D4F9B35|40403     |241669    |
# |SOHTKMO12AB01843B0|46077     |236494    |
# +------------------+----------+----------+

statistics = (
    song_counts
    .select("user_count", "play_count")
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

#              count                mean             stddev min     max
# user_count  378310  121.05181200602681  748.6489783736941   1   90444
# play_count  378310   347.1038513388491  2978.605348838212   1  726885

song_counts.approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)
song_counts.approxQuantile("play_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)

# [1.0,  5.0, 13.0, 104.0,  90444.0]
# [1.0, 11.0, 36.0, 217.0, 726885.0]


# -----------------------------------------------------------------------------
# Limiting
# -----------------------------------------------------------------------------

user_song_count_threshold = 10
song_user_count_threshold = 5

triplets_limited = triplets_not_mismatched

for i in range(0, 10):

    triplets_limited = (
        triplets_limited
        .join(
            triplets_limited.groupBy("user_id").count().where(col("count") > user_song_count_threshold).select("user_id"),
            on="user_id",
            how="inner"
        )
    )

    triplets_limited = (
        triplets_limited
        .join(
            triplets_limited.groupBy("song_id").count().where(col("count") > user_song_count_threshold).select("song_id"),
            on="song_id",
            how="inner"
        )
    )

triplets_limited.cache()
triplets_limited.count()

# 44220298

(
    triplets_limited
    .agg(
        countDistinct(col("user_id")).alias('user_count'),
        countDistinct(col("song_id")).alias('song_count')
    )
    .toPandas()
    .T
    .rename(columns={0: "value"})
)

#              value
# user_count  932143 / 1019318 = 0.9216
# song_count  206076 /  378310 = 0.6967

print(get_user_counts(triplets_limited).approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))
print(get_song_counts(triplets_limited).approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05))

# [ 7.0, 23.0, 34.0,  61.0,  3839.0]
# [11.0, 26.0, 53.0, 169.0, 86546.0]


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# Imports

from pyspark.ml.feature import StringIndexer

# Encoding

user_id_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_encoded")
song_id_indexer = StringIndexer(inputCol="song_id", outputCol="song_id_encoded")

user_id_indexer_model = user_id_indexer.fit(triplets_limited)
song_id_indexer_model = song_id_indexer.fit(triplets_limited)

triplets_limited = user_id_indexer_model.transform(triplets_limited)
triplets_limited = song_id_indexer_model.transform(triplets_limited)


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------

# Imports

from pyspark.sql.window import *

# Splits

training, test = triplets_limited.randomSplit([0.7, 0.3])

test_not_training = test.join(training, on="user_id", how="left_anti")

training.cache()
test.cache()
test_not_training.cache()

print(f"training:          {training.count()}")
print(f"test:              {test.count()}")
print(f"test_not_training: {test_not_training.count()}")

# training:          32055764
# test:              13739347
# test_not_training: 16

# training:          30949623
# test:              13270675
# test_not_training: 0

test_not_training.show(50, False)

# +----------------------------------------+------------------+-----+
# |user_id                                 |song_id           |plays|
# +----------------------------------------+------------------+-----+
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOIJKRX12AB0185397|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOGZBUE12AB0187364|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SORMOAU12AB018956B|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOSHJHA12AB0181410|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOEPWYH12AF72A4813|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SODQXUK12AF72A13E5|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOOBZMH12AB0187399|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOENDMN12A58A78493|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOGWUHI12AB01876BD|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOUOPLF12AB017F40F|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOEQJBS12A8AE475A4|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOAAVUV12AB0186646|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOOYDAZ12A58A7AE08|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOTWCDE12AB018909C|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOAFTRR12AF72A8D4D|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOARUPP12AB01842E0|1    |
# +----------------------------------------+------------------+-----+

counts = test_not_training.groupBy("user_id").count().toPandas().set_index("user_id")["count"].to_dict()

temp = (
    test_not_training
    .withColumn("id", monotonically_increasing_id())
    .withColumn("random", rand())
    .withColumn(
        "row",
        row_number()
        .over(
            Window
            .partitionBy("user_id")
            .orderBy("random")
        )
    )
)

for k, v in counts.items():
    temp = temp.where((col("user_id") != k) | (col("row") < v * 0.7))

temp = temp.drop("id", "random", "row")
temp.cache()

temp.show(50, False)

# +----------------------------------------+------------------+-----+
# |user_id                                 |song_id           |plays|
# +----------------------------------------+------------------+-----+
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOENDMN12A58A78493|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOEPWYH12AF72A4813|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SOSHJHA12AB0181410|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SODQXUK12AF72A13E5|1    |
# |a218fef82a857225ae5fcce5db0ec2ac96851cc2|SORMOAU12AB018956B|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOOYDAZ12A58A7AE08|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOARUPP12AB01842E0|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOAAVUV12AB0186646|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOUOPLF12AB017F40F|1    |
# |42830ed368d1c29396791f0cb1c1bb871f8af06f|SOGWUHI12AB01876BD|1    |
# +----------------------------------------+------------------+-----+

training = training.union(temp.select(training.columns))
test = test.join(temp, on=["user_id", "song_id"], how="left_anti")
test_not_training = test.join(training, on="user_id", how="left_anti")

print(f"training:          {training.count()}")
print(f"test:              {test.count()}")
print(f"test_not_training: {test_not_training.count()}")

# training:          32055774
# test:              13739337
# test_not_training: 0


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

# Imports

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer

from pyspark.mllib.evaluation import RankingMetrics

# Modeling

als = ALS(maxIter=5, regParam=0.01, userCol="user_id_encoded", itemCol="song_id_encoded", ratingCol="plays")
als_model = als.fit(training)
predictions = als_model.transform(test)

predictions = predictions.orderBy(col("user_id"), col("song_id"), col("prediction").desc())
predictions.cache()

predictions.show(50, False)

# +------------------+----------------------------------------+-----+---------------+---------------+-----------+
# |song_id           |user_id                                 |plays|user_id_encoded|song_id_encoded|prediction |
# +------------------+----------------------------------------+-----+---------------+---------------+-----------+
# |SORDKNX12A8C13A45F|00000b722001882066dff9d2da8a775658053ea0|1    |856763.0       |50622.0        |0.63414586 |
# |SOBFEDK12A8C13BB25|00001638d6189236866af9bbf309ae6c2347ffdc|1    |859779.0       |17821.0        |-1.0087988 |
# |SOLOYFG12A8C133391|00001638d6189236866af9bbf309ae6c2347ffdc|1    |859779.0       |19812.0        |-0.74704367|
# |SOOEPEG12A6D4FC7CA|00001638d6189236866af9bbf309ae6c2347ffdc|1    |859779.0       |4703.0         |-0.5360813 |
# |SOWOTHK12A67AD818B|00001638d6189236866af9bbf309ae6c2347ffdc|24   |859779.0       |192657.0       |0.38297927 |
# |SOGTCXJ12A6D4F7076|0000175652312d12576d9e6b84f600caa24c4715|1    |858042.0       |54073.0        |0.16485938 |
# |SOPOLHW12A6D4F7DC4|0000175652312d12576d9e6b84f600caa24c4715|1    |858042.0       |19090.0        |-4.790461  |
# |SOPZAEV12A6D4FAD60|0000175652312d12576d9e6b84f600caa24c4715|4    |858042.0       |76817.0        |7.4432535  |
# |SOBDRND12A8C13FD08|00001cf0dce3fb22b0df0f3a1d9cd21e38385372|1    |684009.0       |2693.0         |1.9837594  |
# |SOGBGBT12AB01809B3|00001cf0dce3fb22b0df0f3a1d9cd21e38385372|1    |684009.0       |4385.0         |1.9319016  |
# |SOLVTNS12AB01809E2|00001cf0dce3fb22b0df0f3a1d9cd21e38385372|2    |684009.0       |5764.0         |1.3183483  |
# |SOPEVJE12A67ADE837|00001cf0dce3fb22b0df0f3a1d9cd21e38385372|1    |684009.0       |6712.0         |1.0873957  |
# |SOVYIYI12A8C138D88|00001cf0dce3fb22b0df0f3a1d9cd21e38385372|1    |684009.0       |1177.0         |1.5493891  |
# |SOWUCFL12AB0188263|00001cf0dce3fb22b0df0f3a1d9cd21e38385372|2    |684009.0       |8660.0         |1.3922336  |
# |SOWZVSV12A8C13BBCD|00001cf0dce3fb22b0df0f3a1d9cd21e38385372|1    |684009.0       |6737.0         |3.0033994  |
# |SOZFJYG12AB0182D24|00001cf0dce3fb22b0df0f3a1d9cd21e38385372|2    |684009.0       |4168.0         |2.5276556  |
# |SOBMSCQ12AAF3B51B7|0000267bde1b3a70ea75cf2b2d216cb828e3202b|1    |549536.0       |1722.0         |1.4236747  |
# |SOCJCVE12A8C13CDDB|0000267bde1b3a70ea75cf2b2d216cb828e3202b|1    |549536.0       |412.0          |1.403591   |
# |SOQLDTI12AB018C80A|0000267bde1b3a70ea75cf2b2d216cb828e3202b|1    |549536.0       |2834.0         |0.76024556 |
# |SORFFOI12A8C135E10|0000267bde1b3a70ea75cf2b2d216cb828e3202b|1    |549536.0       |2969.0         |1.0495535  |
# |SORHJAS12AB0187D3F|0000267bde1b3a70ea75cf2b2d216cb828e3202b|1    |549536.0       |1047.0         |1.1478919  |
# |SOTDKEJ12AB0187AAA|0000267bde1b3a70ea75cf2b2d216cb828e3202b|2    |549536.0       |432.0          |1.2609713  |
# |SOVJFSL12A58A7F6A4|0000267bde1b3a70ea75cf2b2d216cb828e3202b|1    |549536.0       |3809.0         |0.93499345 |
# |SOWMGHQ12A6D4F914D|0000267bde1b3a70ea75cf2b2d216cb828e3202b|1    |549536.0       |674.0          |1.0292748  |
# |SOJJRVI12A6D4FBE49|00003a4459f33b92906be11abe0e93efc423c0ff|1    |783860.0       |2943.0         |9.239005   |
# |SOMZIQI12AB017F9B8|00003a4459f33b92906be11abe0e93efc423c0ff|1    |783860.0       |109545.0       |14.688843  |
# |SOWVBDQ12A8C13503D|00003a4459f33b92906be11abe0e93efc423c0ff|3    |783860.0       |280.0          |1.6913253  |
# |SOBJCFV12A8AE469EE|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |2205.0         |1.2159114  |
# |SOGVKXX12A67ADA0B8|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |406.0          |1.4150923  |
# |SOGXWGC12AF72A8F9A|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |2016.0         |0.9945667  |
# |SOJGSIO12A8C141DBF|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |333.0          |2.4721544  |
# |SOKEYJQ12A6D4F6132|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |233.0          |1.7931228  |
# |SORRCNC12A8C13FDA9|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |328.0          |1.5598544  |
# |SOWCKVR12A8C142411|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |8.0            |1.301859   |
# |SOWGIBZ12A8C136A2E|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|3    |322305.0       |644.0          |1.9280285  |
# |SOYWTUB12A8C13B429|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |1553.0         |1.6583574  |
# |SOZAPQT12A8C142821|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |322305.0       |505.0          |2.1582599  |
# |SOPXBBT12A6D4F7610|00005c6177188f12fb5e2e82cdbd93e8a3f35e64|1    |769896.0       |11856.0        |0.6139259  |
# |SOVATGC12A6D4F9393|00005c6177188f12fb5e2e82cdbd93e8a3f35e64|1    |769896.0       |11199.0        |2.6146474  |
# |SOCGHCY12A58A7C997|000060ca4e6bea0a5c9037fc1bbd7bbabb98c754|1    |412720.0       |14477.0        |1.1390481  |
# |SOHLBZD12A58A7B0AC|000060ca4e6bea0a5c9037fc1bbd7bbabb98c754|1    |412720.0       |5250.0         |1.9614596  |
# |SOILWTV12A6D4F4A4B|000060ca4e6bea0a5c9037fc1bbd7bbabb98c754|1    |412720.0       |6276.0         |1.9676834  |
# |SOKXYIL12AB0189157|000060ca4e6bea0a5c9037fc1bbd7bbabb98c754|2    |412720.0       |7163.0         |0.3728093  |
# |SONPCAF12A81C21DE9|000060ca4e6bea0a5c9037fc1bbd7bbabb98c754|1    |412720.0       |9962.0         |3.560175   |
# |SOURLCY12A8AE47C09|000060ca4e6bea0a5c9037fc1bbd7bbabb98c754|1    |412720.0       |8744.0         |-2.7301188 |
# |SOYOKCE12A58A79862|000060ca4e6bea0a5c9037fc1bbd7bbabb98c754|1    |412720.0       |1572.0         |1.1706172  |
# |SOYUXZJ12A6D4F86F1|000060ca4e6bea0a5c9037fc1bbd7bbabb98c754|1    |412720.0       |5162.0         |1.1354784  |
# |SOABUZM12A6D4FB8C9|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226920.0       |28405.0        |0.5969076  |
# |SOANVMB12AB017F1DD|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226920.0       |88009.0        |0.674686   |
# |SOBJYFB12AB018372D|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |226920.0       |6888.0         |0.6636224  |
# +------------------+----------------------------------------+-----+---------------+---------------+-----------+


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

# Helpers 

def extract_songs_top_k(x, k):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x][0:k]

extract_songs_top_k_udf = udf(lambda x: extract_songs(x, k), ArrayType(IntegerType()))

def extract_songs(x):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x]

extract_songs_udf = udf(lambda x: extract_songs(x), ArrayType(IntegerType()))

# Recommendations

k = 5

topK = als_model.recommendForAllUsers(k)

topK.cache()
topK.count()

# 932143

topK.show(10, False)

# +---------------+---------------------------------------------------------------------------------------------------------+
# |user_id_encoded|recommendations                                                                                          |
# +---------------+---------------------------------------------------------------------------------------------------------+
# |38             |[[107048, 158.95903], [127769, 143.0364], [129688, 142.07109], [113295, 137.7817], [145331, 136.7576]]   |
# |273            |[[199021, 109.92833], [129688, 78.29608], [91827, 61.817234], [39034, 60.938713], [145331, 60.214012]]   |
# |300            |[[129688, 116.0324], [199021, 96.14659], [105961, 79.75125], [145331, 68.81387], [107048, 65.02151]]     |
# |412            |[[190031, 139.41019], [129688, 96.57117], [71670, 91.82175], [145331, 88.915184], [199021, 85.03826]]    |
# |434            |[[199021, 136.32993], [129688, 120.93792], [91827, 71.08011], [66074, 70.87034], [202418, 63.416588]]    |
# |475            |[[199021, 120.76336], [129688, 84.35693], [66074, 65.04297], [107048, 64.911736], [71670, 61.463634]]    |
# |585            |[[129688, 139.38734], [199021, 137.85187], [68593, 131.06927], [91827, 95.74485], [156831, 79.06015]]    |
# |600            |[[129688, 301.28632], [105961, 268.08093], [199021, 244.42038], [145331, 212.10342], [107048, 198.99654]]|
# |611            |[[43631, 131.66966], [199021, 128.25261], [107048, 106.84018], [153363, 99.85662], [105961, 98.22813]]   |
# |619            |[[199021, 96.66749], [129688, 84.49489], [91827, 73.20063], [39034, 60.168304], [71670, 53.74026]]       |
# +---------------+---------------------------------------------------------------------------------------------------------+

recommended_songs = (
    topK
    .withColumn("recommended_songs", extract_songs_top_k_udf(col("recommendations")))
    .select("user_id_encoded", "recommended_songs")
)
recommended_songs.cache()
recommended_songs.count()

# 932143

recommended_songs.show(10, False)

# +---------------+----------------------------------------+
# |user_id_encoded|songs                                   |
# +---------------+----------------------------------------+
# |38             |[107048, 127769, 129688, 113295, 145331]|
# |273            |[199021, 129688, 91827, 39034, 145331]  |
# |300            |[129688, 199021, 105961, 145331, 107048]|
# |412            |[190031, 129688, 71670, 145331, 199021] |
# |434            |[199021, 129688, 91827, 66074, 202418]  |
# |475            |[199021, 129688, 66074, 107048, 71670]  |
# |585            |[129688, 199021, 68593, 91827, 156831]  |
# |600            |[129688, 105961, 199021, 145331, 107048]|
# |611            |[43631, 199021, 107048, 153363, 105961] |
# |619            |[199021, 129688, 91827, 39034, 71670]   |
# +---------------+----------------------------------------+

# Relevant songs

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
)
relevant_songs.cache()
relevant_songs.count()

# 929537

relevant_songs.show(10, False)

# +---------------+-----------------------------------+
# |user_id_encoded|relevant_songs                     |
# +---------------+-----------------------------------+
# |38             |[43243, 32053, 32958, 25699, 33861]|
# |273            |[41816, 27149, 34678, 7667, 44085] |
# |300            |[252, 273, 249, 70526, 19087]      |
# |412            |[28731, 8672, 377, 3113, 12806]    |
# |434            |[8641, 4373, 59438, 9138, 3075]    |
# |475            |[341, 3367, 52732, 5522, 376]      |
# |585            |[10539, 1093, 92301, 1118, 4265]   |
# |600            |[249, 399, 239, 1329, 398]         |
# |611            |[147361, 16719, 4348, 13235, 5355] |
# |619            |[5434, 9311, 20623, 32116, 9872]   |
# +---------------+-----------------------------------+

combined = (
    recommended_songs.join(relevant_songs, on='user_id_encoded', how='inner')
    .rdd
    .map(lambda row: (row[1], row[2]))
)
combined.cache()
combined.count()

# 929537

combined.take(1)

# ([107048, 127769, 129688, 113295, 145331], [43243, 32053, 32958, 25699, 33861])

rankingMetrics = RankingMetrics(combined)
ndcgAtK = rankingMetrics.ndcgAt(k)
print(ndcgAtK)

# 1.8102832147923323e-05
