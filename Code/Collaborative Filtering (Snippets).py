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
matches_manually_accepted.show(10, 20)

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
mismatches.show(10, 20)

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

user_song_count_threshold = 34
song_user_count_threshold = 5

triplets_limited = triplets_not_mismatched

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

from pyspark.mllib.evaluation import RankingMetrics

# Modeling

als = ALS(maxIter=5, regParam=0.01, userCol="user_id_encoded", itemCol="song_id_encoded", ratingCol="plays", implicitPrefs=True)
als_model = als.fit(training)
predictions = als_model.transform(test)

predictions = predictions.orderBy(col("user_id"), col("song_id"), col("prediction").desc())
predictions.cache()

predictions.show(50, False)

# +------------------+----------------------------------------+-----+---------------+---------------+-------------+
# |song_id           |user_id                                 |plays|user_id_encoded|song_id_encoded|prediction   |
# +------------------+----------------------------------------+-----+---------------+---------------+-------------+
# |SOGDQWF12A67AD954F|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |305105.0       |2020.0         |0.036639214  |
# |SOGZCOB12A8C14280E|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |305105.0       |386.0          |0.08718535   |
# |SOKEYJQ12A6D4F6132|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |305105.0       |208.0          |0.11116325   |
# |SOKUECJ12A6D4F6129|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |305105.0       |175.0          |0.1389938    |
# |SOPDRWC12A8C141DDE|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |305105.0       |286.0          |0.09713399   |
# |SOPWKOX12A8C139D43|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|4    |305105.0       |1638.0         |0.041958686  |
# |SOUZBUD12A8C13FD8E|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |305105.0       |984.0          |0.027315382  |
# |SOWNIUS12A8C142815|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|2    |305105.0       |910.0          |0.05989272   |
# |SOWRMTT12A8C137064|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |305105.0       |777.0          |0.06709951   |
# |SOXLKNJ12A58A7E09A|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|3    |305105.0       |628.0          |0.07203257   |
# |SOYEQLD12AB017C713|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |305105.0       |1201.0         |0.051622357  |
# |SOYWTUB12A8C13B429|00004fb90a86beb8bed1e9e328f5d9b6ee7dc03e|1    |305105.0       |1289.0         |0.0497122    |
# |SOBJYFB12AB018372D|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |6082.0         |0.0016309685 |
# |SOCRKNT12AB018940D|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |53167.0        |1.4983097E-4 |
# |SOEPNVO12AF72A7CC9|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |27931.0        |2.8570739E-4 |
# |SOGRNDU12A3F1EB51F|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |222068.0       |90245.0        |6.331475E-5  |
# |SOHYKCX12A6D4F636F|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |57191.0        |1.11624126E-4|
# |SOHYYDE12AB018A608|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |56129.0        |2.2221566E-4 |
# |SOINVHR12AB0189418|00007ed2509128dcdd74ea3aac2363e24e9dc06b|2    |222068.0       |52517.0        |1.4299405E-4 |
# |SOOKJWB12A6D4FD4F8|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |25841.0        |4.09288E-4   |
# |SOPFUBI12A58A79E33|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |73560.0        |7.423946E-5  |
# |SOUZBHL12AF72A4E05|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |75349.0        |4.6063407E-5 |
# |SOVGNWE12A6D4FB90A|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |51435.0        |9.438249E-5  |
# |SOWYFRZ12A6D4FD507|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |61632.0        |8.590981E-5  |
# |SOXBNCI12A8C13B2DF|00007ed2509128dcdd74ea3aac2363e24e9dc06b|1    |222068.0       |69614.0        |7.301029E-5  |
# |SOEWPBR12A58A79271|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |308929.0       |36571.0        |1.7794217E-4 |
# |SOFWMCO12AB01834B9|00009d93dc719d1dbaf13507725a03b9fdeebebb|4    |308929.0       |89112.0        |4.8049995E-5 |
# |SOIDGYO12A8C141C9B|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |308929.0       |85844.0        |3.261717E-5  |
# |SOLMHOY12AB01834C3|00009d93dc719d1dbaf13507725a03b9fdeebebb|6    |308929.0       |85382.0        |6.0017494E-5 |
# |SOOFZUI12A8C13C033|00009d93dc719d1dbaf13507725a03b9fdeebebb|2    |308929.0       |18701.0        |3.874116E-4  |
# |SOQEKID12A8AE45CF5|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |308929.0       |84141.0        |2.3665496E-5 |
# |SOVTBQI12A8C142ABA|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |308929.0       |65655.0        |4.4403674E-5 |
# |SOYXCKN12AB018058C|00009d93dc719d1dbaf13507725a03b9fdeebebb|1    |308929.0       |86376.0        |6.442769E-5  |
# |SOZSOOL12AB01834B2|00009d93dc719d1dbaf13507725a03b9fdeebebb|3    |308929.0       |77220.0        |5.3262418E-5 |
# |SOBHUPL12A670203C9|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |64041.0        |0.0017627005 |
# |SOCRUVF12A6D4F5906|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |3396.0         |0.0035303235 |
# |SODEZKY12A8C134C4D|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |17046.0        |0.0051425677 |
# |SOEGJGK12A8C143C5A|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |95803.0        |2.1319842E-4 |
# |SOEIXYS12A6D4F8109|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |75689.0        |36530.0        |0.0012925802 |
# |SOFSGLT12AB018007B|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |75689.0        |1850.0         |0.028723273  |
# |SOGZKCR12A6D4FBCF8|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |14341.0        |0.0059330305 |
# |SOHPIUC12AB018046E|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|3    |75689.0        |62877.0        |5.996754E-4  |
# |SOHTWNJ12A6701D0EB|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|8    |75689.0        |56141.0        |0.0011730383 |
# |SOJAYEY12AB0185304|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |45892.0        |4.0994305E-4 |
# |SOJHVCP12A6701D0F3|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |86566.0        |3.3705006E-4 |
# |SOJOEIK12A6D4FBE16|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |48305.0        |4.863531E-4  |
# |SOKBDRO12A67020ED9|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |13372.0        |0.008220233  |
# |SOKBHQN12A67ADBAE2|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |75689.0        |12261.0        |0.011528797  |
# |SOKUWEV12A8C13BBEB|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|2    |75689.0        |1317.0         |0.027168337  |
# |SOKZBJA12AB018B10B|0000bb531aaa657c932988bc2f7fd7fc1b2050ec|1    |75689.0        |88529.0        |4.1410798E-4 |
# +------------------+----------------------------------------+-----+---------------+---------------+-------------+


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

# Helpers 

def extract_songs_top_k(x, k):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x][0:k]

extract_songs_top_k_udf = udf(lambda x: extract_songs_top_k(x, k), ArrayType(IntegerType()))

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

# +---------------+-----------------------------------------------------------------------------------------------+
# |user_id_encoded|recommendations                                                                                |
# +---------------+-----------------------------------------------------------------------------------------------+
# |38             |[[54, 1.2309796], [97, 1.0754948], [132, 1.070123], [185, 0.980086], [48, 0.9479991]]          |
# |273            |[[0, 0.88274914], [48, 0.7877278], [31, 0.6852624], [46, 0.67231166], [33, 0.6501744]]         |
# |300            |[[72, 0.6792025], [38, 0.665281], [201, 0.6050984], [145, 0.5780993], [220, 0.56818783]]       |
# |412            |[[0, 1.1526669], [17, 0.97309804], [21, 0.9503651], [20, 0.93405145], [16, 0.90946066]]        |
# |434            |[[54, 0.8748799], [0, 0.8030314], [1, 0.797909], [132, 0.7597165], [97, 0.7535728]]            |
# |475            |[[154, 0.3329019], [15, 0.32719484], [62, 0.32417718], [44, 0.31954965], [27, 0.30711204]]     |
# |585            |[[0, 0.93643475], [11, 0.91167], [90, 0.8503059], [7, 0.6912614], [46, 0.62911165]]            |
# |600            |[[3, 0.53979814], [17, 0.5197733], [52, 0.5073601], [4, 0.50308836], [51, 0.47169417]]         |
# |611            |[[64, 0.35207328], [881, 0.33518124], [283, 0.3049816], [1858, 0.29279813], [2938, 0.28774542]]|
# |619            |[[132, 0.52814245], [70, 0.5198533], [48, 0.47640824], [54, 0.46371958], [0, 0.42702913]]      |
# +---------------+-----------------------------------------------------------------------------------------------+

recommended_songs = (
    topK
    .withColumn("recommended_songs", extract_songs_top_k_udf(col("recommendations")))
    .select("user_id_encoded", "recommended_songs")
)
recommended_songs.cache()
recommended_songs.count()

# 932143

recommended_songs.show(10, 50)

# +---------------+--------------------------+
# |user_id_encoded|         recommended_songs|
# +---------------+--------------------------+
# |             38|    [54, 97, 132, 185, 48]|
# |            273|       [0, 48, 31, 46, 33]|
# |            300|   [72, 38, 201, 145, 220]|
# |            412|       [0, 17, 21, 20, 16]|
# |            434|       [54, 0, 1, 132, 97]|
# |            475|     [154, 15, 62, 44, 27]|
# |            585|        [0, 11, 90, 7, 46]|
# |            600|        [3, 17, 52, 4, 51]|
# |            611|[64, 881, 283, 1858, 2938]|
# |            619|      [132, 70, 48, 54, 0]|
# +---------------+--------------------------+

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

relevant_songs.show(10, 50)

# +---------------+--------------------------------------------------+
# |user_id_encoded|                                    relevant_songs|
# +---------------+--------------------------------------------------+
# |             38|[1290, 923, 1178, 1542, 3512, 6852, 7060, 12253...|
# |            273|[446, 29509, 4, 19515, 3839, 9118, 7567, 859, 5...|
# |            300|[10765, 224, 25098, 4023, 463, 27143, 27005, 71...|
# |            412|[3, 2191, 1195, 4, 70, 4703, 33196, 4265, 44312...|
# |            434|[219, 70, 8, 4211, 458, 285, 17571, 13, 3959, 1...|
# |            475|[15050, 82891, 10087, 2630, 731, 3116, 37137, 2...|
# |            585|[9871, 10830, 26221, 9532, 17, 17901, 11298, 32...|
# |            600|[16063, 375, 16067, 6753, 2952, 600, 867, 13655...|
# |            611|[37984, 1808, 114, 2826, 2037, 2970, 17043, 176...|
# |            619|[754, 8100, 14092, 829, 17310, 41356, 914, 1490...|
# +---------------+--------------------------------------------------+

combined = (
    recommended_songs.join(relevant_songs, on='user_id_encoded', how='inner')
    .rdd
    .map(lambda row: (row[1], row[2]))
)
combined.cache()
combined.count()

# 929537

combined.take(1)

# ([107048, 127769, 129688, 113295, 145331], [1290, 923, 1178, 1542, 3512, 6852, 7060, 12253...])

rankingMetrics = RankingMetrics(combined)
ndcgAtK = rankingMetrics.ndcgAt(k)
print(ndcgAtK)

# 0.06112937494527085
