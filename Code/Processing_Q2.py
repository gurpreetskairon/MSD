## +---------------------------------------------------------------------------------------------------+
## | Processing_Q2.py: DATA420 20S2 :: Assignment 2 :: Data Processing :: Question 2                   |
## |                                                                                                   |
## | __author__ = "Gurpreet Singh"                                                                     |
## | __ID__     = "24701854"                                                                           |
## | __email__  = "gsi58@uclive.ac.nz"                                                                 |
## | __date__   = "29 ‎September ‎2020"                                                                  |
## +---------------------------------------------------------------------------------------------------+

# Python and pyspark modules required

import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()


## +---------------------------------------------------------------------------------------------+
## | Processing 2a: Filter the Taste Profile dataset to remove the songs which were mismatched.  |
## +---------------------------------------------------------------------------------------------+

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

matches_manually_accepted = spark.createDataFrame(sc.parallelize(sid_matches_manually_accepted, 8), schema = mismatches_schema)
matches_manually_accepted.cache()

matches_manually_accepted.show(10, 40)

''' output
+------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
|           song_id|      song_artist|                              song_title|          track_id|                            track_artist|                             track_title|
+------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
|SOFQHZM12A8C142342|     Josipa Lisac|                                 razloga|TRMWMFG128F92FFEF2|                            Lisac Josipa|                            1000 razloga|
|SODXUTF12AB018A3DA|       Lutan Fyah|     Nuh Matter the Crisis Feat. Midnite|TRMWPCD12903CCE5ED|                                 Midnite|                   Nah Matter the Crisis|
|SOASCRF12A8C1372E6|Gaetano Donizetti|L'Elisir d'Amore: Act Two: Come sen v...|TRMHIPJ128F426A2E2|Gianandrea Gavazzeni_ Orchestra E Cor...|L'Elisir D'Amore_ Act 2: Come Sen Va ...|
|SOITDUN12A58A7AACA|     C.J. Chenier|                               Ay, Ai Ai|TRMHXGK128F42446AB|                         Clifton Chenier|                               Ay_ Ai Ai|
|SOLZXUM12AB018BE39|           許志安|                                男人最痛|TRMRSOF12903CCF516|                                Andy Hui|                        Nan Ren Zui Tong|
|SOTJTDT12A8C13A8A6|                S|                                       h|TRMNKQE128F427C4D8|                             Sammy Hagar|                 20th Century Man (Live)|
|SOGCVWB12AB0184CE2|                H|                                       Y|TRMUNCZ128F932A95D|                                Hawkwind|                25 Years (Alternate Mix)|
|SOKDKGD12AB0185E9C|     影山ヒロノブ|Cha-La Head-Cha-La (2005 ver./DRAGON ...|TRMOOAH12903CB4B29|                        Takahashi Hiroki|Maka fushigi adventure! (2005 Version...|
|SOPPBXP12A8C141194|    Αντώνης Ρέμος|                        O Trellos - Live|TRMXJDS128F42AE7CF|                           Antonis Remos|                               O Trellos|
|SODQSLR12A8C133A01|    John Williams|Concerto No. 1 for Guitar and String ...|TRWHMXN128F426E03C|               English Chamber Orchestra|II. Andantino siciliano from Concerto...|
+------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
only showing top 10 rows
'''

print(matches_manually_accepted.count())  

''' Ouput
488
'''

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

''' output
+------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
|           song_id|        song_artist|                              song_title|          track_id|  track_artist|                             track_title|
+------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
|SOUMNSI12AB0182807|Digital Underground|                        The Way We Swing|TRMMGKQ128F9325E10|      Linkwood|           Whats up with the Underground|
|SOCMRBE12AB018C546|         Jimmy Reed|The Sun Is Shining (Digitally Remaste...|TRMMREB12903CEB1B1|    Slim Harpo|               I Got Love If You Want It|
|SOLPHZY12AC468ABA8|      Africa HiTech|                                Footstep|TRMMBOC12903CEB46E|Marcus Worgull|                 Drumstern (BONUS TRACK)|
|SONGHTM12A8C1374EF|     Death in Vegas|                            Anita Berber|TRMMITP128F425D8D0|     Valen Hsu|                                  Shi Yi|
|SONGXCA12A8C13E82E| Grupo Exterminador|                           El Triunfador|TRMMAYZ128F429ECE6|     I Ribelli|                               Lei M'Ama|
|SOMBCRC12A67ADA435|      Fading Friend|                             Get us out!|TRMMNVU128EF343EED|     Masterboy|                      Feel The Heat 2000|
|SOTDWDK12A8C13617B|       Daevid Allen|                              Past Lives|TRMMNCZ128F426FF0E| Bhimsen Joshi|            Raga - Shuddha Sarang_ Aalap|
|SOEBURP12AB018C2FB|  Cristian Paduraru|                              Born Again|TRMMPBS12903CE90E1|     Yespiring|                          Journey Stages|
|SOSRJHS12A6D4FDAA3|         Jeff Mills|                      Basic Human Design|TRMWMEL128F421DA68|           M&T|                           Drumsettester|
|SOIYAAQ12A6D4F954A|           Excepter|                                      OG|TRMWHRI128F147EA8E|    The Fevers|Não Tenho Nada (Natchs Scheint Die So...|
+------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
only showing top 10 rows
'''

print(mismatches.count())  

''' output
19094
'''

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

''' Output
+----------------------------------------+------------------+-----+
|                                 user_id|           song_id|plays|
+----------------------------------------+------------------+-----+
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQEFDN12AB017C52B|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOIUJ12A6701DAA7|    2|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOKKD12A6701F92E|    4|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSDVHO12AB01882C7|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSKICX12A6701F932|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSNUPV12A8C13939B|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSVMII12A6701F92D|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTUNHI12B0B80AFE2|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTXLTZ12AB017C535|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTZDDX12A6701F935|    1|
+----------------------------------------+------------------+-----+
only showing top 10 rows
'''

mismatches_not_accepted = mismatches.join(matches_manually_accepted, on = "song_id", how = "left_anti")
triplets_not_mismatched = triplets.join(mismatches_not_accepted, on = "song_id", how = "left_anti")

print(triplets.count())

''' output
48373586
'''

print(triplets_not_mismatched.count())

''' Output
45795111
'''

## +----------------------------------------------------------------------------------------------------+
## | Processing 2b: Load the audio feature attribute names and types from the audio/attributes          |
## |                directory and use them to define schemas for the audio features themselves.         |
## |                Note that the attribute files and feature datasets share the same prefix and        |
## |                that the attribute types are named consistently. Think about how you can            |
## |                automate the creation of StructType by mapping attribute types to pyspark.sql.types |
## |                objects. You should read the documentation and blog post mentioned above carefully. |
## +----------------------------------------------------------------------------------------------------+

# get the different data types from the second column of the audio\attribute files. These are listed as string values

# hdfs dfs -cat "/data/msd/audio/attributes/*" | awk -F',' '{print $2}' | sort | uniq

''' Output
# NUMERIC
# real
# real 
# string
# string
# STRING
'''

audio_attribute_type_mapping = {
  "NUMERIC": DoubleType(),
  "real": DoubleType(),
  "string": StringType(),
  "STRING": StringType()
}

audio_dataset_names = [
  "msd-jmir-area-of-moments-all-v1.0",
  "msd-jmir-lpc-all-v1.0",
  "msd-jmir-methods-of-moments-all-v1.0",
  "msd-jmir-mfcc-all-v1.0",
  "msd-jmir-spectral-all-all-v1.0",
  "msd-jmir-spectral-derivatives-all-all-v1.0",
  "msd-marsyas-timbral-v1.0",
  "msd-mvd-v1.0",
  "msd-rh-v1.0",
  "msd-rp-v1.0",
  "msd-ssd-v1.0",
  "msd-trh-v1.0",
  "msd-tssd-v1.0"
]

audio_dataset_schemas = {}
for audio_dataset_name in audio_dataset_names:
  #print(audio_dataset_name)
  audio_dataset_path = f"/scratch-network/courses/2020/DATA420-20S2/data/msd/audio/attributes/{audio_dataset_name}.attributes.csv"
  with open(audio_dataset_path, "r") as f:
    rows = [line.strip().split(",") for line in f.readlines()]

  # rename feature columns with a short generic name
  rows[-1][0] = "track_id"
  for i, row in enumerate(rows[0:-1]):
    row[0] = f"feature_{i:04d}"

  audio_dataset_schemas[audio_dataset_name] = StructType([
    StructField(row[0], audio_attribute_type_mapping[row[1]], True) for row in rows
  ])
    
s = str(audio_dataset_schemas[audio_dataset_name])
print(s[0:50] + " ... " + s[-50:])


# check the keys in the audio_dataset_schemas dictionary
audio_dataset_schemas.keys()

''' Output
dict_keys(['msd-jmir-area-of-moments-all-v1.0', 'msd-jmir-lpc-all-v1.0', 'msd-jmir-methods-of-moments-all-v1.0', 'msd-jmir-mfcc-all-v1.0', 
'msd-jmir-spectral-all-all-v1.0', 'msd-jmir-spectral-derivatives-all-all-v1.0', 'msd-marsyas-timbral-v1.0', 'msd-mvd-v1.0', 
'msd-rh-v1.0', 'msd-rp-v1.0', 'msd-ssd-v1.0', 'msd-trh-v1.0', 'msd-tssd-v1.0'])
'''

# check the schema generated for one of the files
audio_dataset_schemas['msd-jmir-area-of-moments-all-v1.0']

''' output
StructType(List(StructField(feature_0000,DoubleType,true),StructField(feature_0001,DoubleType,true),StructField(feature_0002,DoubleType,true),
StructField(feature_0003,DoubleType,true),StructField(feature_0004,DoubleType,true),StructField(feature_0005,DoubleType,true),
StructField(feature_0006,DoubleType,true),StructField(feature_0007,DoubleType,true),StructField(feature_0008,DoubleType,true),
StructField(feature_0009,DoubleType,true),StructField(feature_0010,DoubleType,true),StructField(feature_0011,DoubleType,true),
StructField(feature_0012,DoubleType,true),StructField(feature_0013,DoubleType,true),StructField(feature_0014,DoubleType,true),
StructField(feature_0015,DoubleType,true),StructField(feature_0016,DoubleType,true),StructField(feature_0017,DoubleType,true),
StructField(feature_0018,DoubleType,true),StructField(feature_0019,DoubleType,true),StructField(track_id,StringType,true)))
'''


# load the filesdata from the 'msd-jmir-methods-of-moments-all-v1.0.csv.gz' /audio/feature file with the above created schemas
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
