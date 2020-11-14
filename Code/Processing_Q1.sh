## +---------------------------------------------------------------------------------------------------+
## | Processing_Q1.sh: DATA420 20S2 :: Assignment 2 :: Data Processing :: Question 1                   |
## |                                                                                                   |
## | __author__ = "Gurpreet Singh"                                                                     |
## | __ID__     = "24701854"                                                                           |
## | __email__  = "gsi58@uclive.ac.nz"                                                                 |
## | __date__   = "28 ‎September ‎2020"                                                                  |
## +---------------------------------------------------------------------------------------------------+


## +---------------------------------------------------------------------------------------------------+
## Q1 a. Give an overview of the structure of the datasets, including file formats, data types, and    |
##       how each dataset has been stored in HDFS.                                                     |
## +---------------------------------------------------------------------------------------------------+


## structure of the datasets
hdfs dfs -ls /data/msd

## output
# Found 4 items
# drwxr-xr-x   - hadoop supergroup          0 2019-05-06 14:10 /data/msd/audio
# drwxr-xr-x   - hadoop supergroup          0 2019-05-06 14:17 /data/msd/genre
# drwxr-xr-x   - hadoop supergroup          0 2019-05-06 14:21 /data/msd/main
# drwxr-xr-x   - hadoop supergroup          0 2019-05-06 14:23 /data/msd/tasteprofile
##

hdfs dfs -ls /data/msd/audio

## output
# Found 3 items
# drwxr-xr-x   - hadoop supergroup          0 2019-05-06 13:52 /data/msd/audio/attributes
# drwxr-xr-x   - hadoop supergroup          0 2019-05-06 14:04 /data/msd/audio/features
# drwxr-xr-x   - hadoop supergroup          0 2019-05-06 14:10 /data/msd/audio/statistics

hdfs dfs -ls -a -R /data/msd | awk '{print $8}' | sed -e 's/[^-][^\/]*\//--/g' -e 's/^/ /' -e 's/-/|/'

## output
 # |---audio
 # |-----attributes
 # |-------msd-jmir-area-of-moments-all-v1.0.attributes.csv
 # |-------msd-jmir-lpc-all-v1.0.attributes.csv
 # |-------msd-jmir-methods-of-moments-all-v1.0.attributes.csv
 # |-------msd-jmir-mfcc-all-v1.0.attributes.csv
 # |-------msd-jmir-spectral-all-all-v1.0.attributes.csv
 # |-------msd-jmir-spectral-derivatives-all-all-v1.0.attributes.csv
 # |-------msd-marsyas-timbral-v1.0.attributes.csv
 # |-------msd-mvd-v1.0.attributes.csv
 # |-------msd-rh-v1.0.attributes.csv
 # |-------msd-rp-v1.0.attributes.csv
 # |-------msd-ssd-v1.0.attributes.csv
 # |-------msd-trh-v1.0.attributes.csv
 # |-------msd-tssd-v1.0.attributes.csv
 # |-----features
 # |-------msd-jmir-area-of-moments-all-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-jmir-lpc-all-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-jmir-methods-of-moments-all-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-jmir-mfcc-all-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-jmir-spectral-all-all-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-jmir-spectral-derivatives-all-all-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-marsyas-timbral-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-mvd-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-rh-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-rp-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-ssd-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-trh-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-------msd-tssd-v1.0.csv
 # |---------part-00000.csv.gz
 # |---------part-00001.csv.gz
 # |---------part-00002.csv.gz
 # |---------part-00003.csv.gz
 # |---------part-00004.csv.gz
 # |---------part-00005.csv.gz
 # |---------part-00006.csv.gz
 # |---------part-00007.csv.gz
 # |-----statistics
 # |-------sample_properties.csv.gz
 # |---genre
 # |-----msd-MAGD-genreAssignment.tsv
 # |-----msd-MASD-styleAssignment.tsv
 # |-----msd-topMAGD-genreAssignment.tsv
 # |---main
 # |-----summary
 # |-------analysis.csv.gz
 # |-------metadata.csv.gz
 # |---tasteprofile
 # |-----mismatches
 # |-------sid_matches_manually_accepted.txt
 # |-------sid_mismatches.txt
 # |-----triplets.tsv
 # |-------part-00000.tsv.gz
 # |-------part-00001.tsv.gz
 # |-------part-00002.tsv.gz
 # |-------part-00003.tsv.gz
 # |-------part-00004.tsv.gz
 # |-------part-00005.tsv.gz
 # |-------part-00006.tsv.gz
 # |-------part-00007.tsv.gz


hdfs dfs -ls /data/msd/audio/features


hdfs dfs -ls /data/msd/audio/statistics

# '''outout
# Found 1 items
# -rwxr-xr-x   8 hadoop supergroup   42224669 2019-05-06 14:10 /data/msd/audio/statistics/sample_properties.csv.gz
# '''

hdfs dfs -ls /data/msd/genre

hdfs dfs -ls /data/msd/main

hdfs dfs -ls /data/msd/main

hdfs dfs -ls /data/msd/tasteprofile/triplets.tsv


hdfs dfs -cat /data/msd/genre/msd-MAGD-genreAssignment.tsv | head



## +---------------------------------------------------------------------------------------------------+
## Q1 c. Give an overview of the structure of the datasets, including file formats, data types, and    |
##       how each dataset has been stored in HDFS.                                                     |
## +---------------------------------------------------------------------------------------------------+l

