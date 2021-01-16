#!/usr/bin/env python
# coding: utf-8

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import (concat, col)
from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover,
                                StringIndexer, HashingTF, IDF)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# start a new spark session
spark = SparkSession.builder.appName('articles_classification').getOrCreate()

# reading the file into data object
data = spark.read.csv('articles.csv', header=True, inferSchema=True)
# drop null values and redundant articles
data = data.na.drop() \
    .distinct()

data.show()
# combining Title with Content columns and drop them after
data_combined = data.withColumn('article', concat('Title', 'Content')) \
    .drop('Title') \
    .drop('Content')

# show the categories and the number of articles for each
data_combined.groupBy("Category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# words tokenization
regex_tokenizer = RegexTokenizer(pattern='\\W',
                                 inputCol='article',
                                 outputCol='words')
# apply tokenization
words = regex_tokenizer.transform(data_combined)
words.show()
# stopword remover object
remover = StopWordsRemover(inputCol='words', outputCol='filtered')
# appply remove stopwords
filtered_words = remover.transform(words)
filtered_words.show()
# defining an HashingTF object
hashingTF = HashingTF(inputCol='filtered', outputCol='tf')
# transform the words into vectors
tf = hashingTF.transform(filtered_words)

# the output column is the features which is tf-idf vector
idf = IDF(inputCol='tf', outputCol='features')
idf_model = idf.fit(tf)
# transforming the data into TF-IDF vectors
tf_idf = idf_model.transform(tf)

# class (Category) into number conversion
category_numeric = StringIndexer(inputCol='Category', outputCol='label')
ready_data = category_numeric.fit(tf_idf) \
    .transform(tf_idf)

# can select only features and label columns only
# ready_data = ready_data.select(['label', 'features'])

ready_data.show()
# splitting the data into 70% training and 30% testing
train_data, test_data = ready_data.randomSplit([0.7, 0.3])

# defining LogisticRegression object lr
lr = LogisticRegression()
# training the model
lr_model = lr.fit(train_data)

# getting predictions of test data
test_results = lr_model.transform(test_data)

# Evaluation (accuracy by default)
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print(acc)
