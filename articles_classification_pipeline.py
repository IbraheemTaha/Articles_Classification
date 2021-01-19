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

from pyspark.ml import Pipeline


def to_prepare(data):
    """
    :param data
    :return ready_data

    This  function takes the data which is loaded from the articles.csv file and returns clean and ready data for
    machine learning classification task using pipeline to handle the required stages.
    """

    # drop null values and redundant articles
    data = data.na.drop() \
        .distinct()

    # data.show()

    # combining Title with Content in the new column 'article' and drop them after
    data_combined = data.withColumn('article', concat('Title', 'Content')) \
        .drop('Title') \
        .drop('Content')

    ## show the categories and the number of articles for each
    # data_combined.groupBy("Category") \
    #     .count() \
    #     .orderBy(col("count").desc()) \
    #     .show()

    # words tokenization
    regex_tokenizer = RegexTokenizer(pattern='\\W',
                                     inputCol='article',
                                     outputCol='words')
    # stopword remover object
    remover = StopWordsRemover(inputCol='words', outputCol='filtered')

    # defining an HashingTF object
    hashingTF = HashingTF(inputCol='filtered', outputCol='tf')

    # the output column is the features which is tf-idf vector
    idf = IDF(inputCol='tf', outputCol='features')

    # class (Category) into number conversion
    category_numeric = StringIndexer(inputCol='Category', outputCol='label')

    # defining the pipeline object (object) has the stages of pipelines
    prep_data_pipe = Pipeline(stages=[regex_tokenizer, remover, hashingTF,
                                      idf, category_numeric, ])
    # create the cleaner model
    cleaner = prep_data_pipe.fit(data_combined)
    # implement data cleaning using the pipeline model 'cleaner'
    clean_data = cleaner.transform(data_combined)

    # selecting the label and the features columns (ready for ML models)
    ready_data = clean_data.select(['label', 'features'])

    return ready_data


if __name__ == "__main__":
    # start a new spark session
    spark = SparkSession.builder.appName('articles_classification_pipeline').getOrCreate()

    # reading the file into data object
    data = spark.read.csv('articles.csv', header=True, inferSchema=True)

    # call to_pipeline to prepare and retrieve clean data
    clean_data = to_prepare(data)

    # show the final clean data
    clean_data.show()

    # splitting the data into 70% training and 30% testing
    train_data, test_data = clean_data.randomSplit([0.7, 0.3])

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
