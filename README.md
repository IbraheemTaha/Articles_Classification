# Articles Classification 

Articles classification is a Python classification solution using **_Apache Spark_**. It is a straightforward implementation of main concept of **NLP**, **machine learning**, and **Spark**.

The main task is to create a model (Logistic Regression Model) to classify the the articles and each corresponding category "class".

## Overview

The file **_articles.csv_** has three columns **['Title', 'Content', 'Category']** which can also be seen in the following schema:

```bash
root
 |-- Title: string (nullable = true)
 |-- Content: string (nullable = true)
 |-- Category: string (nullable = true)
```


There are five categories "classes" as follows:

```bash
+----------+-----+                                                              
|  Category|count|
+----------+-----+
|  Football| 3121|
|  Business| 2735|
|  Politics| 2683|
|      Film| 2240|
|Technology| 1487|
+----------+-----+
```



## Stages
The following stages show how the data is handled from starting the spark session , loading and cleaning data, , and reaching the building of the classifier model: 

### Spark Session 
Starting a new Spark session 
```python
# findspark will automatically determine the place of Spark library
import findspark
findspark.init()

from pyspark.sql import SparkSession
# start a new spark session
spark = SparkSession.builder.appName('articles_classification').getOrCreate()

```
### Import
These are required to import to complete this task
```python
from pyspark.sql.functions import (concat, col)
from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover,
                                StringIndexer, HashingTF, IDF)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```

### Loading
Data is loaded directly from the file **_articles.csv_**
```python
# reading the file into data object
data = spark.read.csv('articles.csv', header=True, inferSchema=True)
```
### Cleaning and Tokenizing
```python
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
```

### Hashing (Vectorization)

In this stage, the clean filtered words are converted into vectors of number representing each document. I chose **_TF-IDF_** technique, however there are some others such as **_W2V_** and **_BoW_**. **_TF-IDF_** has several advantages that can show the frequency as well as the importance of a word in a document "the article".


```python
# defining an HashingTF object 
hashingTF = HashingTF(inputCol='filtered', outputCol='tf')
# transform the words into vectors
tf = hashingTF.transform(filtered_words)

# the output column is the features which is tf-idf vector
idf = IDF(inputCol='tf', outputCol='features')
idf_model = idf.fit(tf)
# transforming the data into TF-IDF vectors
tf_idf = idf_model.transform(tf)

```
### Category Indexing
In the following code, the categories are converted into numbers automatically to use them in training the model as labels
```python
# Class (Category) into number conversion
category_numeric = StringIndexer(inputCol='Category', outputCol='label')
ready_data = category_numeric.fit(tf_idf) \
    .transform(tf_idf)
```


### Data Splitting
```python
# splitting the data into 70% training and 30% testing
train_data, test_data = ready_data.randomSplit([0.7, 0.3])
```
### Classifier Building
```python
# defining LogisticRegression object lr
lr = LogisticRegression()
# training the model
lr_model = lr.fit(train_data)
```

### Classification and Evaluation
Finally, classify the categories of test data and evaluate the model based on that.
```python
# getting predictions of test data
test_results = lr_model.transform(test_data)

# Evaluation (accuracy metric by default)
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print(acc)
```

# >>>>>>>>>>> Pipeline <<<<<<<<<<<
# Articles Classification Pipeline

I have added this section as well as the the python file **_articles_classification_pipneline.py_**. The code performs the same task as in the previous section, however this uses pipeline.

## Pipeline Function

The function **_to_prepare_** takes the raw data and return clean and ready data for machine learning classification task using pipeline. The main idea of it that one can only define the required stages, pass them to the pipeline and retrieve the data without fitting or transforming data for each time. These stages are handled by pipeline as shown in the following function:
```python
def to_prepare(data):
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
```
The code in **_main_**:
```python
    # call to_pipeline to prepare and retrieve clean data
    clean_data = to_pipeline(data_combined)
```
The complete code can be found in the uploaded file **_articles_classification_pipneline.py_**.

## Contribution
Pull requests are most welcome and I am open for any discussion!