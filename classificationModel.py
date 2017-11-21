# from pyspark import SparkConf
# from pyspark import SparkContext
# from pyspark import SQLContext
# # from pyspark.sql.functions import udf
# # from pyspark.ml.classification import LogisticRegression
# # from pyspark.sql.types import DoubleType, IntegerType
# # from pyspark.mllib.regression import LabeledPoint
# # from numpy import array
# #
# #
# # def readData(sqlContext):
# #     df = sqlContext.read.format('com.databricks.spark.csv').options(header='false')\
# #                         .option("inferSchema", "false").load("data")
# #     df = df.selectExpr("_c0 as Gender", "_c1 as Length", "_c2 as Diameter",
# #                        "_c3 as Height", "_c4 as Whole", "_c5 as Shucked",
# #                        "_c6 as Viscera", "_c7 as Shell", "_c8 as Rings")
# #     changedTypedf = df.withColumn("Length", df["Length"].cast(DoubleType()))\
# #                       .withColumn("Diameter", df["Diameter"].cast(DoubleType()))\
# #                       .withColumn("Height", df["Height"].cast(DoubleType()))\
# #                       .withColumn("Whole", df["Whole"].cast(DoubleType())) \
# #                       .withColumn("Shucked", df["Shucked"].cast(DoubleType()))\
# #                       .withColumn("Viscera", df["Viscera"].cast(DoubleType()))\
# #                       .withColumn("Shell", df["Shell"].cast(DoubleType()))\
# #                       .withColumn("Rings", df["Rings"].cast(IntegerType()))
# #     def fun(gender):
# #         if gender == "M": return 1
# #         else: return 0
# #     udfBinary = udf(fun, IntegerType())
# #     binaryTypedf = changedTypedf.withColumn("GenderBinary", udfBinary(changedTypedf["Gender"]))
# #     return binaryTypedf
# #
# #
# # def dataSplit(df):
# #     train, test = df.randomSplit([0.6, 0.4])
# #     trainX = train.select("GenderBinary")
# #     trainY = train.select("Length","Diameter","Height","Whole","Shucked","Viscera","Shell","Rings")
# #     testX = train.select("GenderBinary")
# #     testY = train.select("Length","Diameter","Height","Whole","Shucked","Viscera","Shell","Rings")
# #     return trainX, trainY, testX, testY
# #
# #
# # def logisticRegression():
# #
# #     def parseInteraction(line):
# #         lineSplit = line.split
# #     lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
# #     lrModel = lr.fit
# #
# # if __name__ == "__main__":
# #
# #     conf = SparkConf().setMaster("classification").setMaster("local")
# #     sc = SparkContext(conf=conf)
# #     sqlContext = SQLContext(sc)
# #     df = readData(sqlContext)
# #     trainX, trainY, testX, testY = dataSplit(df)
# #     logisticRegression()
# #     print testX.show(10)
#
#
# # from pyspark.ml import Pipeline
# # from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
# #
# # categoricalColumns = ["Length","Diameter","Height","Whole","Shucked","Viscera","Shell","Rings","Gender"]
# # stages = []
# # for categoricalCol in categoricalColumns:
# #
# #   stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
# #   encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
# #   stages += [stringIndexer, encoder]
# #
# # print stages
#
#

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier


def readData(sqlContext):
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='false')\
                        .option("inferSchema", "false").load("data")
    df = df.selectExpr("_c0 as Gender", "_c1 as Length", "_c2 as Diameter",
                       "_c3 as Height", "_c4 as Whole", "_c5 as Shucked",
                       "_c6 as Viscera", "_c7 as Shell", "_c8 as Rings")
    changedTypedf = df.withColumn("Length", df["Length"].cast(DoubleType()))\
                      .withColumn("Diameter", df["Diameter"].cast(DoubleType()))\
                      .withColumn("Height", df["Height"].cast(DoubleType()))\
                      .withColumn("Whole", df["Whole"].cast(DoubleType())) \
                      .withColumn("Shucked", df["Shucked"].cast(DoubleType()))\
                      .withColumn("Viscera", df["Viscera"].cast(DoubleType()))\
                      .withColumn("Shell", df["Shell"].cast(DoubleType()))\
                      .withColumn("Rings", df["Rings"].cast(IntegerType()))
    return changedTypedf


def dfSplit(df):
    train, test = df.randomSplit([0.7, 0.3])
    return train, test


def dfPipeline(train, labelIndex, assembler, classifier):
    pipeline = Pipeline(stages=[labelIndex,assembler,classifier])
    model = pipeline.fit(train)
    return model


def dfPrediction(model, test):
    return model.transform(test)


def dfEvaluation(predictions):
    evaluator = BinaryClassificationEvaluator()
    # auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    # print "ROC curve value: ", auroc
    accuracy = evaluator.evaluate(predictions)
    print "Accuracy : ", accuracy * 100


if __name__ == "__main__":


    conf = SparkConf().setMaster("classification").setMaster("local")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    df = readData(sqlContext)


    train, test = dfSplit(df)


    labelIndex = StringIndexer(inputCol="Gender", outputCol="label")
    assembler = VectorAssembler(inputCols=["Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell", "Rings"]
                                , outputCol="features")

    randomForestClassifier = RandomForestClassifier(labelCol="label", featuresCol="features")
    logesticRegressionClassifier = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    multiNomialRegressionClassifier = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")
    decisionTreeClassifier = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    gradientBoostedTreeClassifier = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

    classifierDict = {
        "Random Forest" : randomForestClassifier,
        "Logestic Regression" : logesticRegressionClassifier,
        "Multinominal Regression" : multiNomialRegressionClassifier,
        # "Decision Tree" : decisionTreeClassifier,
        # "Gradient Boosted Tree Classifier" : gradientBoostedTreeClassifier,


    }

    for eachClassifer in classifierDict:
        print eachClassifer
        model = dfPipeline(train, labelIndex, assembler, classifierDict[eachClassifer])
        predictions = dfPrediction(model, test)
        dfEvaluation(predictions)


