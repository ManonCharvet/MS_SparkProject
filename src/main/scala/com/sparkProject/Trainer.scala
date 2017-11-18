package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._ /** to be able to use $ in the columns selections **/


    /*******************************************************************************
      *
      *       Auteur: Manon CHARVET
      *       To run the code, write in the terminal: ./build_and_submit.sh Trainer
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** 1) LOAD THE DATASET**/
    val parquetFileDF = spark.read.parquet("prepared_trainingset")

   /** 2) TF-IDF **/
    /** a. Split the text into words **/
   val tokenizer = new RegexTokenizer()
     .setPattern("\\W+")
     .setGaps(true)
     .setInputCol("text")
     .setOutputCol("tokens")

    /** b. Remove stop words **/
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("stopWordsRemoved")

    /** c. TF-IDF give a numeric score to each word**/
    val cvModel = new CountVectorizer()
      .setInputCol("stopWordsRemoved")
      .setOutputCol("tfidf")
      .setVocabSize(3)

    /** 3) CATEGORY VARIABLES TO NUMERIC SCORES**/
    /** e. "country2" to numeric**/
    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    /** f. "currency2" to numeric**/
    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    /** 4) PREPARE DATA FOR SPARK.ML **/
    /** g. Assemble the features  "tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"
      * in a unique column "features" **/
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf","days_campaign","hours_prepa","goal","country_indexed","currency_indexed"))
      .setOutputCol("features")

   /** h. Classification model: logistic regression**/
   val lr = new LogisticRegression()
     .setElasticNetParam(0.0)
     .setFitIntercept(true)
     .setFeaturesCol("features")
     .setLabelCol("final_status")
     .setStandardization(true)
     .setPredictionCol("predictions")
     .setRawPredictionCol("raw_predictions") // difference with predictions?
     .setThresholds(Array(0.7, 0.3))
     .setTol(1.0e-6)
     .setMaxIter(300)

   /* Another model tried: Random Forest*/
   val rf = new RandomForestClassifier()
     .setLabelCol("final_status")
     .setFeaturesCol("features")
     .setPredictionCol("predictions")
     .setImpurity("entropy")
     .setFeatureSubsetStrategy("sqrt") //auto
     .setMaxBins(300)

   /** i. Create a pipeline by gathering the 8 aforementioned stages **/
   val pipeline = new Pipeline()
     .setStages(Array(tokenizer,remover,cvModel,indexerCountry,indexerCurrency,assembler,lr))
     //.setStages(Array(tokenizer,remover,cvModel,indexerCountry,indexerCurrency,assembler,rf))


    /** 5) MODEL: TRAINING AND TUNING **/
    /** Split data in Training Set and Test Set **/
    /** j. Split the dataFrame (90%, 10%) **/
    val splits = parquetFileDF.randomSplit(Array(0.9,0.1))
    val (training, test) = (splits(0), splits(1))

    /** Classifier training and hyperparameter tuning **/
    /** k. Prepare and run the grid-search (logistic regression, minDF, f1-score) **/
    // We use a ParamGridBuilder to construct a grid of parameters to search over
    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(10e-8,10e-6,10e-4,10e-2))
      //.addGrid(rf.numTrees, Array(15,20,25)).addGrid(rf.maxDepth,Array(20,30,40)) // if we use the Random Forest Model
      .build()

    // Try all combinations of values and determine best model
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val model = trainValidationSplit.fit(training)

    /** l. Apply the best model computed by the grid-search on the test set **/
    val df_WithPredictions = model.transform(test)

    val test_f1Score = evaluator.evaluate(df_WithPredictions)
    println("F1 score on test data: ")
    println(test_f1Score)

    /** m. Display the results**/
    df_WithPredictions.groupBy($"final_status", $"predictions").count.show()

    /** Save the model **/
    //model.save("/Users/Manon/Documents/Cours/MS/Spark/TP_ParisTech_2017_2018_starter/model_TP4")

  }
}
