package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


object Kaggle {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext


    /** ******************************************************************************
      *
      * Test: Big Data Mining
      *
      *        - Kaggle
      * *******************************************************************************/
    /** 1) LOAD THE DATASET**/
    val train: DataFrame = spark
      .read
      .option("header", "true")
      .option("nullValue","false")
      .csv("Kaggle_train.csv")

    val test: DataFrame = spark
      .read
      .option("header", "true")
      .option("nullValue","false")
      .csv("Kaggle_test.csv")

    //println("Nombre de lignes du dataframe:")
    //println(train.count())

    //Number of each type
    //train.groupBy("Cover_type").count.orderBy($"count".desc).show(99)


    /** numeric variables **/
    val trainCasted = train
      .withColumn("Elevation", $"Elevation".cast("Int"))
      .withColumn("Aspect", $"Aspect".cast("Int"))
      .withColumn("Slope", $"Slope".cast("Int"))
      .withColumn("Horizontal_Distance_To_Hydrology", $"Horizontal_Distance_To_Hydrology".cast("Int"))
      .withColumn("Vertical_Distance_To_Hydrology", $"Vertical_Distance_To_Hydrology".cast("Int"))
      .withColumn("Horizontal_Distance_To_Roadways", $"Horizontal_Distance_To_Roadways".cast("Int"))
      .withColumn("Hillshade_9am", $"Hillshade_9am".cast("Int"))
      .withColumn("Hillshade_Noon", $"Hillshade_Noon".cast("Int"))
      .withColumn("Hillshade_3pm", $"Hillshade_3pm".cast("Int"))
      .withColumn("Horizontal_Distance_To_Fire_Points", $"Horizontal_Distance_To_Fire_Points".cast("Int"))
      /**  .withColumn("Soil_Type2", $"Soil_Type2".cast("Int"))
      .withColumn("Soil_Type3", $"Soil_Type3".cast("Int"))
      .withColumn("Soil_Type4", $"Soil_Type4".cast("Int"))
      .withColumn("Soil_Type5", $"Soil_Type5".cast("Int"))
      .withColumn("Soil_Type6", $"Soil_Type6".cast("Int"))
      .withColumn("Soil_Type7", $"Soil_Type7".cast("Int"))
      .withColumn("Soil_Type8", $"Soil_Type8".cast("Int"))
      .withColumn("Soil_Type9", $"Soil_Type9".cast("Int"))
      .withColumn("Soil_Type10", $"Soil_Type10".cast("Int"))
      .withColumn("Soil_Type11", $"Soil_Type11".cast("Int"))
      .withColumn("Soil_Type12", $"Soil_Type12".cast("Int"))
      .withColumn("Soil_Type13", $"Soil_Type13".cast("Int"))
      .withColumn("Soil_Type14", $"Soil_Type14".cast("Int"))
      .withColumn("Soil_Type15", $"Soil_Type15".cast("Int"))
      .withColumn("Soil_Type16", $"Soil_Type16".cast("Int"))
      .withColumn("Soil_Type17", $"Soil_Type17".cast("Int"))
      .withColumn("Soil_Type18", $"Soil_Type18".cast("Int"))
      .withColumn("Soil_Type19", $"Soil_Type19".cast("Int"))
      .withColumn("Soil_Type20", $"Soil_Type20".cast("Int"))
      .withColumn("Soil_Type21", $"Soil_Type21".cast("Int"))
      .withColumn("Soil_Type22", $"Soil_Type22".cast("Int"))
      .withColumn("Soil_Type23", $"Soil_Type23".cast("Int"))
      .withColumn("Soil_Type24", $"Soil_Type24".cast("Int"))
      .withColumn("Soil_Type25", $"Soil_Type25".cast("Int"))
      .withColumn("Soil_Type26", $"Soil_Type26".cast("Int"))
      .withColumn("Soil_Type27", $"Soil_Type27".cast("Int"))
      .withColumn("Soil_Type28", $"Soil_Type28".cast("Int"))
      .withColumn("Soil_Type29", $"Soil_Type29".cast("Int"))
      .withColumn("Soil_Type30", $"Soil_Type30".cast("Int"))
      .withColumn("Soil_Type31", $"Soil_Type31".cast("Int"))
      .withColumn("Soil_Type32", $"Soil_Type32".cast("Int"))
      .withColumn("Soil_Type33", $"Soil_Type33".cast("Int"))
      .withColumn("Soil_Type34", $"Soil_Type34".cast("Int"))
      .withColumn("Soil_Type35", $"Soil_Type35".cast("Int"))
      .withColumn("Soil_Type36", $"Soil_Type36".cast("Int"))
      .withColumn("Soil_Type37", $"Soil_Type37".cast("Int"))
      .withColumn("Soil_Type38", $"Soil_Type38".cast("Int"))
      .withColumn("Soil_Type39", $"Soil_Type39".cast("Int"))
      .withColumn("Soil_Type40", $"Soil_Type40".cast("Int"))**/

    val testCasted = test
      .withColumn("Elevation", $"Elevation".cast("Int"))
      .withColumn("Aspect", $"Aspect".cast("Int"))
      .withColumn("Slope", $"Slope".cast("Int"))
      .withColumn("Horizontal_Distance_To_Hydrology", $"Horizontal_Distance_To_Hydrology".cast("Int"))
      .withColumn("Vertical_Distance_To_Hydrology", $"Vertical_Distance_To_Hydrology".cast("Int"))
      .withColumn("Horizontal_Distance_To_Roadways", $"Horizontal_Distance_To_Roadways".cast("Int"))
      .withColumn("Hillshade_9am", $"Hillshade_9am".cast("Int"))
      .withColumn("Hillshade_Noon", $"Hillshade_Noon".cast("Int"))
      .withColumn("Hillshade_3pm", $"Hillshade_3pm".cast("Int"))
      .withColumn("Horizontal_Distance_To_Fire_Points", $"Horizontal_Distance_To_Fire_Points".cast("Int"))
      /** .withColumn("Soil_Type2", $"Soil_Type2".cast("Int"))
      .withColumn("Soil_Type3", $"Soil_Type3".cast("Int"))
      .withColumn("Soil_Type4", $"Soil_Type4".cast("Int"))
      .withColumn("Soil_Type5", $"Soil_Type5".cast("Int"))
      .withColumn("Soil_Type6", $"Soil_Type6".cast("Int"))
      .withColumn("Soil_Type7", $"Soil_Type7".cast("Int"))
      .withColumn("Soil_Type8", $"Soil_Type8".cast("Int"))
      .withColumn("Soil_Type9", $"Soil_Type9".cast("Int"))
      .withColumn("Soil_Type10", $"Soil_Type10".cast("Int"))
      .withColumn("Soil_Type11", $"Soil_Type11".cast("Int"))
      .withColumn("Soil_Type12", $"Soil_Type12".cast("Int"))
      .withColumn("Soil_Type13", $"Soil_Type13".cast("Int"))
      .withColumn("Soil_Type14", $"Soil_Type14".cast("Int"))
      .withColumn("Soil_Type15", $"Soil_Type15".cast("Int"))
      .withColumn("Soil_Type16", $"Soil_Type16".cast("Int"))
      .withColumn("Soil_Type17", $"Soil_Type17".cast("Int"))
      .withColumn("Soil_Type18", $"Soil_Type18".cast("Int"))
      .withColumn("Soil_Type19", $"Soil_Type19".cast("Int"))
      .withColumn("Soil_Type20", $"Soil_Type20".cast("Int"))
      .withColumn("Soil_Type21", $"Soil_Type21".cast("Int"))
      .withColumn("Soil_Type22", $"Soil_Type22".cast("Int"))
      .withColumn("Soil_Type23", $"Soil_Type23".cast("Int"))
      .withColumn("Soil_Type24", $"Soil_Type24".cast("Int"))
      .withColumn("Soil_Type25", $"Soil_Type25".cast("Int"))
      .withColumn("Soil_Type26", $"Soil_Type26".cast("Int"))
      .withColumn("Soil_Type27", $"Soil_Type27".cast("Int"))
      .withColumn("Soil_Type28", $"Soil_Type28".cast("Int"))
      .withColumn("Soil_Type29", $"Soil_Type29".cast("Int"))
      .withColumn("Soil_Type30", $"Soil_Type30".cast("Int"))
      .withColumn("Soil_Type31", $"Soil_Type31".cast("Int"))
      .withColumn("Soil_Type32", $"Soil_Type32".cast("Int"))
      .withColumn("Soil_Type33", $"Soil_Type33".cast("Int"))
      .withColumn("Soil_Type34", $"Soil_Type34".cast("Int"))
      .withColumn("Soil_Type35", $"Soil_Type35".cast("Int"))
      .withColumn("Soil_Type36", $"Soil_Type36".cast("Int"))
      .withColumn("Soil_Type37", $"Soil_Type37".cast("Int"))
      .withColumn("Soil_Type38", $"Soil_Type38".cast("Int"))
      .withColumn("Soil_Type39", $"Soil_Type39".cast("Int"))
      .withColumn("Soil_Type40", $"Soil_Type40".cast("Int"))**/


    /** Category variables to numeric scores **/
    val indexer = new StringIndexer()
      .setInputCol("Cover_Type")
      .setOutputCol("label")
      .setHandleInvalid("keep")
      .fit(trainCasted)

/**   val indexed = indexer.fit(train).transform(train)

    println(s"Transformed string column '${indexer.getInputCol}' " +
      s"to indexed column '${indexer.getOutputCol}'")
    indexed.select($"Cover_Type",$"label").show()



    val inputColSchema = indexed.schema(indexer.getOutputCol)**/

    /** Elevation**/
    //train.groupBy("Elevation").count.orderBy($"Elevation".asc).show(99)
    // from 1863 to 2037 in each category
    /** Aspect **/
    //train.groupBy("Aspect").count.orderBy($"Aspect".asc).show(99)
    // from 0 (110) to 187 (24)
    /** Soil_Type1 **/
    //train.groupBy("Soil_Type1").count.orderBy($"Soil_Type1".asc).show(99)
    // 0: 14000, 1: 355
    /** Soil_Type6 **/
    //train.groupBy("Soil_Type6").count.orderBy($"count".desc).show(99)
    //0: 14470, 1:650
    /** Soil_Type14 **/
    //train.groupBy("Soil_Type14").count.orderBy($"count".desc).show(99)
    //0: 14951, 1: 169



    //train.printSchema()



    /** boolean, categorical variables **/
    val indexerWilderness_Area1 = new StringIndexer()
      .setInputCol("Wilderness_Area1")
      .setOutputCol("Wilderness_Area1_indexed")
      .setHandleInvalid("keep")
    val indexerWilderness_Area2 = new StringIndexer()
      .setInputCol("Wilderness_Area2")
      .setOutputCol("Wilderness_Area2_indexed")
      .setHandleInvalid("keep")
    val indexerWilderness_Area3 = new StringIndexer()
      .setInputCol("Wilderness_Area3")
      .setOutputCol("Wilderness_Area3_indexed")
      .setHandleInvalid("keep")
    val indexerWilderness_Area4 = new StringIndexer()
      .setInputCol("Wilderness_Area4")
      .setOutputCol("Wilderness_Area4_indexed")
      .setHandleInvalid("keep")

   val indexerSoil_Type1 = new StringIndexer()
      .setInputCol("Soil_Type1")
      .setOutputCol("Soil_Type1_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type2 = new StringIndexer()
      .setInputCol("Soil_Type2")
      .setOutputCol("Soil_Type2_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type3 = new StringIndexer()
      .setInputCol("Soil_Type3")
      .setOutputCol("Soil_Type3_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type4 = new StringIndexer()
      .setInputCol("Soil_Type4")
      .setOutputCol("Soil_Type4_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type5 = new StringIndexer()
      .setInputCol("Soil_Type5")
      .setOutputCol("Soil_Type5_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type6 = new StringIndexer()
      .setInputCol("Soil_Type6")
      .setOutputCol("Soil_Type6_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type7 = new StringIndexer()
      .setInputCol("Soil_Type7")
      .setOutputCol("Soil_Type7_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type8 = new StringIndexer()
      .setInputCol("Soil_Type8")
      .setOutputCol("Soil_Type8_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type9 = new StringIndexer()
      .setInputCol("Soil_Type9")
      .setOutputCol("Soil_Type9_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type10 = new StringIndexer()
      .setInputCol("Soil_Type10")
      .setOutputCol("Soil_Type10_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type11 = new StringIndexer()
      .setInputCol("Soil_Type11")
      .setOutputCol("Soil_Type11_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type12 = new StringIndexer()
      .setInputCol("Soil_Type12")
      .setOutputCol("Soil_Type12_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type13 = new StringIndexer()
      .setInputCol("Soil_Type13")
      .setOutputCol("Soil_Type13_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type14 = new StringIndexer()
      .setInputCol("Soil_Type14")
      .setOutputCol("Soil_Type14_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type15 = new StringIndexer()
      .setInputCol("Soil_Type15")
      .setOutputCol("Soil_Type15_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type16 = new StringIndexer()
      .setInputCol("Soil_Type16")
      .setOutputCol("Soil_Type16_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type17 = new StringIndexer()
      .setInputCol("Soil_Type17")
      .setOutputCol("Soil_Type17_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type18 = new StringIndexer()
      .setInputCol("Soil_Type18")
      .setOutputCol("Soil_Type18_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type19 = new StringIndexer()
      .setInputCol("Soil_Type19")
      .setOutputCol("Soil_Type19_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type20 = new StringIndexer()
      .setInputCol("Soil_Type20")
      .setOutputCol("Soil_Type20_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type21 = new StringIndexer()
      .setInputCol("Soil_Type21")
      .setOutputCol("Soil_Type21_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type22 = new StringIndexer()
      .setInputCol("Soil_Type22")
      .setOutputCol("Soil_Type22_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type23 = new StringIndexer()
      .setInputCol("Soil_Type23")
      .setOutputCol("Soil_Type23_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type24 = new StringIndexer()
      .setInputCol("Soil_Type24")
      .setOutputCol("Soil_Type24_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type25 = new StringIndexer()
      .setInputCol("Soil_Type25")
      .setOutputCol("Soil_Type25_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type26 = new StringIndexer()
      .setInputCol("Soil_Type26")
      .setOutputCol("Soil_Type26_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type27 = new StringIndexer()
      .setInputCol("Soil_Type27")
      .setOutputCol("Soil_Type27_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type28 = new StringIndexer()
      .setInputCol("Soil_Type28")
      .setOutputCol("Soil_Type28_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type29 = new StringIndexer()
      .setInputCol("Soil_Type29")
      .setOutputCol("Soil_Type29_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type30 = new StringIndexer()
      .setInputCol("Soil_Type30")
      .setOutputCol("Soil_Type30_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type31 = new StringIndexer()
      .setInputCol("Soil_Type31")
      .setOutputCol("Soil_Type31_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type32 = new StringIndexer()
      .setInputCol("Soil_Type32")
      .setOutputCol("Soil_Type32_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type33 = new StringIndexer()
      .setInputCol("Soil_Type33")
      .setOutputCol("Soil_Type33_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type34 = new StringIndexer()
      .setInputCol("Soil_Type34")
      .setOutputCol("Soil_Type34_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type35 = new StringIndexer()
      .setInputCol("Soil_Type35")
      .setOutputCol("Soil_Type35_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type36 = new StringIndexer()
      .setInputCol("Soil_Type36")
      .setOutputCol("Soil_Type36_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type37 = new StringIndexer()
      .setInputCol("Soil_Type37")
      .setOutputCol("Soil_Type37_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type38 = new StringIndexer()
      .setInputCol("Soil_Type38")
      .setOutputCol("Soil_Type38_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type39 = new StringIndexer()
      .setInputCol("Soil_Type39")
      .setOutputCol("Soil_Type39_indexed")
      .setHandleInvalid("keep")
    val indexerSoil_Type40 = new StringIndexer()
      .setInputCol("Soil_Type40")
      .setOutputCol("Soil_Type40_indexed")
      .setHandleInvalid("keep")



    val assembler = new VectorAssembler()
      .setInputCols(Array("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
        "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1_indexed", "Wilderness_Area2_indexed",
        "Wilderness_Area3_indexed", "Wilderness_Area4_indexed","Soil_Type1_indexed","Soil_Type2_indexed",
        "Soil_Type3_indexed","Soil_Type4_indexed","Soil_Type5_indexed","Soil_Type6_indexed",
        "Soil_Type9_indexed","Soil_Type10_indexed","Soil_Type11_indexed","Soil_Type12_indexed",
        "Soil_Type13_indexed","Soil_Type14_indexed","Soil_Type16_indexed","Soil_Type17_indexed",
        "Soil_Type18_indexed","Soil_Type19_indexed","Soil_Type20_indexed","Soil_Type21_indexed","Soil_Type22_indexed",
        "Soil_Type23_indexed","Soil_Type24_indexed","Soil_Type26_indexed","Soil_Type27_indexed",
        "Soil_Type28_indexed","Soil_Type29_indexed","Soil_Type30_indexed","Soil_Type31_indexed","Soil_Type32_indexed",
        "Soil_Type33_indexed","Soil_Type34_indexed","Soil_Type35_indexed","Soil_Type36_indexed","Soil_Type37_indexed",
        "Soil_Type38_indexed","Soil_Type39_indexed","Soil_Type40_indexed"))
      .setOutputCol("features")




/**        val assembler = new VectorAssembler()
      .setInputCols(Array("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
        "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1_indexed",
        "Wilderness_Area3_indexed", "Wilderness_Area4_indexed","Soil_Type3_indexed","Soil_Type4_indexed",
        "Soil_Type10_indexed","Soil_Type30_indexed", "Soil_Type38_indexed","Soil_Type39_indexed","Soil_Type40_indexed"))
      .setOutputCol("features")


    val assembler = new VectorAssembler()
      .setInputCols(Array("Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
          "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
          "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2",
          "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5",
          "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12",
          "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
          "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26",
          "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33",
          "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"))
      .setOutputCol("features")

    **/

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setImpurity("entropy")
      .setFeatureSubsetStrategy("sqrt") //auto
    //.setMaxDepth("None")
      //.setNumTrees(10)
      .setMaxBins(300)
      //.setSeed(1234)

   // val pipeline = new Pipeline()
   //   .setStages(Array(indexer,assembler,rf))


    /** Back to original category for Cover_Type**/
    val converter = new IndexToString()
      .setInputCol("predictions")
      .setOutputCol("OriginalLabel")
      .setLabels(indexer.labels)
    //  val converted = converter.transform(indexed)

    //converted.show(20)





    val pipeline = new Pipeline()
      .setStages(Array(indexer,indexerWilderness_Area1,indexerWilderness_Area2,indexerWilderness_Area3,
        indexerWilderness_Area4,indexerSoil_Type1,indexerSoil_Type2,indexerSoil_Type3,indexerSoil_Type4,
        indexerSoil_Type5,indexerSoil_Type6,indexerSoil_Type9,indexerSoil_Type10,
        indexerSoil_Type11,indexerSoil_Type12,indexerSoil_Type13,indexerSoil_Type14,
        indexerSoil_Type16,indexerSoil_Type17,indexerSoil_Type18,indexerSoil_Type19,indexerSoil_Type20,
        indexerSoil_Type21,indexerSoil_Type22,indexerSoil_Type23,indexerSoil_Type24,
        indexerSoil_Type26,indexerSoil_Type27,indexerSoil_Type28,indexerSoil_Type29,indexerSoil_Type30,
        indexerSoil_Type31,indexerSoil_Type32,indexerSoil_Type33,indexerSoil_Type34,indexerSoil_Type35,
        indexerSoil_Type36,indexerSoil_Type37,indexerSoil_Type38,indexerSoil_Type39,indexerSoil_Type40,assembler,rf,
        converter))

 /**

 val pipeline = new Pipeline()
      .setStages(Array(indexer,indexerWilderness_Area1,indexerWilderness_Area3,indexerWilderness_Area4,
        indexerSoil_Type3,indexerSoil_Type4,indexerSoil_Type10,indexerSoil_Type30,indexerSoil_Type38,
        indexerSoil_Type39,indexerSoil_Type40,assembler,rf,converter))



   **/

    //val splits = train.randomSplit(Array(0.9,0.1))
    //val (training, testing) = (splits(0), splits(1))


    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(20))
      //.addGrid(rf.maxBins, Array(5,10,15,20,25))
      .addGrid(rf.maxDepth,Array(30)) //useful to limit this to avoid overfitting the training data
      //.addGrid(rf.impurity,Array("entropy","gini"))
      //.addGrid(rf.featureSubsetStrategy,Array("auto","sqrt"))
      //.addGrid(rf.minInfoGain,Seq(0.01,0.05)) //can help the model resist overfitting, because decisions that barely help divide the training input may in fact not helpfully divide future data at all
      .build()



    val cv = new CrossValidator()
      //.setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("predictions")
         .setMetricName("accuracy"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)



    // TrainValidationSplit will try all combinations of values and determine best model using the evaluator

/**   val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("predictions")
      //.setPredictionCol("predictionsLabeled")
      //.setMetricName("f1")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

  **/


  //  trainCasted.printSchema()
  //  testCasted.printSchema()

    //val model = trainValidationSplit.fit(trainCasted)
    val model = cv.fit(trainCasted)
    //val model = pipeline.fit(trainCasted)



    val df_WithPredictions = model.transform(testCasted)

    //To have a column int and named Cover_Type as requiered in Kaggle
   // val df_WithPredictionsCasted = df_WithPredictions
   //   .withColumn("Cover_Type", $"predictions".cast("Int"))



     val df_WithPredictionsSubmit = df_WithPredictions
       .withColumn("Cover_Type", $"OriginalLabel".cast("Int"))

    //df_WithPredictions.groupBy("Cover_Type").count.orderBy($"count".desc).show(99)
    df_WithPredictions.groupBy("predictions").count.orderBy($"count".desc).show(99)
    df_WithPredictions.groupBy("OriginalLabel").count.orderBy($"count".desc).show(99)
    //df_WithPredictionsCasted.groupBy("predictions").count.orderBy($"count".desc).show(99)
   // converted.groupBy("predictions").count.orderBy($"count".desc).show(99)

    df_WithPredictions.select($"Id",$"OriginalLabel").show(20)
/**   df_WithPredictionsCasted.select($"Id",$"Cover_Type").show(10)

  **/

    df_WithPredictionsSubmit.select($"Id",$"Cover_Type")
      .write.mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv("/Users/Manon/Documents/Cours/Ms/Big Data Mining/Kaggle/results_scala.csv")



  }
}
