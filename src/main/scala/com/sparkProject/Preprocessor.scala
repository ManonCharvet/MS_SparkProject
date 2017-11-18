package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.functions.udf

object Preprocessor {

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


    import spark.implicits._ /**permet d'utiliser le $ pour colonne**/

    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** 1 - CHARGEMENT DES DONNEES **/
    /**a)option("header","true") pour considérer la premiere ligne comme le nom des colonnes **/
    /**option("nullValue","false") Coder les false comme des valeurs nulles**/
    val data: DataFrame = spark
        .read
        .option("header", "true")
        .option("nullValue","false")
        .csv("train.csv")


    /**b)Afficher le nombre de lignes et le nombre de colonnes dans le dataFrame.**/
    println("Nombre de lignes du dataframe:")
    println(data.count())
    println("Nombre de colonnes du dataframe:")
    /**data.columns.foreach(println)**/
    println(data.columns.size)

    /**c)Afficher le dataFrame sous forme de table.**/
    data.show()

    /**d)Afficher le schéma du dataFrame (nom des colonnes et le type des données contenues dans chacune d’elles).**/
    data.printSchema()

    /**e)Assigner le type “Int” aux colonnes qui vous semblent contenir des entiers.**/
    data.withColumn("backers_count", $"backers_count".cast(sql.types.IntegerType))
      .withColumn("final_status", $"final_status".cast(sql.types.IntegerType))
      .withColumn("launched_at", $"launched_at".cast(sql.types.IntegerType))
      .withColumn("deadline", $"deadline".cast(sql.types.IntegerType))
      .printSchema()
      /**.toDF("project_id","name","desc","goal","keywords","disable_communication")

    val data_integer = data.withColumn("backers_count", $"backers_count".cast(IntegerType))**/



    /** 2 - CLEANING **/
    /** FEATURE ENGINEERING: Ajouter et manipuler des colonnes: Certaines opérations sur les colonnes sont déjà
      * implémentées dans Spark, mais il est souvent nécessaire de faire appel à des fonctions plus complexes. Dans
      * ce cas on peut créer des UDF (User Defined Functions) qui permettent d’implémenter de nouvelles opérations
      * sur les colonnes.
      */

    /**a) Afficher une description statistique des colonnes de type Int**/
    data.describe("backers_count", "final_status").show /**ATTENTION "USD" en valeurs!! Pas de sens**/

    /**b) Observer les autres colonnes, et proposer des cleanings à faire sur les données**/
    data.describe("disable_communication", "country", "currency").show
    /**groupBy count
      * dropDuplicates
      */



    /**c) enlever la colonne "disable_communication". cette colonne est très largement majoritairement à "false"**/
    data.drop(data.col("disable_communication")).printSchema()

    /**d) Fuites du futur: "backers_count". Il s'agit du nombre de personnes FINAL ayant investi dans chaque projet,
      * or ce nombre n'est connu qu'après la fin de la campagne. Quand on voudra appliquer notre modèle, les données
      * du futur ne sont pas présentes. On ne peut donc pas les utiliser comme input pour un modèle.
       */
    data.drop(data.col("backers_count"))
      .drop(data.col("state_changed_at"))
      .printSchema()


    /**e) "currency" et "country" il semble y avoir des inversions entre ces deux colonnes et du nettoyage à faire
      * en utilisant les deux colonnes.
      * En particulier on peut remarquer que quand country=null le country à l'air d'être dans currency.
      */
    data.filter($"country".isNull).groupBy("currency").count.orderBy($"count".desc).show(50)

    /**UDFS**/
    /**Country**/
    // Define your udf:
    def Udf_country = udf{(country: String, currency: String) =>
      if (country == null)
        currency
      else
        country
    }
    // To use the created UDF:
    val dataCountryFixed: DataFrame = data.withColumn("country2", Udf_country($"country", $"currency"))
    println("Probleme country ")


    /**Currency**/
    def Udf_currency = udf{(currency: String) =>
      if (currency != null)
        if (currency.length != 3)
          null
        else
          currency
      else
        null
    }
    // To use the created UDF:
    val dataCurrencyFixed: DataFrame = dataCountryFixed.withColumn("currency2", Udf_currency($"currency"))

    println("Probleme currency")

    //print(dataCountryFixed.show(20))

    //dataCurrencyFixed.filter($"country".isNull).groupBy("currency2").count.orderBy($"count".desc).show(10)

    dataCurrencyFixed.filter($"country".isNull).show(10)


    /**f) Afficher le nombre d’éléments de chaque classe: principalement des 0 et des 1**/
    dataCurrencyFixed.groupBy("final_status").count.orderBy($"count".desc).show(50)


    /**g) Conserver uniquement les lignes final_status = 0 (Fail) ou 1 (Success).**/
    val dataFinalStat = dataCurrencyFixed.filter("final_status = 0 or final_status = 1")
    //dataFinalStat.groupBy("final_status").count.orderBy($"count".desc).show(50)



    /**3- Manipuler les colonnes**/
    /**a)Ajoutez une colonne days_campaign qui représente la durée de la campagne en jours **/


    /**______________________________________________________________**/
    /**Tout appliquer**/
    val dataMAJ = data.withColumn("backers_count", $"backers_count".cast(sql.types.IntegerType))
      .withColumn("final_status", $"final_status".cast(sql.types.IntegerType))
      .withColumn("launched_at", $"launched_at".cast(sql.types.IntegerType))
      .withColumn("deadline", $"deadline".cast(sql.types.IntegerType))
      .drop(data.col("disable_communication"))
      .drop(data.col("backers_count"))
      .drop(data.col("state_changed_at"))
      .withColumn("country2", Udf_country($"country", $"currency"))
      .withColumn("currency2", Udf_currency($"currency"))
    /**.printSchema()**/




    def Udf_campaign = udf{(launched_at: Integer, deadline: Integer) =>
      if (deadline != null)
        if (launched_at != null)
          (deadline-launched_at)/86400
        else
          -1
      else
        -1

    }
    // To use the created UDF:
    val dataDaysCampaign: DataFrame = dataMAJ.withColumn("days_campaign", Udf_campaign($"launched_at", $"deadline"))

    //dataDaysCampaign.show(10)


    /**b) Ajoutez une colonne: nombre d’heures de préparation de la campagne entre “created_at” et “launched_at”**/
    def Udf_prepa = udf{(created_at: Integer, launched_at: Integer) =>
      if (created_at != null)
        if (launched_at != null)
          (created_at-launched_at)/3600
        else
          -1
      else
        -1

    }
    // To use the created UDF:
    val dataHoursPrepa: DataFrame = dataDaysCampaign.withColumn("hours_prepa", Udf_prepa($"created_at", $"launched_at"))



    /**c) Supprimer les colonnes “launched_at”, “created_at” et “deadline”, elles ne sont pas exploitables pour un modèle.**/
    val dataCleaner: DataFrame = dataHoursPrepa.drop(data.col("launched_at"))
      .drop(data.col("created_at"))
      .drop(data.col("deadline"))



    /**d) Ajoutez une colonne text, qui contient la concaténation des Strings des colonnes “name”, “desc” et “keywords”.**/

    def Udf_concatenation = udf{(name: String, desc: String, keywords: String) =>

      name + " " + desc + " " + keywords

    }
    // To use the created UDF:
    val dataText: DataFrame = dataCleaner.withColumn("text", Udf_concatenation($"name", $"desc", $"keywords"))



    /**4 -Valeurs nulles**/
    /**Remplacer les valeurs nulles des colonnes "days_campaign”, "hours_prepa", "goal" par la valeur -1.**/

   // val dataFixMissing: DataFrame = dataText.withColumn("days_campaign",regexp_replace(data("days_campaign"), null, "-1"))
   //   .withColumn("hours_prepa",regexp_replace(data("hours_prepa"), null, "-1"))
   //   .withColumn("goal",regexp_replace(data("goal"), null, "-1"))

    val dfReady: DataFrame = dataText
      .filter($"goal" > 0)
      .na
      .fill(Map(
        "days_campaign" -> -1,
        "hours_prepa" -> -1,
        "goal" -> -1
      ))

    //vérifier l'equilibrage pour la classification
    dfReady.groupBy("final_status").count.orderBy($"count".desc).show

    //Filtrer les classes qui nous interessent
    //Final_status contient d'autres etats que failed ou succeed. On ne sait pas s'il faut les
    //enlever ou les considérer comme failed
    val datafiltered = dfReady.filter($"final_status".isin(0,1))

    datafiltered.show(50)
    println(datafiltered.count)

    /**WRITING DATAFRAME, format parquet**/
    datafiltered.write.mode(SaveMode.Overwrite).parquet("/Users/Manon/Documents/Cours/MS/Spark/TP_ParisTech_2017_2018_starter/traincleaned")


    /**_________________________________________________________**/
    /**toutes les premières instructions dans une seule commande**/
    data.withColumn("backers_count", $"backers_count".cast(sql.types.IntegerType))
      .withColumn("final_status", $"final_status".cast(sql.types.IntegerType))
      .withColumn("launched_at", $"launched_at".cast(sql.types.IntegerType))
      .withColumn("deadline", $"deadline".cast(sql.types.IntegerType))
      .drop(data.col("disable_communication"))
      .drop(data.col("backers_count"))
      .drop(data.col("state_changed_at"))
      .withColumn("country2", Udf_country($"country", $"currency"))
      .withColumn("currency2", Udf_currency($"currency"))
      /**.printSchema()**/


    /**Redécaler les colonnes avec un regex**/
    /**data.withColumn("descNew", regexp_replace(data("desc"), ".\"{2,}", " ")).take(5).foreach(println)**/

    /**Ecrit en console**/
    /**val df2 = df.withColumn("replaced",regexp_replace($"value","\"{2,}", " "))
      * df2.select("replaced").write.text("/Users/Manon/Documents/Cours/MS/Spark/TP_ParisTech_2017_2018_starter/train2.csv")
      **/
  }

}
