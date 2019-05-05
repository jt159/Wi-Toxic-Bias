

import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.rand

object Cleaner {



  def cleanData(frameToClean: DataFrame, sparkSession : SparkSession): DataFrame = {

    //FIRST : Select the column I need to use
    val trainingSetWithSelectedColums: DataFrame =  frameToClean.select(
      "target",
      "comment_text",
      "severe_toxicity",
      "obscene",
      "identity_attack",
      "insult",
      "threat"
    )//TODO check if some other columns could be selected

    //CLEANING COLUMNS

    //Put default values to null values
    var trainingSetWithCleanedNulValues = trainingSetWithSelectedColums
    trainingSetWithSelectedColums.columns.foreach( col => {
      if(col == "comment_text") {
        //String case
        trainingSetWithCleanedNulValues = trainingSetWithCleanedNulValues.na.fill("", Seq(col))
      }else{
        //Int Or Double case
        trainingSetWithCleanedNulValues = trainingSetWithCleanedNulValues.na.fill(0.0, Seq(col))
      }
    })


    // SET target column from double to a 0 or 1 value regarding different to 0.5 (default transitional target)
    val targetColumn = trainingSetWithCleanedNulValues("target")
    val cleanerTarget = udf[Double, Double]{x => {if(x > 0.5) 1.0 else 0.0}}
    val cleanedTargetColum = cleanerTarget(targetColumn)
    val trainingSetWithTargetValueConverted = trainingSetWithCleanedNulValues.withColumn("target", cleanedTargetColum)

    //Limited the trainset size since have no solution to unlock JVM
    val limitedSizeTrainingSetWithTargetValueConverted = trainingSetWithTargetValueConverted.orderBy(rand()).limit(300000)

    return limitedSizeTrainingSetWithTargetValueConverted

  }

}
