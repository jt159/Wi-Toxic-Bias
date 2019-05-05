

import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.hadoop.fs.Path


object Main extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

 val spark: SparkSession =
        SparkSession
            .builder()
            .appName("Wi Session 2")
            .config("spark.master", "local")
            .config("spark.driver.memory", "4g")
            .config("num-executors", "20")
            .config("executor-memory", "32g")
            .config("executor-cores", "3")
            .getOrCreate()


  val dataFile = "./src/main/data/train.csv"

  println("======================================")
  println("===== MESSAGE TOXICITY PREDICTION ====")
  println("======================================")

  println("The data training file is located in : " + dataFile)
  println("------------------------------------------")

  
  val Tuple2(model, testSet) = Trainer.trainModel(dataFile, spark)
  
  val result = Predicter.runPrediction(model, testSet, spark)

 

  spark.stop()


}