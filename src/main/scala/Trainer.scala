
import org.apache.spark.sql.{DataFrame, SparkSession}

import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel

/**
  *  All the functions linking to the trainning part of the program
  */
object Trainer {

  val schemaStructure = StructType(
    StructField("id", IntegerType, true) ::
      StructField("target", DoubleType, true) ::
      StructField("comment_text", StringType, true) ::
      StructField("severe_toxicity", DoubleType, true) ::
      StructField("obscene", DoubleType, true) ::
      StructField("identity_attack", DoubleType, true) ::
      StructField("insult", DoubleType, true) ::
      StructField("threat", DoubleType, true) ::
      StructField("asian", DoubleType, true) ::
      StructField("atheist", DoubleType, true) ::
      StructField("bisexual", DoubleType, true) ::
      StructField("black", DoubleType, true) ::
      StructField("buddhist", DoubleType, true) ::
      StructField("christian", DoubleType, true) ::
      StructField("female", DoubleType, true) ::
      StructField("heterosexual", DoubleType, true) ::
      StructField("hindu", DoubleType, true) ::
      StructField("homosexual_gay_or_lesbian", DoubleType, true) ::
      StructField("intellectual_or_learning_disability", DoubleType, true) ::
      StructField("jewish", DoubleType, true) ::
      StructField("latino", DoubleType, true) ::
      StructField("male", DoubleType, true) ::
      StructField("muslim", DoubleType, true) ::
      StructField("other_disability", DoubleType, true) ::
      StructField("other_gender", DoubleType, true) ::
      StructField("other_race_or_ethnicity", DoubleType, true) ::
      StructField("other_religion", DoubleType, true) ::
      StructField("other_sexual_orientation", DoubleType, true) ::
      StructField("physical_disability", DoubleType, true) ::
      StructField("psychiatric_or_mental_illness", DoubleType, true) ::
      StructField("transgender", DoubleType, true) ::
      StructField("white", DoubleType, true) ::
      StructField("created_date", DateType, true) ::
      StructField("publication_id", IntegerType, true) ::
      StructField("parent_id", DoubleType, true) ::
      StructField("article_id", DoubleType, true) ::
      StructField("rating", StringType, true) ::
      StructField("funny", DoubleType, true) ::
      StructField("wow", DoubleType, true) ::
      StructField("sad", DoubleType, true) ::
      StructField("likes", DoubleType, true) ::
      StructField("disagree", DoubleType, true) ::
      StructField("sexual_explicit", DoubleType, true) ::
      StructField("identity_annotator_count", DoubleType, true) ::
      StructField("toxicity_annotator_count", DoubleType, true) :: Nil
  )

  /**
    *
    * @param trainFilePath A string respresenting the path of the file to load
    * @param sparkSession The spark session started at the beginning of the program
    * @return A pipeline with the model and the last part of unused Dataframe of the training set to run the prediction
    */
  def trainModel(trainFilePath: String, sparkSession : SparkSession): Tuple2[PipelineModel, DataFrame] = {

    //Loading the training set file
    val trainningData: DataFrame = sparkSession.read.format("csv")
      .schema(schemaStructure)
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load(trainFilePath)

    //Running cleaning data process
    val cleanedData = Cleaner.cleanData(trainningData, sparkSession)

    println("-------- Cleanning Data -------------------")

    //Running data indexation process
    val indexPipeline = Indexer.processToIndexation(cleanedData)
    val trainingDataFormated = indexPipeline.fit(cleanedData).transform(cleanedData).select("target", "features")

    trainingDataFormated.cache()

    //Split Data in two to use a part for training and a part for testing the model 80% to train cause little set used
    val Array(trainingDataSplitted, testDataSplitted) = trainingDataFormated.randomSplit(Array(0.8, 0.2))




    println("-------- Training Data -------------------")

    //Running logistic regression
    val reg = new LogisticRegression()
      .setLabelCol("target")
      .setFeaturesCol("features")
      .setAggregationDepth(10)
      .setElasticNetParam(0.5)
      .setFitIntercept(true)
      .setMaxIter(10)
      .setRegParam(0.05)

    val pip = new Pipeline().setStages(Array(
      reg
    ))

    println("--------------- Training started -------------------")
    val lrModel: PipelineModel = pip.fit(trainingDataSplitted)

    val lrm: LogisticRegressionModel = lrModel
            .stages
            .last.asInstanceOf[LogisticRegressionModel]

    println("Coefficients: " + lrm.coefficients)
    println(" Intercept: " + lrm.intercept)
    
  
    lrModel.write.overwrite().save("linear-regression-model")

    val model = PipelineModel.read.load("linear-regression-model")

    println("-------- Tranning model is finiched -------------------")
    
    

    return Tuple2(model, testDataSplitted)
  }

}
