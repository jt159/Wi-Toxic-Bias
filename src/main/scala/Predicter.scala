import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SparkSession}



object Predicter {

    def runPrediction(model:PipelineModel , testSet:DataFrame , spark: SparkSession):Boolean = {
        println("--------------- Predicter started -------------------")
        val prediction = model.transform(testSet)
        val predictionValues = prediction.select("target", "prediction")

        println("--------------- Training finished -------------------")
        println("-----------------------------------------------------")
        println("--------------- Results of prediction ---------------")

        val truePositive = predictionValues.filter(predictionValues.col("prediction") === 1.0).filter(predictionValues.col("target") === predictionValues.col("prediction")).count()
        val trueNegative = predictionValues.filter(predictionValues.col("prediction") === 1.0).filter(predictionValues.col("target") =!= predictionValues.col("prediction")).count()
        val falseNegative = predictionValues.filter(predictionValues.col("prediction") === 0.0).filter(predictionValues.col("target") =!= predictionValues.col("prediction")).count()
        val falsePositive = predictionValues.filter(predictionValues.col("prediction") === 0.0).filter(predictionValues.col("target") === predictionValues.col("prediction")).count()

        println("True Positive : " + truePositive)
        println("True Negative : " + trueNegative)
        println("False Negative : " + falseNegative)
        println("False Positive : " + falsePositive)

        val recall = truePositive / (truePositive + falseNegative).toDouble
        val precision = truePositive / (truePositive + falsePositive).toDouble

        println("Recall " + recall)
        println("Precision" + precision)



        return true

    }
    
}