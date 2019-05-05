import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer}
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{VectorAssembler}


object Indexer {

  def processToIndexation(data: DataFrame): Pipeline = {


    val comment_textIndexer =  new StringIndexer().setInputCol ("comment_text").setOutputCol ("comment_textIndex")

    val encoder = new VectorAssembler()
      .setInputCols(Array("severe_toxicity",
        "obscene", "identity_attack", "insult", "threat"))
      .setOutputCol("features")


    return new Pipeline().setStages(Array(comment_textIndexer, encoder))

  }

}
