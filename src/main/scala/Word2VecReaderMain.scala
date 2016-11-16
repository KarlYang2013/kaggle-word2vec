import com.databricks.spark.corenlp.functions._
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}


object Word2VecTrainerMain {

  val conf = new SparkConf()
    .setAppName("mlpoc")
    .setMaster("local[*]")
    .set("spark.sql.shuffle.partitions", "8")

  val sc = new SparkContext(conf)
  val sqlContext = SparkSession.builder.config(conf).getOrCreate()

  def sentimentToDouble(cleanedTraining: DataFrame): DataFrame = {
    cleanedTraining.withColumn("sentimentDouble", cleanedTraining("sentiment").cast("double"))
      .drop("sentiment")
      .withColumnRenamed("sentimentDouble", "sentiment")
  }

  def main(args: Array[String]): Unit = {
    import sqlContext.implicits._

    val model = Word2VecModel.load("word2vec.model")

    model.findSynonyms("music", 10).show(truncate = false)

  }

  def toWordsLists = udf { s: Seq[String] =>
    s.flatMap(
      _.replaceAll("[^a-zA-Z]", " ") // keep letters only
        .replaceAll("\\s+", " ") // remove extra empty characters
        .toLowerCase
        .split(" ")
        .toSeq
    )
  }
}