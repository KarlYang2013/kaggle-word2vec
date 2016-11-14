import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{CountVectorizer, StopWordsRemover, Word2Vec}
import com.databricks.spark.corenlp.functions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.jsoup.Jsoup


object Word2VecMain {

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

    val train = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .option("inferSchema", "true")
      .load("src/main/resources/data/labeledTrainData.tsv")

    val unlabeledTrain = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .option("inferSchema", "true")
      .load("src/main/resources/data/unlabeledTrainData.tsv")

    val test = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .option("inferSchema", "true")
      .load("src/main/resources/data/testData.tsv")

    val data = train.select("review").union(unlabeledTrain.select("review"))
      .select(cleanxml('review).as('doc))
      .select(ssplit('doc).as('sentence))
      .select(toWordsLists('sentence).as('words))

    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("features")
      .setWindowSize(10)
      .setMinCount(40)

    val model = word2Vec.fit(data)

    model.findSynonyms("music", 10).show(truncate = false)

//    val model = pipeline.fit(cleanedUnlabeled)

//    val predictions = model.transform(cleanedTest)
//    predictions.selectExpr("id", "cast(prediction as int) sentiment")
//      .repartition(1)
//      .write
//      .format("com.databricks.spark.csv")
//      .option("header", "true")
//      .save("target/predictions.csv")
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