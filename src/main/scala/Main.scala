import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{CountVectorizer, StopWordsRemover, Word2Vec}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
import org.jsoup.Jsoup


object Main {

  val conf = new SparkConf()
    .setAppName("mlpoc")
    .setMaster("local[*]")
    .set("spark.sql.shuffle.partitions", "2")

  val sc = new SparkContext(conf)
  val sqlContext = SparkSession.builder.config(conf).getOrCreate()

  def sentimentToDouble(cleanedTraining: DataFrame): DataFrame = {
    cleanedTraining.withColumn("sentimentDouble", cleanedTraining("sentiment").cast("double"))
      .drop("sentiment")
      .withColumnRenamed("sentimentDouble", "sentiment")
  }

  def main(args: Array[String]): Unit = {

    val training = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .option("inferSchema", "true")
      .load("src/main/resources/data/labeledTrainData.tsv")

    val test = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .option("inferSchema", "true")
      .load("src/main/resources/data/testData.tsv")

    var cleanedTraining = cleanData(training)
    cleanedTraining = sentimentToDouble(cleanedTraining)
    val cleanedTest = cleanData(test)

    cleanedTraining = cleanedTraining

    cleanedTraining.printSchema()

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("words2")

    val countVectorizer = new CountVectorizer()
      .setInputCol("words2")
      .setOutputCol("features")
      .setVocabSize(5000)

    val randomForest = new RandomForestClassifier()
      .setLabelCol("sentiment")
      .setFeaturesCol("features")
      .setNumTrees(100)

    val pipeline = new Pipeline().setStages(
      Array(
        stopWordsRemover,
        countVectorizer
        , randomForest
      )
    )

    val model = pipeline.fit(cleanedTraining)

    val predictions = model.transform(cleanedTest)
    predictions.selectExpr("id", "cast(prediction as int) sentiment")
      .repartition(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save("target/predictions.csv")
  }

  private def cleanData(dataFrame: DataFrame) = {
    val toCleanWordsFunction = (s: String) =>
      Jsoup.parse(s).text() // remove all html tags
      .replaceAll("[^a-zA-Z]", " ") // keep letters only
      .replaceAll("\\s+", " ") // remove extra empty characters
      .toLowerCase
      .split(" ")
      .toSeq

    dataFrame
      .withColumn("words", udf(toCleanWordsFunction).apply(dataFrame("review")))
      .drop("review")
  }
}