import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.RegressionMetrics
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.math._


class DecisionTreeRegressor(
    trainRDD: RDD[Array[Double]],
    maxDepth: Int = 5,
    minInstancesPerNode: Int = 1,
    maxBins: Int = 10
) extends Serializable {
    private val X: RDD[Array[Double]] = trainRDD.map(row => row.init).cache()
    private val y: RDD[Double] = trainRDD.map(row => row.last).cache()

    // Build the decision tree
    val tree: Any = buildTree(X, y, 0)

    // Build the decision tree recursively
    private def buildTree(X: RDD[Array[Double]], y: RDD[Double], depth: Int): Any = {
        val nSamples = X.count()
        val numBins = math.min(maxBins, nSamples.toInt)
        val nFeatures = X.first().length
        println(s"<====Building tree at depth $depth with $nSamples samples and $nFeatures features====>")
        
        // Calculate the mean and variance of the target variables
        val yMean = y.mean()
        val variance = y.map(value => math.pow(value - yMean, 2)).mean()

        // Stop conditions
        if (nSamples == 0 || depth >= maxDepth || variance == 0) return yMean

        // Find the best split
        var bestSplit: Option[(Int, Double)] = None
        var bestVariance = Double.PositiveInfinity
        
        for (featureIndex <- 0 until nFeatures) {
            // Compute candidate thresholds once; consider sampling or quantiles here
            val featureRDD: RDD[Double] = X.map(row => row(featureIndex))
            // Define the quantile levels, e.g. 10%, 20%, ..., 90%.
            val quantileLevels = (1 until numBins).map(i => i.toDouble / numBins).toArray
            println(s"Num bins: ${numBins} quantile levels: ${quantileLevels.mkString(", ")}")
            // Compute approximate quantiles for the feature.
            val candidateThresholds = approxQuantile(featureRDD, quantileLevels)

            for (threshold <- candidateThresholds) {
                val leftMask = X.map(row => row(featureIndex) <= threshold)
                val rightMask = X.map(row => row(featureIndex) > threshold)
                
                val leftY = y.zip(leftMask).filter(_._2).map(_._1).cache()
                val rightY = y.zip(rightMask).filter(!_._2).map(_._1).cache()
                
                val leftCount = leftY.count()
                val rightCount = rightY.count()
                
                if (leftCount >= minInstancesPerNode && rightCount >= minInstancesPerNode) {
                val leftMean = leftY.mean()
                val rightMean = rightY.mean()

                val leftVariance = if (leftCount > 0) leftY.map(value => math.pow(value - leftMean, 2)).mean() else 0
                val rightVariance = if (rightCount > 0) rightY.map(value => math.pow(value - rightMean, 2)).mean() else 0

                val weightedVariance = (leftVariance * leftCount + rightVariance * rightCount) / nSamples
                
                if (weightedVariance < bestVariance) {
                    bestVariance = weightedVariance
                    bestSplit = Some((featureIndex, threshold))
                }
                }

                leftY.unpersist()
                rightY.unpersist()
            }
        }

        if (bestSplit.isEmpty) return yMean
        
        val (featureIndex, threshold) = bestSplit.get

        // Split the data into left and right branches
        val leftMask = X.map(row => row(featureIndex) <= threshold)
        val rightMask = X.map(row => row(featureIndex) > threshold)

        val leftX = X.zip(leftMask).filter(_._2).map(_._1).cache()
        val rightX = X.zip(rightMask).filter(!_._2).map(_._1).cache()

        val leftY = y.zip(leftMask).filter(_._2).map(_._1).cache()
        val rightY = y.zip(rightMask).filter(!_._2).map(_._1).cache()
        
        // Recursively build trees
        val leftTree = buildTree(leftX, leftY, depth + 1)
        val rightTree = buildTree(rightX, rightY, depth + 1)
        
        leftX.unpersist()
        rightX.unpersist()
        leftY.unpersist()
        rightY.unpersist()

        Map(
        "featureIndex" -> featureIndex,
        "threshold" -> threshold,
        "left" -> leftTree,
        "right" -> rightTree
        )
    }

    def approxQuantile(
        rdd: RDD[Double],
        probalities: Array[Double],
        sampleFraction: Double = 0.1
    ): Array[Double] = {
        val sampledRDD = rdd.sample(withReplacement = false, sampleFraction).collect()
        
        val sortedSample = sampledRDD.sorted
        val n = sortedSample.length

        // For each quantile level q, compute the index in the sorted array.
        probalities.map { q =>
            val index = math.min(n - 1, math.floor(q * n).toInt)
            sortedSample(index)
        }
    }

    // Traverse the tree to make predictions
    private def traverseTree(tree: Any, X: Array[Double]): Double = {
        tree match {
        case node if node.isInstanceOf[Map[String, Any]] =>
            val mapNode = node.asInstanceOf[Map[String, Any]]
            val featureIndex = mapNode("featureIndex").asInstanceOf[Int]
            val threshold = mapNode("threshold").asInstanceOf[Double]
            if (X(featureIndex) <= threshold) {
            traverseTree(mapNode("left"), X)
            } else {
            traverseTree(mapNode("right"), X)
            }
        case leaf: Double => leaf
        case None => 
            println("Warning: Encountered None in the tree. Returning default value 0.0.")
            0.0
        }
    }

    def predict(testRDD: RDD[Array[Double]]): RDD[Double] = {
        testRDD.map(features => traverseTree(tree, features))
    }
}

object Low_Level_Operations {
    def main(args: Array[String]): Unit = {
        // Initialize SparkSession
        val spark = SparkSession.builder()
                                .appName("DecisionTreeRegressor")
                                .master("local[*]")
                                .config("spark.logConfig", "false")
                                .getOrCreate()

        val sc = spark.sparkContext
        sc.setLogLevel("WARN")

        // Load the CSV file as a text file and filter out the header
        val lines: RDD[String] = sc.textFile("train.csv")
        val header: String = lines.first()
        val dataRDD: RDD[String] = lines.filter(line => line != header)

        // Define a function to process each line
        def processTripLine(line: String): Array[Double] = {
            val fields = line.split(",")
            val pickupDatetime = LocalDateTime.parse(fields(2), DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
            val passengerCount = fields(4).toInt
            val pickupLongitude = fields(5).toDouble
            val pickupLatitude = fields(6).toDouble
            val dropoffLongitude = fields(7).toDouble
            val dropoffLatitude = fields(8).toDouble
            val tripDuration = fields(10).toInt

            val pickupMinutes = pickupDatetime.getHour * 60 + pickupDatetime.getMinute
            val pickupDayOfWeek = pickupDatetime.getDayOfWeek.getValue 
            val pickupMonth = pickupDatetime.getMonthValue
            val distance = sqrt(pow(pickupLongitude - dropoffLongitude, 2) + pow(pickupLatitude - dropoffLatitude, 2))

            Array(
                passengerCount.toDouble,
                pickupLatitude,
                pickupLongitude,
                distance,
                pickupMinutes.toDouble,
                pickupDayOfWeek.toDouble,
                pickupMonth.toDouble,
                tripDuration.toDouble
            )
        }

        // Apply the processing function to each line
        val processedRDD: RDD[Array[Double]] = dataRDD.map(processTripLine)

        // Filter the data based on the conditions
        val filteredRDD: RDD[Array[Double]] = processedRDD.filter(row => row(0) > 0) // passenger_count > 0
                                                        .filter(row => row(7) < 22 * 3600) // trip_duration < 22 hours
                                                        .filter(row => row(3) > 0) // distance > 0

        // Print the first few rows for verification
        filteredRDD.take(5).foreach(row => println(row.mkString(", ")))
        
        val Array(trainRDD, testRDD) = filteredRDD.randomSplit(Array(0.8, 0.2), seed = 42)

        val regressor = new DecisionTreeRegressor(trainRDD, maxDepth = 5, minInstancesPerNode = 1)

        // Predict on the test set
        val predictions: RDD[Double] = regressor.predict(testRDD.map(row => row.init))

        // Combine predictions with actual labels
        val predictionsAndLabels: RDD[(Double, Double)] = predictions.zip(testRDD.map(row => row.last))
        
        val metrics = new RegressionMetrics(predictionsAndLabels)

        println(s"Root Mean Squared Error (RMSE): ${metrics.rootMeanSquaredError}")
        println(s"Mean Absolute Error (MAE): ${metrics.meanAbsoluteError}")

        // Stop SparkSession
        spark.stop()
    }
}