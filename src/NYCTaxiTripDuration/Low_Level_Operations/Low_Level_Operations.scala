import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.math._


/**
 * A class representing a Decision Tree Regressor model.
 * 
 * @param trainRDD RDD of training data, where each row is an array of doubles.
 * @param maxDepth Maximum depth of the tree.
 * @param minInstancesPerNode Minimum number of samples required to split an internal node.
 * @param maxBins Maximum number of bins to use for quantile approximation.
 * @param seed Random seed for sampling.
 */
class DecisionTreeRegressor(
    trainRDD: RDD[Array[Double]],
    maxDepth: Int = 5,
    minInstancesPerNode: Int = 1,
    maxBins: Int = 32,
    seed: Int = 42
) extends Serializable {

    // Separate features and labels and cache them
    private val X: RDD[Array[Double]] = trainRDD.map(row => row.init).cache()
    private val y: RDD[Double] = trainRDD.map(row => row.last).cache()

    // Build the decision tree once during initialization
    val tree: Any = buildTree(X, y, 0)

    /**
     * Recursively builds the decision tree.
     * 
     * @param X RDD of features.
     * @param y RDD of labels.
     * @param depth Current depth of the tree.
     * @return The constructed decision tree.
     */
    private def buildTree(X: RDD[Array[Double]], y: RDD[Double], depth: Int): Any = {
        val nSamples = X.count()
        // For candidate binning, use maxBins but not more than samples available
        val numBins = math.min(maxBins, nSamples.toInt)
        val nFeatures = X.first().length

        println(s"<==== Building tree at depth $depth with $nSamples samples and $nFeatures features ====>")
        
        // Compute target statistics
        val yMean = y.mean()
        val variance = y.map(value => math.pow(value - yMean, 2)).mean()

        // Stopping condition: no samples, reached max depth, or node is pure
        if (nSamples == 0 || depth >= maxDepth || variance == 0) return yMean

        var bestSplit: Option[(Int, Double)] = None
        var bestVariance = Double.PositiveInfinity

        val dataXY: RDD[(Array[Double], Double)] = X.zip(y)

        for (featureIndex <- 0 until nFeatures) {
            println(s"Processing feature $featureIndex")

            // Extract values for the current feature
            val featureValues: RDD[Double] = X.map(row => row(featureIndex))
            // Define quantile levels, e.g., evenly spaced between 0 and 1 (excluding 0 and 1)
            val quantileLevels: Array[Double] = (1 until numBins).map(i => i.toDouble / numBins).toArray

            // Get candidate thresholds using approximate quantiles
            val candidateThresholds: Array[Double] = approxQuantile(featureValues, quantileLevels, 0.1, seed)
            
            // Evaluate each candidate threshold
            for (threshold <- candidateThresholds) {
                // Filter the data into left and right splits using a single combined RDD
                val leftData: RDD[(Array[Double], Double)] = dataXY.filter { case (features, label) =>
                    features(featureIndex) <= threshold
                }
                val rightData: RDD[(Array[Double], Double)] = dataXY.filter { case (features, label) =>
                    features(featureIndex) > threshold
                }

                val leftCount = leftData.count()
                val rightCount = nSamples - leftCount

                if (leftCount >= minInstancesPerNode && rightCount >= minInstancesPerNode) {
                    // Compute means for left and right splits
                    val leftMean = leftData.map(_._2).mean()
                    val rightMean = rightData.map(_._2).mean()

                    // Compute variance for left and right splits
                    val leftVar = if (leftCount > 0) leftData.map { case (_, label) => math.pow(label - leftMean, 2)}.mean() else 0.0
                    val rightVar = if (rightCount > 0) rightData.map { case (_, label) => math.pow(label - rightMean, 2)}.mean() else 0.0

                    // Weighted variance over the node
                    val weightedVariance = (leftVar * leftCount + rightVar * rightCount) / nSamples

                    if (weightedVariance < bestVariance) {
                        bestVariance = weightedVariance
                        bestSplit = Some((featureIndex, threshold))
                    }
                }
            }
        }

        // If no valid split is found, return the mean as a leaf node
        if (bestSplit.isEmpty) return yMean

        println(s"Best split found at feature ${bestSplit.get._1} with threshold ${bestSplit.get._2}")

        val (featureIndex, threshold) = bestSplit.get

        // Now, split the data using the best split found
        val dataXYSplit: RDD[(Array[Double], Double)] = X.zip(y).cache()

        val leftData = dataXYSplit.filter { case (features, label) => features(featureIndex) <= threshold }.cache()
        val rightData = dataXYSplit.filter { case (features, label) => features(featureIndex) > threshold }.cache()

        val leftX = leftData.map(_._1).cache()
        val leftY = leftData.map(_._2).cache()
        val rightX = rightData.map(_._1).cache()
        val rightY = rightData.map(_._2).cache()

        // Recursively build the left and right subtrees
        val leftTree = buildTree(leftX, leftY, depth + 1)
        val rightTree = buildTree(rightX, rightY, depth + 1)

        // Unpersist intermediate RDDs
        dataXYSplit.unpersist()
        leftData.unpersist()
        rightData.unpersist()
        leftX.unpersist()
        leftY.unpersist()
        rightX.unpersist()
        rightY.unpersist()

        Map(
            "featureIndex" -> featureIndex,
            "threshold" -> threshold,
            "left" -> leftTree,
            "right" -> rightTree
        )
    }

    /**
     * Computes approximate quantiles using a sample of the data.
     * 
     * @param rdd RDD of feature values.
     * @param probabilities Array of probabilities for which to compute quantiles.
     * @param sampleFraction Fraction of data to sample for quantile approximation.
     * @param seed Random seed for sampling.
     * @return Array of approximate quantiles.
     */
    def approxQuantile(
        rdd: RDD[Double],
        probabilities: Array[Double],
        sampleFraction: Double = 0.1,
        seed: Int = 42
    ): Array[Double] = {
        val sampledData = rdd.sample(false, sampleFraction, seed).collect()

        if (sampledData.isEmpty) {
            return rdd.collect().sorted
        }

        // Sort the sampled data
        val sortedSample = sampledData.sorted
        val n = sortedSample.length

        // Compute approximate quantiles
        probabilities.map { q =>
            val idx = math.min(n - 1, math.floor(q * n).toInt)
            sortedSample(idx)
        }
    }

    /**
     * Traverses the decision tree to make a prediction.
     * 
     * @param tree The decision tree.
     * @param features The feature vector for which to make a prediction.
     * @return The predicted value.
     */
    private def traverseTree(tree: Any, features: Array[Double]): Double = {
        tree match {
            case node: Map[String, Any] =>
                val featureIndex = node("featureIndex").asInstanceOf[Int]
                val threshold = node("threshold").asInstanceOf[Double]
                if (features(featureIndex) <= threshold)
                    traverseTree(node("left"), features)
                else
                    traverseTree(node("right"), features)
            case leaf: Double => leaf
        }
    }

    /**
     * Predicts the target value for a given feature vector.
     * 
     * @param features The feature vector for which to make a prediction.
     * @return The predicted value.
     */
    def predict(features: Array[Double]): Double = {
        traverseTree(tree, features)
    }
}


object Low_Level_Operations {
    def main(args: Array[String]): Unit = {
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

        // A function to process each line. Feature engineering, conversion to numeric types.
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

        val processedRDD: RDD[Array[Double]] = dataRDD.map(processTripLine)

        // Filter the data based on the conditions
        val filteredRDD: RDD[Array[Double]] = processedRDD.filter(row => row(0) > 0) // passenger_count > 0
                                                        .filter(row => row(7) < 22 * 3600) // trip_duration < 22 hours
                                                        .filter(row => row(3) > 0) // distance > 0
        
        val Array(trainRDD, testRDD) = filteredRDD.randomSplit(Array(0.8, 0.2), seed = 42)

        val model = new DecisionTreeRegressor(trainRDD, maxDepth = 5, minInstancesPerNode = 1)
        println("Decision Tree Regressor model built successfully.")

        val sampleTestCases = testRDD.take(10)
        println("Sample test cases:")
        sampleTestCases.foreach { row =>
            val prediction: Double = model.predict(row.init)
            println(s"Features: ${row.init.mkString(", ")}, Acutal: ${row.last} Prediction: $prediction")
        }

        val labelsAndPredictions = testRDD.map { row =>
            val prediction: Double = model.predict(row.init)
            (row.last, prediction)
        }

        // Mean Absolute Error
        val testMAE = labelsAndPredictions.map { case (v, p) => math.abs(v - p) }.mean()
        println(s"Test Mean Absolute Error (MAE) = $testMAE")

        // Root Mean Squared Error
        val testRMSE = sqrt(labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean())
        println(s"Test Root Mean Squared Error (RMSE) = $testRMSE")

        val yMean = labelsAndPredictions.map(_._1).mean()
        // Residual sum of squares
        val ssRes = labelsAndPredictions.map { case (label, prediction) =>
            math.pow(label - prediction, 2)
        }.sum()

        // Total sum of squares
        val ssTot = labelsAndPredictions.map { case (label, _) =>
            math.pow(label - yMean, 2)
        }.sum()
        val r2 = 1 - ssRes / ssTot
        println(s"Test R^2 = $r2")

        spark.stop()
    }
}