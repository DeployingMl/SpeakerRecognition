package com.example.speakerrecognition


import java.util.*

class RecognizeSpeaker (
    inLabels: List<String>,
    inAverageWindowDurationMs: Long,
    inDetectionThreshold: Float,
    inSuppressionMS: Int,
    inMinimumCount: Int,
    inMinimumTimeBetweenSamplesMS: Long
) {
    private val labels = inLabels
    private val labelsCount = labels.size
    private val minimumTimeBetweenSamplesMs = inMinimumTimeBetweenSamplesMS
    private val averageWindowDurationMs = inAverageWindowDurationMs
    private val minimumCount = inMinimumCount
    private val  detectionThreshold = inDetectionThreshold
    private val suppressionMS = inSuppressionMS
    // Working variables.
    @ExperimentalStdlibApi
    private val previousResults: Deque<Pair<Long, FloatArray>> = ArrayDeque()
    private var previousTopLabel = AUDIENCE_LABEL
    private var previousTopLabelScore = 0.0f
    private var previousTopLabelTime = Long.MIN_VALUE

    class RecognitionResult(val foundSpeaker: String, val score:Float, val isNewSpeaker: Boolean)

    private class ScoreForSorting(val score: Float, val index:Int) : Comparable<ScoreForSorting> {
        override fun compareTo(other: ScoreForSorting): Int {
            return when {
                score > other.score -> {
                    -1
                }
                score < other.score -> {
                    1
                }
                else -> {
                    0
                }
            }
        }
    }
    @kotlin.ExperimentalStdlibApi
    fun processLatestResults(currentResults: FloatArray, currentTimeMS: Long): RecognitionResult {
        if (currentResults.size != labelsCount) {
            throw RuntimeException(
                "The results for recognition should contain "
                        + labelsCount
                        + " elements, but there are "
                        + currentResults.size
            )
        }
        if (!previousResults.isEmpty() && currentTimeMS < previousResults.first.first) {
            throw RuntimeException(
                "You must feed results in increasing time order, but received a timestamp of "
                        + currentTimeMS
                        + " that was earlier than the previous one of "
                        + previousResults.first.first
            )
        }

        var howManyResults = previousResults.size

        // Ignore any results that are coming in too fast
        if (howManyResults> 1) {
            val timeSinceMostRecent = currentTimeMS - previousResults.last.first
            if (timeSinceMostRecent < minimumTimeBetweenSamplesMs){
                return RecognitionResult(previousTopLabel, previousTopLabelScore, false)
            }
        }
        // Add the latest results to the head of the queue
        previousResults.addFirst(Pair(currentTimeMS, currentResults))
        // Prune any earlier results that are too old for the averaging window
        val timeLimit = currentTimeMS - averageWindowDurationMs
        while (previousResults.first.first < timeLimit) {
            previousResults.removeFirst()
        }
        howManyResults = previousResults.size
        // if there are too few results, assume the result will be unreliable and bail
//        val earliestTime = previousResults.first.first
//        val sampleDuration = currentTimeMS - earliestTime

        if (howManyResults < minimumCount) {
            return RecognitionResult(previousTopLabel, 0.0f, false)
        }

        // Calculate the average score across all the results in the window
        val averageScores = FloatArray(labelsCount)
        for (previousResult in previousResults) {
            val scoresTensor = previousResult.second
            var i = 0
            while (i < scoresTensor.size) {
                averageScores[i] += scoresTensor[i] / howManyResults
                ++ i
            }
        }

        // sort the averaged results in descending order
        val sortedAverageScores = arrayOfNulls<ScoreForSorting>(labelsCount)
        for (i in 0 until labelsCount) {
            sortedAverageScores[i] = ScoreForSorting(averageScores[i], i)
        }
        Arrays.sort(sortedAverageScores)
        // See if the latest top score is enough to trigger a detection.
        val currentTopIndex = sortedAverageScores[0]!!.index
        val currentTopLabel = labels[currentTopIndex]
        val currentTopScore = sortedAverageScores[0]!!.score
        // if we have recently had another label trigger, assume one that occurs  too soon
        // afterwards is a bad result
//        Log.d("RESULT", "Pre $previousTopLabel")
        val timeSinceLastTop: Long = if (previousTopLabel == AUDIENCE_LABEL || previousTopLabelTime
            == Long.MIN_VALUE) {
            Long.MAX_VALUE
        }else {
            currentTimeMS - previousTopLabelTime
        }
        val isNewSpeaker: Boolean
        if (currentTopScore > detectionThreshold && timeSinceLastTop > suppressionMS){
            previousTopLabel  = currentTopLabel
            previousTopLabelScore = currentTopScore
            previousTopLabelTime = currentTimeMS
            isNewSpeaker = true
        } else {
            isNewSpeaker = false
        }
        return RecognitionResult(currentTopLabel, currentTopScore, isNewSpeaker)
    }

    companion object {
        const val AUDIENCE_LABEL = "_Audience_Applause"
        const val MINIMUM_TIME_FRACTION: Long = 4
    }
}

