package com.example.speakerrecognition

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.Process
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.util.concurrent.locks.ReentrantLock

class MainActivity : AppCompatActivity() {

    // initialize variables
    private var recordingBuffer = ShortArray(RECORDING_LENGTH)
    private var recordingOffset = 0
    private var shouldContinueRecording = true
    private var shouldContinueRecognition = true
    private val recordingBufferLock = ReentrantLock()
    private var recordingThread: Thread? = null
    private var recognitionThread: Thread? = null

    private var interpreter: Interpreter? = null
    private val options = Interpreter.Options()
    private lateinit var labels:List<String>

    private var recognizeSpeaker:RecognizeSpeaker? = null
//    private var speakerActivity:SpeakerActivity? = null

    private val handler = Handler()
    private var benjaminTextView: TextView? = null
    private var jenTextView: TextView? = null
    private var juliaTextView: TextView? = null
    private var margaretTextView: TextView? = null
    private var nelsonTextView: TextView? = null
    private var selectedTextView: TextView? = null
//




    private val audioRecord = AudioRecord(MediaRecorder.AudioSource.DEFAULT,
        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        recordingBuffer.size)

    @kotlin.ExperimentalStdlibApi
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        jenTextView = findViewById(R.id.jens_stoltenberg)
        margaretTextView = findViewById(R.id.margaret_tacher)
        benjaminTextView = findViewById(R.id.benjamin_netanyahu)
        juliaTextView = findViewById(R.id.julia_gillard)

        val model = FileUtil.loadMappedFile(this, MODEL_FILENAME)
        // load the model interpreter
        try {
            interpreter = Interpreter(model, options)
        }catch (e: Exception) {
            throw RuntimeException(e)
        }

        // load labels from the disk
        val labels = FileUtil.loadLabels(this, LABEL_FILENAME)

        recognizeSpeaker = RecognizeSpeaker(
            labels, AVERAGE_WINDOW_DURATION_MS, DETECTION_THRESHOLD,
            SUPPRESSION_MS, MINIMUM_COUNT, MINIMUM_TIME_BETWEEN_SAMPLES_MS
        )

        if (allPermissionsGranted()) {
            startRecording()
            startRecognition(interpreter!!, labels)
        } else {
            ActivityCompat.requestPermissions(
                this, PERMISSIONS, REQUEST_CODE
            )
        }
    }

    private fun record() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)

        val audioBuffer = ShortArray(RECORDING_LENGTH)

        audioRecord.startRecording()
        while (shouldContinueRecording) {
            val numberRead = audioRecord.read(audioBuffer, 0, audioBuffer.size)
            val maxLength = recordingBuffer.size
            val newRecordingOffset = recordingOffset + numberRead
            val secondCopyLength = 0.coerceAtLeast(newRecordingOffset - maxLength)
            val firstCopyLength = numberRead - secondCopyLength
            // We store off all the data for the recognition thread to access. The ML
            // thread will copy out of this buffer into its own, while holding the
            // lock, so this should be thread safe.
            recordingBufferLock.lock()
            recordingOffset = try {
                System.arraycopy(
                    audioBuffer, 0, recordingBuffer,
                    recordingOffset, firstCopyLength
                )
                System.arraycopy(
                    audioBuffer, firstCopyLength, recordingBuffer,
                    0, secondCopyLength
                )
                newRecordingOffset % maxLength
            } finally {
                recordingBufferLock.unlock()
            }
        }
        audioRecord.stop()
        audioRecord.release()
    }

    @kotlin.ExperimentalStdlibApi
    private fun recognize(model:Interpreter, labels:List<String>) {
        val inputBuffer = ShortArray(RECORDING_LENGTH)
        val floatInputBuffer = Array(RECORDING_LENGTH) { FloatArray(1) }
        val outputScores = Array(1) {FloatArray(labels.size)}

        while (shouldContinueRecognition) {
            recordingBufferLock.lock()

            try {
                val maxLength = recordingBuffer.size
                val firstCopyLength = maxLength - recordingOffset
                val secondCopyLength = recordingOffset
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer,
                    0, firstCopyLength)
                System.arraycopy(recordingBuffer, 0, inputBuffer,
                    firstCopyLength, secondCopyLength)
            } finally {
                recordingBufferLock.unlock()
            }

            for (i in 0 until RECORDING_LENGTH) {
                floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f
            }

            val inputArray = arrayOf<Any>(floatInputBuffer)
            val outputMap: MutableMap<Int, Any> = HashMap()
            outputMap[0] = outputScores

            model.runForMultipleInputsOutputs(inputArray, outputMap)

            val currentTime = System.currentTimeMillis()
            val result = recognizeSpeaker!!.processLatestResults(
                outputScores[0], currentTime
            )

            runOnUiThread {
                if (!result.foundSpeaker.startsWith("_") && result.isNewSpeaker) {
                    var labelIndex = 0
                    for (i in labels.indices){
                        if (labels[i] == result.foundSpeaker){
                            labelIndex = i
                            Log.d("RESULTS", "INDEX: $i SPEAKER: ${result.foundSpeaker}")
                        }
                    }
                    when(labelIndex){

                        // update the recognized speaker, order is according to how we saved labels
                        // during training
                        1 -> selectedTextView = jenTextView
                        2 -> selectedTextView = margaretTextView
                        3 -> selectedTextView = benjaminTextView
                        4 -> selectedTextView = juliaTextView

                    }
                    if (selectedTextView !=null) {
                        selectedTextView!!.setBackgroundColor(
                            ContextCompat.getColor(this, android.R.color.holo_blue_bright))

                        // delay update of selected view
                        handler.postDelayed(
                            {
                                selectedTextView!!.setBackgroundColor(
                                ContextCompat.getColor(this,
                                    android.R.color.holo_blue_bright))
                            }, 1000)
                    }

                }
            }
        }

    }

    @Synchronized
    fun startRecording() {
        if (recordingThread != null) {
            return
        }
        shouldContinueRecording = true
        recordingThread = Thread(
            Runnable { record() }
        )
        recordingThread!!.start()
    }

    @kotlin.ExperimentalStdlibApi
    @Synchronized
    fun startRecognition(model:Interpreter, labels: List<String>) {
        if (recognitionThread != null) {
            return
        }
        recognitionThread =
            Thread(
                Runnable {
                    recognize(model, labels)
                }
            )
        recognitionThread!!.start()
    }

    override fun onStop() {
        shouldContinueRecording = false
        shouldContinueRecognition = false
        super.onStop()
    }

    override fun onDestroy(){
        try {
            Log.d("MODEL","Closing Interpreter")
            interpreter!!.close()
        }catch (e:java.lang.Exception) {
            throw java.lang.RuntimeException(e)
        }
        super.onDestroy()
    }

    // Process result from permission request dialog and if request has been granted record audio
    // and perform recognition else show permission not granted
    @kotlin.ExperimentalStdlibApi
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>,
                                            grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE && grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (allPermissionsGranted()) {
                startRecording()
                startRecognition(interpreter!!, labels)
            }else {
                Toast.makeText(this, "Permissions not granted", Toast.LENGTH_SHORT)
                    .show()
                finish()
            }
        }
    }

    // check if record audio permission specified in the manifest has been granted
    private fun allPermissionsGranted() = PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        const val REQUEST_CODE = 13
        private val PERMISSIONS = arrayOf(Manifest.permission.RECORD_AUDIO)
        private const val SAMPLE_RATE = 16000
        private const val SAMPLE_DURATION_MS = 2000
        const val RECORDING_LENGTH = (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000)
        const val AVERAGE_WINDOW_DURATION_MS: Long = 1000
        const val DETECTION_THRESHOLD = 0.50f
        const val SUPPRESSION_MS = 1500
        const val MINIMUM_COUNT = 3
        const val MINIMUM_TIME_BETWEEN_SAMPLES_MS:Long = 30

        const val MODEL_FILENAME = "tflite_model.tflite"
        const val LABEL_FILENAME = "labels.txt"
    }
}

