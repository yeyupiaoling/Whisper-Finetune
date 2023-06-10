package com.yeyupiaoling.whisper

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class RecordActivity : AppCompatActivity() {
    private var audioRecord: AudioRecord? = null
    private var mIsRecording = false
    private var minBufferSize = 0
    private var tempFile: File? = null
    private var resultTextView: TextView? = null
    private var mRecordButton: Button? = null
    private var audioView: AudioView? = null
    private var whisperContext: WhisperContext? = null

    companion object {
        private val TAG = AudioFileActivity::class.java.name

        // 采样率
        const val SAMPLE_RATE = 16000

        // 声道数
        const val CHANNEL = AudioFormat.CHANNEL_IN_MONO

        // 返回的音频数据的格式
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        // assets里面的模型路径
        private const val modelPath = "models/ggml-model.bin"
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_record)
        // 请求权限
        if (!hasPermission()) {
            requestPermission()
        }

        minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, AUDIO_FORMAT)
        resultTextView = findViewById(R.id.result_text)
        // 录音情况显示器
        audioView = findViewById(R.id.audioView)
        audioView?.setStyle(
            AudioView.ShowStyle.STYLE_HOLLOW_LUMP, AudioView.ShowStyle.STYLE_NOTHING
        )
        mRecordButton = findViewById(R.id.record_button)
        mRecordButton!!.setOnTouchListener { v: View?, event: MotionEvent ->
            if (event.action == MotionEvent.ACTION_UP) {
                mIsRecording = false
                stopRecording()
                mRecordButton!!.text = "按下录音"
            } else if (event.action == MotionEvent.ACTION_DOWN) {
                mIsRecording = true
                startRecording()
                mRecordButton!!.text = "录音中..."
            }
            true
        }
        mRecordButton!!.isEnabled = false
        // 启动协程
        lifecycleScope.launch {
            loadModel()
        }

    }

    @SuppressLint("SetTextI18n")
    private suspend fun loadModel() = withContext(Dispatchers.IO) {
        val showText = "正在加载模型：$modelPath ...\n"
        // 在 UI 线程中更新 UI
        withContext(Dispatchers.Main) {
            resultTextView!!.text = showText
        }
        whisperContext =
            WhisperContext.createContextFromAsset(application.assets, modelPath)
        // 在 UI 线程中更新 UI
        withContext(Dispatchers.Main) {
            mRecordButton!!.isEnabled = true
            resultTextView!!.text = showText + "模型加载成功"
            Toast.makeText(this@RecordActivity, "模型加载成功", Toast.LENGTH_SHORT).show()
        }
    }

    // 开始录音
    private fun startRecording() {
        try {
            if (ActivityCompat.checkSelfPermission(
                    this, Manifest.permission.RECORD_AUDIO
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                requestPermission()
                return
            }
            tempFile = File.createTempFile("recording", "wav")
            // 创建录音器
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL, AUDIO_FORMAT, minBufferSize
            )
        } catch (e: IllegalStateException) {
            e.printStackTrace()
        }
        // 开启一个线程将录音数据写入文件
        val recordingAudioThread = Thread {
            try {
                writeAudioDataToWavFile(tempFile!!)
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        recordingAudioThread.start()
        // 启动录音器
        audioRecord!!.startRecording()
        audioView!!.visibility = View.VISIBLE
    }

    // 停止录音
    private fun stopRecording() {
        // 停止录音器
        audioRecord!!.stop()
        audioRecord!!.release()
        audioRecord = null
        audioView!!.visibility = View.GONE
        // 开始识别
        try {
            val startTime = System.currentTimeMillis()
            resultTextView!!.text = "正在识别中..."
            val audioData = decodeWaveFile(tempFile!!)
            // 启动协程
            lifecycleScope.launch {
                mRecordButton!!.isEnabled = false
                // 在协程中调用suspend函数
                val text = whisperContext?.transcribeData(audioData)
                val endTime = System.currentTimeMillis()
                // 在 UI 线程中更新 UI
                withContext(Dispatchers.Main) {
                    // 把结果显示在 TextView 中
                    val showText = "识别结果：${text.toString()}\n" +
                            "音频时间：${audioData.size / (16000 / 1000)} ms\n" +
                            "识别时间：${endTime - startTime} ms\n"
                    resultTextView!!.text = showText
                    Log.d(TAG, showText)
                    mRecordButton!!.isEnabled = true
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            mRecordButton!!.isEnabled = true
        }
    }

    // 保存音频
    @Throws(IOException::class)
    private fun writeAudioDataToWavFile(file: File) {
        val fos = FileOutputStream(file)
        val bos = ByteArrayOutputStream()
        val buffer = ByteArray(minBufferSize)
        audioRecord!!.startRecording()
        while (mIsRecording) {
            val readSize = audioRecord!!.read(buffer, 0, minBufferSize)
            if (readSize > 0) {
                bos.write(buffer, 0, readSize)
                audioView!!.post { audioView!!.setWaveData(buffer) }
            }
        }
        val audioData = bos.toByteArray()
        val totalAudioLen = audioData.size.toLong()
        val totalDataLen = totalAudioLen + 36
        writeWAVHeader(fos, totalAudioLen, totalDataLen, SAMPLE_RATE, 1, 16)
        fos.write(audioData)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (audioRecord != null) {
            audioRecord!!.release()
        }
        if (lifecycleScope.isActive) {
            lifecycleScope.cancel()
        }
        lifecycleScope.launch {
            whisperContext!!.release()
        }
    }

    // check had permission
    private fun hasPermission(): Boolean {
        return checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED && checkSelfPermission(
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED
    }

    // request permission
    private fun requestPermission() {
        requestPermissions(
            arrayOf(
                Manifest.permission.RECORD_AUDIO, Manifest.permission.WRITE_EXTERNAL_STORAGE
            ), 1
        )
    }
}