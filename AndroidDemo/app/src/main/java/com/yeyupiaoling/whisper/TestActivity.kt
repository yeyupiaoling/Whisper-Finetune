package com.yeyupiaoling.whisper

import android.annotation.SuppressLint
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class TestActivity : AppCompatActivity() {
    private var whisperContext: WhisperContext? = null
    private var resultTextView: TextView? = null
    private var numEdit: EditText? = null
    private var startBtn: Button? = null
    private var samplePath: File? = null

    companion object {
        private val TAG = AudioFileActivity::class.java.name
        private const val wavPath = "samples/test.wav"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)

        samplePath = File(application.filesDir, "samples")
        assets.open(wavPath).use { input ->
            samplePath!!.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        numEdit = findViewById(R.id.num_edit)
        resultTextView = findViewById(R.id.result_text)
        resultTextView!!.movementMethod = ScrollingMovementMethod.getInstance()
        startBtn = findViewById(R.id.start_button)
        startBtn!!.setOnClickListener { _: View? -> start() }
        startBtn!!.isEnabled = false
        lifecycleScope.launch { loadModel() }
    }

    @SuppressLint("SetTextI18n")
    private suspend fun loadModel() = withContext(Dispatchers.IO) {
        val modelFile = ModelManager.getSelectedModelFile(applicationContext)
        if (modelFile == null) {
            withContext(Dispatchers.Main) {
                resultTextView!!.text = "未找到模型，请先返回首页导入或下载模型"
                Toast.makeText(this@TestActivity, "请先配置模型", Toast.LENGTH_SHORT).show()
                finish()
            }
            return@withContext
        }

        val showText = "正在加载模型：${modelFile.absolutePath} ...\n"
        withContext(Dispatchers.Main) { resultTextView!!.text = showText }
        whisperContext = WhisperContext.createContextFromFile(modelFile.absolutePath)
        withContext(Dispatchers.Main) {
            startBtn!!.isEnabled = true
            resultTextView!!.text = showText + "模型加载成功"
            Toast.makeText(this@TestActivity, "模型加载成功", Toast.LENGTH_SHORT).show()
        }
    }

    private fun start() {
        resultTextView!!.text = ""
        startBtn!!.isEnabled = false
        val startTime2 = System.currentTimeMillis()
        val audioData = decodeWaveFile(samplePath!!)
        val endTime1 = System.currentTimeMillis()
        val dataLen = audioData.size / (16000 / 1000)
        var showText = "读取音频时间：${endTime1 - startTime2} ms\n音频时间：${dataLen} ms\n"
        Log.d(TAG, showText)
        val num = numEdit!!.text.toString().toInt()
        var runNum = 0f
        lifecycleScope.launch {
            val startTime = System.currentTimeMillis()
            for (i in 0 until num) {
                val startTime1 = System.currentTimeMillis()
                val text = whisperContext?.transcribeData(audioData)
                withContext(Dispatchers.Main) {
                    showText = "${showText}\n识别结果：${text.toString()}\n" +
                            "识别时间：${System.currentTimeMillis() - startTime1} ms\n"
                    resultTextView!!.text = showText
                    Log.d(TAG, showText)
                    runNum++
                }
            }
            val endTime = System.currentTimeMillis()
            val avgMs = (endTime - startTime).toFloat() / runNum
            val rtf = avgMs / dataLen.toFloat()
            showText = "${showText}\n==================================\n" +
                    "测试次数：${runNum}\n" +
                    "平均识别时间：${avgMs} ms\n" +
                    "实时率（RTF）为：${rtf}"
            resultTextView!!.text = showText
            Log.d(TAG, showText)
            startBtn!!.isEnabled = true
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        lifecycleScope.launch { whisperContext?.release() }
    }
}
