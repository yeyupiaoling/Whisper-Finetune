package com.yeyupiaoling.whisper

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class AudioFileActivity : AppCompatActivity() {
    private var whisperContext: WhisperContext? = null
    private var resultTextView: TextView? = null
    private var selectAudioBtn: Button? = null

    companion object {
        private val TAG = AudioFileActivity::class.java.name

        // assets里面的模型路径
        private const val modelPath = "models/ggml-model.bin"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_audio_file)

        // 请求权限
        if (!hasPermission()) {
            requestPermission()
        }
        resultTextView = findViewById(R.id.result_text)
        selectAudioBtn = findViewById(R.id.select_audio_btn)
        // 打开文件管理器
        selectAudioBtn!!.setOnClickListener { v: View? ->
            val intent =
                Intent(Intent.ACTION_GET_CONTENT, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI)
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
            intent.setDataAndType(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, "audio/*")
            startActivityForResult(intent, 1)
        }
        selectAudioBtn!!.isEnabled = false
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
            selectAudioBtn!!.isEnabled = true
            resultTextView!!.text = showText + "模型加载成功"
            Toast.makeText(this@AudioFileActivity, "模型加载成功", Toast.LENGTH_SHORT).show()
        }
    }


    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (data !== null) {
                val audioFilePath = getPathFromURI(this, data.data!!)
                val file = File(audioFilePath!!)
                // 开始识别
                try {
                    selectAudioBtn!!.isEnabled = false
                    val startTime = System.currentTimeMillis()
                    resultTextView!!.text = "正在识别中..."
                    val audioData = decodeWaveFile(file)
                    // 启动协程
                    lifecycleScope.launch {
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
                            selectAudioBtn!!.isEnabled = true
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    selectAudioBtn!!.isEnabled = true
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (lifecycleScope.isActive) {
            lifecycleScope.cancel()
        }
        lifecycleScope.launch {
            whisperContext!!.release()
        }
    }

    // check had permission
    private fun hasPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            checkSelfPermission(Manifest.permission.READ_MEDIA_AUDIO) == PackageManager.PERMISSION_GRANTED
        } else {
            checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        }
    }

    // request permission
    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestPermissions(arrayOf(Manifest.permission.READ_MEDIA_AUDIO), 1)
        } else {
            requestPermissions(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), 1)
        }
    }
}