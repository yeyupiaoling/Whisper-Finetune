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
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_audio_file)

        if (!hasPermission()) {
            requestPermission()
        }
        resultTextView = findViewById(R.id.result_text)
        selectAudioBtn = findViewById(R.id.select_audio_btn)
        selectAudioBtn!!.setOnClickListener { _: View? ->
            val intent = Intent(Intent.ACTION_GET_CONTENT, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI)
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
            intent.setDataAndType(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, "audio/*")
            startActivityForResult(intent, 1)
        }
        selectAudioBtn!!.isEnabled = false
        lifecycleScope.launch { loadModel() }
    }

    @SuppressLint("SetTextI18n")
    private suspend fun loadModel() = withContext(Dispatchers.IO) {
        val modelFile = ModelManager.getSelectedModelFile(applicationContext)
        if (modelFile == null) {
            withContext(Dispatchers.Main) {
                resultTextView!!.text = "未找到模型，请先返回首页导入或下载模型"
                Toast.makeText(this@AudioFileActivity, "请先配置模型", Toast.LENGTH_SHORT).show()
                finish()
            }
            return@withContext
        }

        val showText = "正在加载模型：${modelFile.absolutePath} ...\n"
        withContext(Dispatchers.Main) { resultTextView!!.text = showText }
        whisperContext = WhisperContext.createContextFromFile(modelFile.absolutePath)
        withContext(Dispatchers.Main) {
            selectAudioBtn!!.isEnabled = true
            resultTextView!!.text = showText + "模型加载成功"
            Toast.makeText(this@AudioFileActivity, "模型加载成功", Toast.LENGTH_SHORT).show()
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && data != null) {
            val audioFilePath = getPathFromURI(this, data.data!!)
            val file = File(audioFilePath!!)
            try {
                selectAudioBtn!!.isEnabled = false
                val startTime = System.currentTimeMillis()
                resultTextView!!.text = "正在识别中..."
                val audioData = decodeWaveFile(file)
                lifecycleScope.launch {
                    val text = whisperContext?.transcribeData(audioData)
                    val endTime = System.currentTimeMillis()
                    withContext(Dispatchers.Main) {
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

    override fun onDestroy() {
        super.onDestroy()
        if (lifecycleScope.isActive) lifecycleScope.cancel()
        lifecycleScope.launch { whisperContext?.release() }
    }

    private fun hasPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            checkSelfPermission(Manifest.permission.READ_MEDIA_AUDIO) == PackageManager.PERMISSION_GRANTED
        } else {
            checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestPermissions(arrayOf(Manifest.permission.READ_MEDIA_AUDIO), 1)
        } else {
            requestPermissions(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), 1)
        }
    }
}
