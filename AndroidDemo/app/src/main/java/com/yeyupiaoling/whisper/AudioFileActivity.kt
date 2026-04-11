package com.yeyupiaoling.whisper

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AudioFileActivity : AppCompatActivity() {
    private var whisperContext: WhisperContext? = null
    private var resultTextView: TextView? = null
    private var selectAudioBtn: Button? = null
    private var progressBar: ProgressBar? = null
    private var progressTextView: TextView? = null

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
        progressBar = findViewById(R.id.progress_bar)
        progressTextView = findViewById(R.id.progress_text)

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
        if (requestCode == 1 && resultCode == RESULT_OK && data?.data != null) {
            transcribeLargeWav(data.data!!)
        }
    }

    @SuppressLint("SetTextI18n")
    private fun transcribeLargeWav(uri: Uri) {
        lifecycleScope.launch {
            selectAudioBtn!!.isEnabled = false
            resultTextView!!.text = ""
            showPreparingUi()
            val startTime = System.currentTimeMillis()

            try {
                val allText = withContext(Dispatchers.IO) {
                    val audioFile = prepareAudioForWhisper(this@AudioFileActivity, uri)
                    try {
                        val wav = readWavInfo(audioFile)

                        val chunkSec = 30
                        val overlapSec = 1
                        val chunkFrames = wav.sampleRate * chunkSec
                        val hopFrames = wav.sampleRate * (chunkSec - overlapSec)

                        val sb = StringBuilder()
                        var startFrame = 0L
                        var index = 0
                        val total = ((wav.totalFrames + hopFrames - 1) / hopFrames).toInt().coerceAtLeast(1)

                        withContext(Dispatchers.Main) { startRecognizingUi(total) }

                        while (startFrame < wav.totalFrames) {
                            val chunk = readWavChunkAs16kMonoFloat(
                                file = audioFile,
                                wav = wav,
                                startFrame = startFrame,
                                framesToRead = chunkFrames
                            )
                            if (chunk.isEmpty()) break

                            val text = whisperContext?.transcribeData(chunk).orEmpty()
                            if (text.isNotBlank()) {
                                sb.append(text.trim()).append('\n')
                            }

                            index++
                            val progress = 15 + (index * 80 / total)
                            withContext(Dispatchers.Main) {
                                updateRecognizingUi(index, total, progress)
                            }

                            startFrame += hopFrames
                        }

                        withContext(Dispatchers.Main) { showFinalizingUi() }
                        sb.toString().trim()
                    } finally {
                        audioFile.delete()
                    }
                }

                val endTime = System.currentTimeMillis()
                val showText = "识别结果：$allText\n识别时间：${endTime - startTime} ms\n"
                resultTextView!!.text = showText
                showCompletedUi()
                Log.d(TAG, showText)
            } catch (e: Exception) {
                e.printStackTrace()
                resultTextView!!.text = "识别失败：${e.message}"
                showErrorUi(e.message ?: "未知错误")
            } finally {
                selectAudioBtn!!.isEnabled = true
            }
        }
    }

    private fun showPreparingUi() {
        progressBar?.visibility = View.VISIBLE
        progressTextView?.visibility = View.VISIBLE
        progressBar?.isIndeterminate = true
        progressTextView?.text = "正在解码/转换音频..."
    }

    private fun startRecognizingUi(total: Int) {
        progressBar?.isIndeterminate = false
        progressBar?.max = 100
        progressBar?.progress = 15
        progressTextView?.text = "正在识别 0/$total 段... (15%)"
    }

    private fun updateRecognizingUi(index: Int, total: Int, progress: Int) {
        progressBar?.isIndeterminate = false
        progressBar?.progress = progress.coerceIn(0, 99)
        progressTextView?.text = "正在识别 $index/$total 段... (${progress.coerceIn(0, 99)}%)"
    }

    private fun showFinalizingUi() {
        progressBar?.isIndeterminate = false
        progressBar?.progress = 100
        progressTextView?.text = "正在整理结果... (100%)"
    }

    private fun showCompletedUi() {
        progressBar?.isIndeterminate = false
        progressBar?.progress = 100
        progressTextView?.text = "识别完成"
    }

    private fun showErrorUi(message: String) {
        progressBar?.isIndeterminate = false
        progressBar?.progress = 0
        progressTextView?.text = "识别失败：$message"
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
