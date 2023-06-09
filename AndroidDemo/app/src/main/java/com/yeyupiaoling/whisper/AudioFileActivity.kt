package com.yeyupiaoling.whisper

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class AudioFileActivity : AppCompatActivity() {
    private var whisperContext: WhisperContext? = null

    companion object {
        private val TAG = AudioFileActivity::class.java.name
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_audio_file)

        // 请求权限
        if (!hasPermission()) {
            requestPermission()
        }
        val selectAudioBtn = findViewById<Button>(R.id.select_audio_btn)
        selectAudioBtn.setOnClickListener { v: View? ->
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "audio/*"
            startActivityForResult(intent, 1)
        }
        selectAudioBtn.isEnabled = false
        whisperContext =
            WhisperContext.createContextFromAsset(application.assets, "models/ggml-model.bin")
        selectAudioBtn.isEnabled = true
        Toast.makeText(this@AudioFileActivity, "模型加载成功", Toast.LENGTH_SHORT).show()
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val audioPath: String?
        if (resultCode == RESULT_OK) {
            if (requestCode == 1) {
                if (data == null) {
                    Log.w("onActivityResult", "user photo data is null")
                    return
                }
                val imageUri = data.data
                Log.w("onActivityResult", "imageUri: $imageUri")
                audioPath = getPathFromURI(this@AudioFileActivity, imageUri!!)
                try {
                    val file = File(audioPath!!)
                    val audioData = decodeWaveFile(file)
                    // 启动协程
                    lifecycleScope.launch {
                        // 在协程中调用 fetchData 函数
                        val text = whisperContext?.transcribeData(audioData)
                        // 在 UI 线程中更新 UI
                        withContext(Dispatchers.Main) {
                            // 把结果显示在 TextView 中
                            Log.d(TAG, text.toString())
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
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