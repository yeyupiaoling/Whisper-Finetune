package com.yeyupiaoling.whisper

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    companion object {
        private const val REQUEST_IMPORT_MODEL = 1001
    }

    private lateinit var modelStatusText: TextView
    private lateinit var importModelBtn: Button
    private lateinit var downloadModelBtn: Button
    private lateinit var startRecordBtn: Button
    private lateinit var startFileBtn: Button
    private lateinit var testModelBtn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelStatusText = findViewById(R.id.model_status_text)
        importModelBtn = findViewById(R.id.import_model_btn)
        downloadModelBtn = findViewById(R.id.download_model_btn)
        startRecordBtn = findViewById(R.id.start_record_activity_btn)
        startFileBtn = findViewById(R.id.start_file_activity_btn)
        testModelBtn = findViewById(R.id.start_test_activity_btn)

        importModelBtn.setOnClickListener { openModelPicker() }
        downloadModelBtn.setOnClickListener { showDownloadDialog() }

        startRecordBtn.setOnClickListener { _: View? ->
            startActivity(Intent(this@MainActivity, RecordActivity::class.java))
        }
        startFileBtn.setOnClickListener { _: View? ->
            startActivity(Intent(this@MainActivity, AudioFileActivity::class.java))
        }
        testModelBtn.setOnClickListener { _: View? ->
            startActivity(Intent(this@MainActivity, TestActivity::class.java))
        }

        initializeModelStatus()
    }

    override fun onResume() {
        super.onResume()
        initializeModelStatus()
    }

    private fun openModelPicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        intent.type = "*/*"
        startActivityForResult(intent, REQUEST_IMPORT_MODEL)
    }

    private fun showDownloadDialog() {
        val models = ModelManager.PRESET_MODELS
        val items = models.map { "${it.displayName} (${it.sizeHint})" }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("选择要下载的模型")
            .setItems(items) { _, which ->
                downloadModel(models[which])
            }
            .show()
    }

    private fun downloadModel(model: ModelManager.PresetModel) {
        setActionEnabled(false)
        modelStatusText.text = "正在下载模型：${model.displayName}..."

        lifecycleScope.launch {
            try {
                val file = withContext(Dispatchers.IO) {
                    ModelManager.downloadPresetModel(this@MainActivity, model) { downloaded, total ->
                        runOnUiThread {
                            if (total > 0) {
                                val progress = (downloaded * 100 / total).toInt()
                                modelStatusText.text = "正在下载模型：${model.displayName}... ${progress}%"
                            } else {
                                modelStatusText.text = "正在下载模型：${model.displayName}... ${downloaded / 1024 / 1024}MB"
                            }
                        }
                    }
                }
                Toast.makeText(this@MainActivity, "下载完成：${file.name}", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "下载失败：${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                refreshModelStatus()
                setActionEnabled(true)
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_IMPORT_MODEL || resultCode != RESULT_OK) return
        val uri = data?.data ?: return

        setActionEnabled(false)
        modelStatusText.text = "正在导入模型..."
        lifecycleScope.launch {
            try {
                val file = ModelManager.importModelFromUri(this@MainActivity, uri)
                Toast.makeText(this@MainActivity, "导入完成：${file.name}", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, "导入失败：${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                refreshModelStatus()
                setActionEnabled(true)
            }
        }
    }

    private fun initializeModelStatus() {
        lifecycleScope.launch {
            // 仅在当前没有可用模型时，尝试自动准备 assets 中的默认模型。
            if (ModelManager.getSelectedModelFile(this@MainActivity) == null) {
                setActionEnabled(false)
                modelStatusText.text = "正在检查默认模型..."
            }
            withContext(Dispatchers.IO) {
                ModelManager.ensureDefaultModelSelected(this@MainActivity)
            }
            refreshModelStatus()
            setActionEnabled(true)
        }
    }

    private fun refreshModelStatus() {
        val modelFile = ModelManager.getSelectedModelFile(this)
        val hasModel = modelFile != null

        modelStatusText.text = if (hasModel) {
            "当前模型：${modelFile!!.name}\n${modelFile.absolutePath}"
        } else {
            "当前未配置模型，请先导入本地模型或下载预置模型。"
        }

        startRecordBtn.isEnabled = hasModel
        startFileBtn.isEnabled = hasModel
        testModelBtn.isEnabled = hasModel
    }

    private fun setActionEnabled(enabled: Boolean) {
        importModelBtn.isEnabled = enabled
        downloadModelBtn.isEnabled = enabled
    }
}
