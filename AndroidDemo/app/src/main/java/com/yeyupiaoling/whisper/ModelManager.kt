package com.yeyupiaoling.whisper

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

object ModelManager {
    private const val DEFAULT_ASSET_MODEL_PATH = "models/ggml-model.bin"
    private const val DEFAULT_MODEL_FILE_NAME = "ggml-model.bin"

    data class PresetModel(
        val key: String,
        val displayName: String,
        val fileName: String,
        val url: String,
        val sizeHint: String,
    )

    val PRESET_MODELS = listOf(
        PresetModel(
            "tiny",
            "tiny",
            "ggml-tiny.bin",
            "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-tiny.bin",
            "~75MB"
        ),
        PresetModel(
            "base",
            "base",
            "ggml-base.bin",
            "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-base.bin",
            "~150MB"
        ),
        PresetModel(
            "small",
            "small",
            "ggml-small.bin",
            "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-small.bin",
            "~500MB"
        ),
        PresetModel(
            "medium",
            "medium",
            "ggml-medium.bin",
            "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-medium.bin",
            "~1.5GB"
        ),
        PresetModel(
            "large-v3",
            "large-v3",
            "ggml-large-v3.bin",
            "https://www.modelscope.cn/models/cjc1887415157/whisper.cpp/resolve/master/ggml-large-v3.bin",
            "~3.1GB"
        ),
    )

    private const val PREF_NAME = "whisper_model_prefs"
    private const val KEY_MODEL_PATH = "selected_model_path"

    fun getSelectedModelFile(context: Context): File? {
        val path = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .getString(KEY_MODEL_PATH, null)
        // 优先使用用户已选择且仍然存在的模型文件。
        if (!path.isNullOrBlank()) {
            val file = File(path)
            if (file.exists() && file.isFile) {
                return file
            }
        }

        // 如果没有已保存模型，则尝试回退到应用私有目录中的默认模型副本。
        val defaultFile = File(context.filesDir, "models/$DEFAULT_MODEL_FILE_NAME")
        return if (defaultFile.exists() && defaultFile.isFile) defaultFile else null
    }

    private fun saveModelPath(context: Context, path: String) {
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_MODEL_PATH, path)
            .apply()
    }

    suspend fun importModelFromUri(context: Context, uri: Uri): File = withContext(Dispatchers.IO) {
        val modelDir = getModelDir(context)
        val fileName = queryName(context, uri) ?: "imported-model.bin"
        val safeName = if (fileName.endsWith(".bin")) fileName else "$fileName.bin"
        val outFile = File(modelDir, safeName)

        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "无法读取所选模型文件" }
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        saveModelPath(context, outFile.absolutePath)
        outFile
    }

    suspend fun ensureDefaultModelSelected(context: Context): File? = withContext(Dispatchers.IO) {
        val selectedFile = getSelectedModelFile(context)
        // 已有可用模型时不覆盖用户当前选择。
        if (selectedFile != null) {
            saveModelPath(context, selectedFile.absolutePath)
            return@withContext selectedFile
        }

        // assets 中不存在默认模型时，保持当前未配置状态。
        if (!hasAssetModel(context, DEFAULT_ASSET_MODEL_PATH)) {
            return@withContext null
        }

        val targetFile = File(getModelDir(context), DEFAULT_MODEL_FILE_NAME)
        // 目标文件不存在时才从 assets 复制，避免重复拷贝大模型。
        if (!targetFile.exists() || !targetFile.isFile) {
            context.assets.open(DEFAULT_ASSET_MODEL_PATH).use { input ->
                FileOutputStream(targetFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        saveModelPath(context, targetFile.absolutePath)
        targetFile
    }

    suspend fun downloadPresetModel(
        context: Context,
        model: PresetModel,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit,
    ): File = withContext(Dispatchers.IO) {
        val modelDir = getModelDir(context)
        val targetFile = File(modelDir, model.fileName)
        val tempFile = File(modelDir, "${model.fileName}.download")

        val connection = (URL(model.url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15000
            readTimeout = 30000
            requestMethod = "GET"
            instanceFollowRedirects = true
        }

        connection.connect()
        if (connection.responseCode !in 200..299) {
            connection.disconnect()
            throw RuntimeException("下载失败，HTTP ${connection.responseCode}")
        }

        val total = connection.contentLengthLong
        var downloaded = 0L

        connection.inputStream.use { input ->
            FileOutputStream(tempFile).use { output ->
                val buffer = ByteArray(8 * 1024)
                while (true) {
                    val len = input.read(buffer)
                    if (len <= 0) break
                    output.write(buffer, 0, len)
                    downloaded += len
                    onProgress(downloaded, total)
                }
                output.flush()
            }
        }

        connection.disconnect()

        if (targetFile.exists()) {
            targetFile.delete()
        }
        tempFile.renameTo(targetFile)
        saveModelPath(context, targetFile.absolutePath)
        targetFile
    }

    private fun getModelDir(context: Context): File {
        val modelDir = File(context.filesDir, "models")
        if (!modelDir.exists()) {
            modelDir.mkdirs()
        }
        return modelDir
    }

    private fun hasAssetModel(context: Context, assetPath: String): Boolean {
        return try {
            context.assets.open(assetPath).close()
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun queryName(context: Context, uri: Uri): String? {
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && cursor.moveToFirst()) {
                return cursor.getString(index)
            }
        }
        return null
    }
}
