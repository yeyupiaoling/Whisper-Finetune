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
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true",
            "~75MB"
        ),
        PresetModel(
            "base",
            "base",
            "ggml-base.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true",
            "~150MB"
        ),
        PresetModel(
            "small",
            "small",
            "ggml-small.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin?download=true",
            "~500MB"
        ),
        PresetModel(
            "medium",
            "medium",
            "ggml-medium.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin?download=true",
            "~1.5GB"
        ),
        PresetModel(
            "large-v3",
            "large-v3",
            "ggml-large-v3.bin",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true",
            "~3.1GB"
        ),
    )

    private const val PREF_NAME = "whisper_model_prefs"
    private const val KEY_MODEL_PATH = "selected_model_path"

    fun getSelectedModelFile(context: Context): File? {
        val path = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .getString(KEY_MODEL_PATH, null)
        if (path.isNullOrBlank()) return null
        val file = File(path)
        return if (file.exists() && file.isFile) file else null
    }

    private fun saveModelPath(context: Context, path: String) {
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_MODEL_PATH, path)
            .apply()
    }

    suspend fun importModelFromUri(context: Context, uri: Uri): File = withContext(Dispatchers.IO) {
        val modelDir = File(context.filesDir, "models")
        if (!modelDir.exists()) modelDir.mkdirs()

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

    suspend fun downloadPresetModel(
        context: Context,
        model: PresetModel,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit,
    ): File = withContext(Dispatchers.IO) {
        val modelDir = File(context.filesDir, "models")
        if (!modelDir.exists()) modelDir.mkdirs()

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
