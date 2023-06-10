package com.yeyupiaoling.whisper

import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

// 读取音频数据用于输入预测
fun decodeWaveFile(file: File): FloatArray {
    val baos = ByteArrayOutputStream()
    file.inputStream().use { it.copyTo(baos) }
    val buffer = ByteBuffer.wrap(baos.toByteArray())
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    val channel = buffer.getShort(22).toInt()
    buffer.position(44)
    val shortBuffer = buffer.asShortBuffer()
    val shortArray = ShortArray(shortBuffer.limit())
    shortBuffer.get(shortArray)
    return FloatArray(shortArray.size / channel) { index ->
        when (channel) {
            1 -> (shortArray[index] / 32767.0f).coerceIn(-1f..1f)
            else -> ((shortArray[2 * index] + shortArray[2 * index + 1]) / 32767.0f / 2.0f).coerceIn(
                -1f..1f
            )
        }
    }
}


// 给录音的流加上文件头
@Throws(IOException::class)
fun writeWAVHeader(
    fos: FileOutputStream, totalAudioLen: Long, totalDataLen: Long,
    sampleRate: Int, channels: Int, bitRate: Int
) {
    val byteRate = bitRate.toLong() * channels * sampleRate / 8
    val header = ByteArray(44)
    header[0] = 'R'.code.toByte() // RIFF/WAVE header
    header[1] = 'I'.code.toByte()
    header[2] = 'F'.code.toByte()
    header[3] = 'F'.code.toByte()
    header[4] = (totalDataLen and 0xffL).toByte()
    header[5] = (totalDataLen shr 8 and 0xffL).toByte()
    header[6] = (totalDataLen shr 16 and 0xffL).toByte()
    header[7] = (totalDataLen shr 24 and 0xffL).toByte()
    header[8] = 'W'.code.toByte()
    header[9] = 'A'.code.toByte()
    header[10] = 'V'.code.toByte()
    header[11] = 'E'.code.toByte()
    header[12] = 'f'.code.toByte() // 'fmt ' chunk
    header[13] = 'm'.code.toByte()
    header[14] = 't'.code.toByte()
    header[15] = ' '.code.toByte()
    header[16] = 16 // 4 bytes: size of 'fmt ' chunk
    header[17] = 0
    header[18] = 0
    header[19] = 0
    header[20] = 1 // format = 1
    header[21] = 0
    header[22] = channels.toByte()
    header[23] = 0
    header[24] = (sampleRate and 0xff).toByte()
    header[25] = (sampleRate shr 8 and 0xff).toByte()
    header[26] = (sampleRate shr 16 and 0xff).toByte()
    header[27] = (sampleRate shr 24 and 0xff).toByte()
    header[28] = (byteRate and 0xffL).toByte()
    header[29] = (byteRate shr 8 and 0xffL).toByte()
    header[30] = (byteRate shr 16 and 0xffL).toByte()
    header[31] = (byteRate shr 24 and 0xffL).toByte()
    header[32] = (channels * bitRate / 8).toByte()
    header[33] = 0
    header[34] = bitRate.toByte()
    header[35] = 0
    header[36] = 'd'.code.toByte()
    header[37] = 'a'.code.toByte()
    header[38] = 't'.code.toByte()
    header[39] = 'a'.code.toByte()
    header[40] = (totalAudioLen and 0xffL).toByte()
    header[41] = (totalAudioLen shr 8 and 0xffL).toByte()
    header[42] = (totalAudioLen shr 16 and 0xffL).toByte()
    header[43] = (totalAudioLen shr 24 and 0xffL).toByte()
    fos.write(header, 0, 44)
}


// 根据返回的URI转换为路径
fun getPathFromURI(context: Context, uri: Uri): String? {
    try {
        val returnCursor: Cursor? =
            context.contentResolver.query(uri, null, null, null, null)
        val nameIndex: Int = returnCursor!!.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        returnCursor.moveToFirst()
        val name: String = returnCursor.getString(nameIndex)
        val file = File(context.getFilesDir(), name)
        val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
        val outputStream = FileOutputStream(file)
        var read: Int
        val maxBufferSize = 1 * 1024 * 1024
        val bytesAvailable: Int = inputStream!!.available()
        val bufferSize = Math.min(bytesAvailable, maxBufferSize)
        val buffers = ByteArray(bufferSize)
        while (inputStream.read(buffers).also { read = it } != -1) {
            outputStream.write(buffers, 0, read)
        }
        returnCursor.close()
        inputStream.close()
        outputStream.close()
        return file.getPath()
    } catch (e: Exception) {
        e.printStackTrace()
    }
    return null
}
