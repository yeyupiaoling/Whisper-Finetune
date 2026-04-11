package com.yeyupiaoling.whisper

import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.RandomAccessFile
import kotlin.math.floor
import kotlin.math.min

data class WavInfo(
    val sampleRate: Int,
    val channels: Int,
    val bitsPerSample: Int,
    val dataStart: Long,
    val dataSize: Long
) {
    val bytesPerSample: Int get() = bitsPerSample / 8
    val bytesPerFrame: Int get() = bytesPerSample * channels
    val totalFrames: Long get() = dataSize / bytesPerFrame
}

fun copyUriToTempFile(context: Context, uri: Uri): File {
    val outFile = File(context.cacheDir, "audio_${System.currentTimeMillis()}.tmp")
    context.contentResolver.openInputStream(uri).use { input ->
        requireNotNull(input) { "Cannot open input stream for uri: $uri" }
        FileOutputStream(outFile).use { out ->
            val buffer = ByteArray(16 * 1024)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                out.write(buffer, 0, n)
            }
        }
    }
    return outFile
}

fun readWavInfo(file: File): WavInfo {
    RandomAccessFile(file, "r").use { raf ->
        val riff = ByteArray(4)
        raf.readFully(riff)
        require(String(riff, Charsets.US_ASCII) == "RIFF") { "Not RIFF file" }

        raf.skipBytes(4) // file size

        val wave = ByteArray(4)
        raf.readFully(wave)
        require(String(wave, Charsets.US_ASCII) == "WAVE") { "Not WAVE file" }

        var sampleRate = 0
        var channels = 0
        var bitsPerSample = 0
        var dataStart = -1L
        var dataSize = -1L

        while (raf.filePointer < raf.length()) {
            val chunkIdBytes = ByteArray(4)
            raf.readFully(chunkIdBytes)
            val chunkId = String(chunkIdBytes, Charsets.US_ASCII)
            val chunkSize = readIntLE(raf).toLong() and 0xffffffffL

            when (chunkId) {
                "fmt " -> {
                    val audioFormat = readShortLE(raf).toInt() and 0xffff
                    channels = readShortLE(raf).toInt() and 0xffff
                    sampleRate = readIntLE(raf)
                    raf.skipBytes(6) // byteRate + blockAlign
                    bitsPerSample = readShortLE(raf).toInt() and 0xffff

                    require(audioFormat == 1) { "Only PCM WAV supported (audioFormat=$audioFormat)" }

                    val remain = chunkSize - 16
                    if (remain > 0) raf.skipBytes(remain.toInt())
                }

                "data" -> {
                    dataStart = raf.filePointer
                    dataSize = chunkSize
                    raf.seek(raf.filePointer + chunkSize)
                }

                else -> {
                    raf.seek(raf.filePointer + chunkSize)
                }
            }

            // word alignment
            if (chunkSize % 2L == 1L) raf.skipBytes(1)
        }

        require(sampleRate > 0 && channels > 0 && bitsPerSample == 16 && dataStart >= 0 && dataSize > 0) {
            "Invalid WAV or unsupported format. Need PCM16 WAV."
        }

        return WavInfo(sampleRate, channels, bitsPerSample, dataStart, dataSize)
    }
}

fun readWavChunkAs16kMonoFloat(
    file: File,
    wav: WavInfo,
    startFrame: Long,
    framesToRead: Int
): FloatArray {
    val safeFrames = min(framesToRead.toLong(), wav.totalFrames - startFrame).toInt()
    if (safeFrames <= 0) return floatArrayOf()

    val bytesToRead = safeFrames * wav.bytesPerFrame
    val raw = ByteArray(bytesToRead)

    RandomAccessFile(file, "r").use { raf ->
        raf.seek(wav.dataStart + startFrame * wav.bytesPerFrame)
        raf.readFully(raw)
    }

    // PCM16 -> mono float
    val mono = FloatArray(safeFrames)
    var p = 0
    for (i in 0 until safeFrames) {
        var sum = 0f
        for (ch in 0 until wav.channels) {
            val lo = raw[p].toInt() and 0xff
            val hi = raw[p + 1].toInt()
            val s = (hi shl 8) or lo
            val sample = s.toShort() / 32768.0f
            sum += sample
            p += 2
        }
        mono[i] = sum / wav.channels
    }

    return if (wav.sampleRate == 16000) mono else resampleLinear(mono, wav.sampleRate, 16000)
}

private fun resampleLinear(input: FloatArray, inRate: Int, outRate: Int): FloatArray {
    if (input.isEmpty()) return input
    val outLen = floor(input.size.toDouble() * outRate / inRate).toInt().coerceAtLeast(1)
    val out = FloatArray(outLen)
    val scale = inRate.toDouble() / outRate
    for (i in 0 until outLen) {
        val srcPos = i * scale
        val i0 = srcPos.toInt().coerceIn(0, input.lastIndex)
        val i1 = (i0 + 1).coerceIn(0, input.lastIndex)
        val t = srcPos - i0
        out[i] = ((1 - t) * input[i0] + t * input[i1]).toFloat()
    }
    return out
}

private fun readIntLE(raf: RandomAccessFile): Int {
    val b0 = raf.read()
    val b1 = raf.read()
    val b2 = raf.read()
    val b3 = raf.read()
    return (b0 and 0xff) or ((b1 and 0xff) shl 8) or ((b2 and 0xff) shl 16) or ((b3 and 0xff) shl 24)
}

private fun readShortLE(raf: RandomAccessFile): Short {
    val b0 = raf.read()
    val b1 = raf.read()
    return (((b1 and 0xff) shl 8) or (b0 and 0xff)).toShort()
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
        val file = File(context.filesDir, name)
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
        return file.path
    } catch (e: Exception) {
        e.printStackTrace()
    }
    return null
}
