package com.yeyupiaoling.whisper

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Point
import android.util.AttributeSet
import android.view.View

class AudioView : View {
    private var upShowStyle = ShowStyle.STYLE_HOLLOW_LUMP
    private var downShowStyle = ShowStyle.STYLE_WAVE
    private var waveData: ByteArray? = null
    var pointList: MutableList<Point>? = null
    private var lumpPaint: Paint? = null
    var wavePath = Path()

    constructor(context: Context?) : super(context) {
        init()
    }

    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs) {
        init()
    }

    constructor(context: Context?, attrs: AttributeSet?, defStyleAttr: Int) : super(
        context,
        attrs,
        defStyleAttr
    ) {
        init()
    }

    private fun init() {
        lumpPaint = Paint()
        lumpPaint!!.isAntiAlias = true
        lumpPaint!!.color = LUMP_COLOR
        lumpPaint!!.strokeWidth = 2f
        lumpPaint!!.style = Paint.Style.STROKE
    }

    fun setWaveData(data: ByteArray) {
        waveData = readyData(data)
        genSamplingPoint(data)
        invalidate()
    }

    fun setStyle(upShowStyle: ShowStyle, downShowStyle: ShowStyle) {
        this.upShowStyle = upShowStyle
        this.downShowStyle = downShowStyle
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        wavePath.reset()
        for (i in 0 until LUMP_COUNT) {
            if (waveData == null) {
                canvas.drawRect(
                    ((LUMP_WIDTH + LUMP_SPACE) * i).toFloat(),
                    (
                            LUMP_MAX_HEIGHT - LUMP_MIN_HEIGHT).toFloat(),
                    (
                            (LUMP_WIDTH + LUMP_SPACE) * i + LUMP_WIDTH).toFloat(),
                    LUMP_MAX_HEIGHT.toFloat(),
                    lumpPaint!!
                )
                continue
            }
            when (upShowStyle) {
                ShowStyle.STYLE_HOLLOW_LUMP -> drawLump(canvas, i, false)
                ShowStyle.STYLE_WAVE -> drawWave(canvas, i, false)
                else -> {}
            }
            when (downShowStyle) {
                ShowStyle.STYLE_HOLLOW_LUMP -> drawLump(canvas, i, true)
                ShowStyle.STYLE_WAVE -> drawWave(canvas, i, true)
                else -> {}
            }
        }
    }

    /**
     * 绘制曲线
     *
     * @param canvas
     * @param i
     * @param reversal
     */
    private fun drawWave(canvas: Canvas, i: Int, reversal: Boolean) {
        if (pointList == null || pointList!!.size < 2) {
            return
        }
        val ratio = SCALE * if (reversal) -1 else 1
        if (i < pointList!!.size - 2) {
            val point = pointList!![i]
            val nextPoint = pointList!![i + 1]
            val midX = point.x + nextPoint.x shr 1
            if (i == 0) {
                wavePath.moveTo(point.x.toFloat(), LUMP_MAX_HEIGHT - point.y * ratio)
            }
            wavePath.cubicTo(
                midX.toFloat(), LUMP_MAX_HEIGHT - point.y * ratio,
                midX.toFloat(), LUMP_MAX_HEIGHT - nextPoint.y * ratio,
                nextPoint.x.toFloat(), LUMP_MAX_HEIGHT - nextPoint.y * ratio
            )
            canvas.drawPath(wavePath, lumpPaint!!)
        }
    }

    /**
     * 绘制矩形条
     */
    private fun drawLump(canvas: Canvas, i: Int, reversal: Boolean) {
        val minus = if (reversal) -1 else 1
        val top = LUMP_MAX_HEIGHT - (LUMP_MIN_HEIGHT + waveData!![i] * SCALE) * minus
        canvas.drawRect(
            (LUMP_SIZE * i).toFloat(),
            top,
            (
                    LUMP_SIZE * i + LUMP_WIDTH).toFloat(),
            LUMP_MAX_HEIGHT.toFloat(),
            lumpPaint!!
        )
    }

    /**
     * 生成波形图的采样数据，减少计算量
     *
     * @param data
     */
    private fun genSamplingPoint(data: ByteArray) {
        if (upShowStyle != ShowStyle.STYLE_WAVE && downShowStyle != ShowStyle.STYLE_WAVE) {
            return
        }
        if (pointList == null) {
            pointList = ArrayList()
        } else {
            pointList!!.clear()
        }
        pointList!!.add(Point(0, 0))
        var i = WAVE_SAMPLING_INTERVAL
        while (i < LUMP_COUNT) {
            pointList!!.add(Point(LUMP_SIZE * i, waveData!![i].toInt()))
            i += WAVE_SAMPLING_INTERVAL
        }
        pointList!!.add(Point(LUMP_SIZE * LUMP_COUNT, 0))
    }

    /**
     * 可视化样式
     */
    enum class ShowStyle {
        /**
         * 空心的矩形小块
         */
        STYLE_HOLLOW_LUMP,

        /**
         * 曲线
         */
        STYLE_WAVE,

        /**
         * 不显示
         */
        STYLE_NOTHING
    }

    companion object {
        // 频谱数量
        private const val LUMP_COUNT = 128
        private const val LUMP_WIDTH = 6
        private const val LUMP_SPACE = 2
        private const val LUMP_MIN_HEIGHT = LUMP_WIDTH
        private const val LUMP_MAX_HEIGHT = 200 //TODO: HEIGHT
        private const val LUMP_SIZE = LUMP_WIDTH + LUMP_SPACE
        private val LUMP_COLOR = Color.parseColor("#6de8fd")
        private const val WAVE_SAMPLING_INTERVAL = 3
        private const val SCALE = (LUMP_MAX_HEIGHT / LUMP_COUNT).toFloat()

        /**
         * 预处理数据
         *
         * @return
         */
        private fun readyData(fft: ByteArray): ByteArray {
            val newData = ByteArray(LUMP_COUNT)
            var abs: Byte
            for (i in 0 until LUMP_COUNT) {
                abs = Math.abs(fft[i].toInt()).toByte()
                //描述：Math.abs -128时越界
                newData[i] = if (abs < 0) 127 else abs
            }
            return newData
        }
    }
}