package com.example.digitclassifier

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * 封装 TFLite 数字识别模型。
 * 预处理逻辑与 PC 端 main.py 的 preprocess_for_mnist 完全一致：
 *   中心裁剪 → 灰度 → Otsu 自动阈值二值化 → 抠数字外接框 → 居中成正方 → 缩放 28×28
 *   （黑底白字、数字居中填满，对齐 MNIST 训练分布，解决实时识别拉胯问题）
 */
class DigitClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private val inputSize = 28
    private val numClasses = 10

    fun loadModel(assetName: String = "digit_classifier.tflite") {
        val afd = context.assets.openFd(assetName)
        val fileChannel = FileInputStream(afd.fileDescriptor).channel
        val mapped = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength
        )
        interpreter = Interpreter(mapped)
    }

    /** 输入一张 Bitmap，返回 (预测数字, 置信度) */
    fun classify(bitmap: Bitmap): Pair<Int, Float> {
        val inter = interpreter ?: throw IllegalStateException("模型未加载")

        // 1. 中心正方形裁剪（和 PC 端 roi 裁剪一致）
        val size = minOf(bitmap.width, bitmap.height)
        val x = (bitmap.width - size) / 2
        val y = (bitmap.height - size) / 2
        val cropped = Bitmap.createBitmap(bitmap, x, y, size, size)

        // 2. 取灰度像素（[0,1]）
        val px = IntArray(size * size)
        cropped.getPixels(px, 0, size, 0, 0, size, size)
        val gray = FloatArray(px.size) { i ->
            val p = px[i]
            (0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF)) / 255f
        }

        // 3. Otsu 自动阈值（比固定 mean 稳，铅笔/暗光都行）
        val threshold = otsuThreshold(gray)
        val bin = BooleanArray(gray.size) { gray[it] * 255f > threshold }  // 亮(纸)=true
        // 亮像素占多数=纸底 → 反相，确保最终"字=true 底=false"
        val invert = bin.count { it } > bin.size / 2
        val fg = BooleanArray(gray.size) { if (invert) !bin[it] else bin[it] }  // true=字

        // 4. 抠数字外接框
        var minX = size; var maxX = -1; var minY = size; var maxY = -1
        for (i in fg.indices) {
            if (fg[i]) {
                val cx = i % size; val cy = i / size
                if (cx < minX) minX = cx
                if (cx > maxX) maxX = cx
                if (cy < minY) minY = cy
                if (cy > maxY) maxY = cy
            }
        }
        if (maxX < 0) { minX = 0; maxX = size - 1; minY = 0; maxY = size - 1 }  // 没检测到字则退回整图

        // 5. 裁出字区域 → 加黑边居中成正方 → 缩放到 28×28
        val dw = maxX - minX + 1; val dh = maxY - minY + 1
        val side = maxOf(dw, dh)
        val square = FloatArray(side * side)  // 0=黑底
        for (cy in 0 until dh) for (cx in 0 until dw) {
            if (fg[(minY + cy) * size + (minX + cx)]) {
                val tx = (side - dw) / 2 + cx
                val ty = (side - dh) / 2 + cy
                square[ty * side + tx] = 1f
            }
        }
        val input = ByteBuffer.allocateDirect(4 * inputSize * inputSize)
            .order(ByteOrder.nativeOrder())
        for (ty in 0 until inputSize) for (tx in 0 until inputSize) {
            val sx = (tx * side / inputSize).coerceAtMost(side - 1)
            val sy = (ty * side / inputSize).coerceAtMost(side - 1)
            input.putFloat(square[sy * side + sx])
        }
        input.rewind()

        // 6. 推理
        val output = Array(1) { FloatArray(numClasses) }
        inter.run(input, output)

        val probs = output[0]
        var bestIdx = 0
        var bestVal = probs[0]
        for (i in probs.indices) {
            if (probs[i] > bestVal) {
                bestVal = probs[i]
                bestIdx = i
            }
        }
        return Pair(bestIdx, bestVal)
    }

    /** Otsu 自动阈值：返回 0~255 的灰度阈值，把字和背景自动分开 */
    private fun otsuThreshold(gray: FloatArray): Int {
        val hist = IntArray(256)
        for (v in gray) hist[(v * 255f).toInt().coerceIn(0, 255)]++
        val total = gray.size.toFloat()
        var sum = 0f
        for (i in hist.indices) sum += i * hist[i]
        var sumB = 0f; var wB = 0f; var maxVar = 0f; var threshold = 0
        for (i in hist.indices) {
            wB += hist[i]
            if (wB == 0f) continue
            val wF = total - wB
            if (wF == 0f) break
            sumB += i * hist[i]
            val mB = sumB / wB
            val mF = (sum - sumB) / wF
            val between = wB * wF * (mB - mF) * (mB - mF)
            if (between > maxVar) { maxVar = between; threshold = i }
        }
        return threshold
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
