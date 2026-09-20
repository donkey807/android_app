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
 * 预处理逻辑与 PC 端 main.py 的 realtime_inference 完全一致：
 *   中心裁剪 → resize 28×28 → 灰度归一化 → 若背景偏亮则反相（对齐 MNIST 白字黑底）
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

        // 2. resize 到 28×28
        val resized = Bitmap.createScaledBitmap(cropped, inputSize, inputSize, true)

        // 3. 灰度 + 归一化到 [0,1]
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        val grays = FloatArray(pixels.size)
        var sum = 0f
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            val gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
            grays[i] = gray
            sum += gray
        }

        // 4. MNIST 是白字黑底；若平均亮度偏高（白纸黑字场景）则反相
        val invert = (sum / grays.size) > 0.5f

        // 5. 拼成 [1, 28, 28, 1] float32 ByteBuffer（native 字节序）
        val input = ByteBuffer.allocateDirect(4 * inputSize * inputSize)
            .order(ByteOrder.nativeOrder())
        for (v in grays) {
            input.putFloat(if (invert) 1f - v else v)
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

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
