package com.example.digitclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.digitclassifier.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var classifier: DigitClassifier
    private lateinit var cameraExecutor: ExecutorService
    private var modelLoaded = false

    // 运行时申请相机权限
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else Toast.makeText(this, "需要相机权限才能识别", Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        classifier = DigitClassifier(this)
        try {
            classifier.loadModel()
            modelLoaded = true
        } catch (e: Exception) {
            modelLoaded = false
            Toast.makeText(
                this,
                "未找到 digit_classifier.tflite：请先在本机运行 main.py 导出，或把它放进 app/src/main/assets/",
                Toast.LENGTH_LONG
            ).show()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(Size(640, 640))
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                if (modelLoaded) {
                    try {
                        val bitmap = imageProxy.toBitmap()
                        val (digit, conf) = classifier.classify(bitmap)
                        runOnUiThread {
                            binding.resultText.text =
                                "数字 $digit    置信度 ${"%.2f".format(conf)}"
                            binding.resultText.setTextColor(
                                if (conf > 0.7f) Color.GREEN
                                else Color.rgb(255, 165, 0)
                            )
                        }
                    } catch (_: Exception) {
                        // 单帧异常忽略，下一帧继续
                    }
                }
                imageProxy.close()
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier.close()
    }
}
