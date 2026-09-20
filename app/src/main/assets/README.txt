把训练好的 TFLite 模型放到这个目录，命名为：

    digit_classifier.tflite

获取方式（二选一）：
  1. 在本机运行一次：python main.py
     —— main.py 会自动把导出的 .tflite 写到这里（android_app/app/src/main/assets/）。
  2. 手动复制：把 PC 端 saved_models 或导出的 digit_classifier.tflite 复制进来并重命名。

没有这个文件，App 启动会提示"未找到模型"，但工程本身可正常编译。
