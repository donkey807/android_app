# DigitClassifier —— 手机端实时手写数字识别（Android）

基于你 PC 端 `1_for_day/main.py` 训练的 MNIST 模型，用 **TFLite + CameraX** 打包成 Android App，
手机摄像头实时识别 0~9。无需联网、无需网页。

## 目录结构

```
1_for_day/
├── main.py                      # PC 端：训练 + 导出 ONNX(PC实时) + 导出 TFLite(本工程)
└── android_app/                 # ← 本 Android 工程（用 Android Studio 打开这个文件夹）
    ├── build.gradle             # 根：AGP 8.3 / Kotlin 1.9.22
    ├── settings.gradle
    ├── gradle.properties
    ├── gradle/wrapper/...        # Gradle 8.4 配置（Android Studio 首次打开会自动补全 wrapper）
    └── app/
        ├── build.gradle         # 依赖：CameraX 1.3.1 + TFLite 2.16.1
        └── src/main/
            ├── AndroidManifest.xml
            ├── java/com/example/digitclassifier/
            │   ├── MainActivity.kt        # 摄像头 + 实时推理主逻辑
            │   └── DigitClassifier.kt     # TFLite 封装 + 预处理（与 PC 端一致）
            ├── res/...                    # 布局 / 主题 / 字符串
            └── assets/
                └── digit_classifier.tflite  # ← 模型（见下方"准备模型"）
```

## 准备模型（关键一步）

App 需要一个 `digit_classifier.tflite` 放进 `app/src/main/assets/`：

**方式 A（推荐）**：在本机 `1_for_day/` 目录运行
```bash
python main.py
```
`main()` 会在训练完成后自动把 `.tflite` 导出到
`android_app/app/src/main/assets/digit_classifier.tflite`，无需手动拷贝。

**方式 B（手动）**：已有 `.tflite` 就直接复制进来并重命名。

> 没放模型也能编译，但启动会 Toast 提示"未找到模型"。

## 在 Android Studio 里跑起来

1. 装好 **Android Studio**（Hedgehog 或更新），SDK 装 **API 34 (Android 14)** 平台。
2. `File → Open` 选择 `1_for_day/android_app` 文件夹 → 等待 Gradle Sync 完成
   （首次会自动下载 Gradle 8.4 与依赖，需联网）。
3. 手机开启 **开发者选项 → USB 调试**，用数据线连电脑，弹窗选"允许"。
4. 工具栏选设备为你的手机，点 ▶ Run（或 Shift+F10）。
5. App 打开后授权相机，把写有数字的纸举到镜头前 → 底部实时显示「数字 X  置信度 Y.YY」。

## 预处理说明（手机端 = PC 端）

`DigitClassifier.kt` 的预处理和 `main.py` 的 `realtime_inference` 完全一致，保证两边口径相同：

1. 摄像头帧 → 中心正方形裁剪
2. resize 到 28×28
3. 灰度化 + 归一化到 [0,1]
4. **若平均亮度 > 0.5（白纸黑字场景）则反相**，把黑字变白字，对齐 MNIST 白字黑底
5. 拼成 `[1,28,28,1]` float32 → 喂给 TFLite

## 没有 Android Studio？用 GitHub 云端构建 APK（零本地安装）

不想装 ~1.5GB 的 Android Studio，就把本工程推到 GitHub，让它自动编译出 APK：

1. **准备模型**：先在 PC 上跑一次 `python main.py`，让 `app/src/main/assets/digit_classifier.tflite` 存在
   （CI 只负责编译，不会训练；缺模型 APK 也能出，但装上会提示找不到模型）。
2. **建 GitHub 仓库**：新建一个空仓库，把本 `android_app/` 文件夹里的**全部内容**作为仓库根目录上传
   （即 `settings.gradle`、`app/`、`.github/` 等直接在仓库根，不要多套一层 `android_app/`）。
3. **推代码**：`git push` 到 `main` 或 `master` 分支（或在仓库 Actions 页点 **Run workflow** 手动触发）。
4. **等构建**：进仓库 **Actions** 标签，看 `Build Debug APK` 跑完（约 3~5 分钟，首次要下载依赖）。
5. **下 APK**：构建页底部 **Artifacts** → `digit-classifier-debug` 下载 zip，解压得到 `app-debug.apk`。
6. **装手机**：把 apk 传到手机（数据线/网盘/邮件），开启"允许安装未知来源应用"后点击安装。

> 出的是 **debug** 包，无需签名，可直接装。AGP 8.3 + Gradle 8.4 由工作流里的
> `gradle-version: 8.4` 与 `build.gradle` 版本号锁定，机器一致。

## 已知局限

- MNIST 是干净、居中、标准手写体。真实手写的粗细/倾斜/纸张反光会和它有一定差距，
  识别率不如 PC 端测试集（通常 98~99%）那么高。模型训练时已加旋转/平移/缩放增强缓解。
- 想要更准：用手机摄像头按同样预处理采一批你自己的手写数字，替换掉 MNIST 重新训练。

## 可选优化

- 体积再压：在 `main.py` 的 `export_tflite()` 里加
  `converter.optimizations = [tf.lite.Optimize.DEFAULT]`（动态范围量化，几乎无损）。
- 若识别慢（一般不会）：接 TFLite GPU 代理，加 `tensorflow-lite-gpu` 依赖并 `Interpreter.Options().setDelegate(GpuDelegate())`。
