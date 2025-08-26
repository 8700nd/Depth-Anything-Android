package com.ml.shubham0204.depthanything

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DelegateFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.File

enum class DelegateType { CPU, GPU, NNAPI }

class DepthAnything(context: Context, val modelName: String, delegateType: DelegateType) {

    private val tflite: Interpreter
    private val inputDim: Int
    private val outputDim: Int

    init {
        // Load TFLite model
        val model = loadModelFile(context, modelName)
        val options = Interpreter.Options()
        when (delegateType) {
            DelegateType.GPU -> {
                try {
                    options.addDelegate(GpuDelegate())
                } catch (e: Exception) {
                    e.printStackTrace()
                    Log.w("DepthAnything", "GPU delegate not available, falling back to CPU")
                }
            }
            DelegateType.NNAPI -> {
                options.setUseNNAPI(true)
            }
            DelegateType.CPU -> {

            }
        }
        tflite = Interpreter(model, options)

        // Get input and output dimensions from the model itself
        val inputShape = tflite.getInputTensor(0).shape()
        val outputShape = tflite.getOutputTensor(0).shape()

        inputDim = inputShape[1] // Assuming NHWC format: [1, height, width, channels]
        outputDim = outputShape[1] // Assuming NHWC format: [1, height, width, 1]

        Log.d("DepthAnything", "Input dim: $inputDim, Output dim: $outputDim")
        Log.d("DepthAnything", "Input type: ${tflite.getInputTensor(0).dataType()}, Output type: ${tflite.getOutputTensor(0).dataType()}")
    }

    @Throws(Exception::class)
    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = fileDescriptor.createInputStream()
        val tempFile = File.createTempFile("model", ".tflite", context.cacheDir)
        val outputStream = FileOutputStream(tempFile)

        inputStream.copyTo(outputStream)
        inputStream.close()
        outputStream.close()

        val fileChannel = FileInputStream(tempFile).channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size())
    }

    private val rotateTransform = Matrix().apply { postRotate(90f) }

    suspend fun predict(inputImage: Bitmap): Pair<Bitmap, Long> =
        withContext(Dispatchers.Default) {
            // Check the input tensor data type
            val inputDataType = tflite.getInputTensor(0).dataType()

            // Preprocess image based on the model's expected input type
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputDim, inputDim, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f)) // Normalize to [0, 1]
                .build()

            var tensorImage = TensorImage(inputDataType)
            tensorImage.load(inputImage)
            tensorImage = imageProcessor.process(tensorImage)

            // Prepare output
            val outputDataType = tflite.getOutputTensor(0).dataType()
            val outputShape = intArrayOf(1, outputDim, outputDim, 1)
            val outputSize = outputShape[1] * outputShape[2] * when (outputDataType) {
                DataType.FLOAT32 -> 4
                DataType.UINT8 -> 1
                else -> 4
            }

            val outputBuffer = ByteBuffer.allocateDirect(outputSize).apply {
                order(ByteOrder.nativeOrder())
            }

            // Run inference
            val t1 = System.currentTimeMillis()
            tflite.run(tensorImage.buffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - t1

            // Process output
            outputBuffer.rewind()
            val depthMap = processOutput(outputBuffer, outputDataType, outputDim)
            val scaledDepthMap = Bitmap.createScaledBitmap(
                depthMap,
                inputImage.width,
                inputImage.height,
                true
            )

            Pair(scaledDepthMap, inferenceTime)
        }

    private fun processOutput(outputBuffer: ByteBuffer, dataType: DataType, dim: Int): Bitmap {
        return when (dataType) {
            DataType.FLOAT32 -> processFloatOutput(outputBuffer, dim)
            DataType.UINT8 -> processQuantizedOutput(outputBuffer, dim)
            else -> processFloatOutput(outputBuffer, dim) // Default to float processing
        }
    }

    private fun processFloatOutput(outputBuffer: ByteBuffer, dim: Int): Bitmap {
        val floatBuffer = outputBuffer.asFloatBuffer()
        val pixels = FloatArray(dim * dim)
        floatBuffer.get(pixels)

        // Log min/max values for debugging
        val min = pixels.minOrNull() ?: 0f
        val max = pixels.maxOrNull() ?: 1f
        Log.d("DepthAnything", "Float output range: min=$min, max=$max")

        // Create depth map with proper normalization
        val bitmap = Bitmap.createBitmap(dim, dim, Bitmap.Config.ARGB_8888)
        val range = max - min

        for (y in 0 until dim) {
            for (x in 0 until dim) {
                val value = pixels[y * dim + x]
                // Apply inverse mapping (darker = closer, lighter = farther)
                val normalizedValue = if (range > 0) ((value - min) / range * 255).toInt() else 0
                val clampedValue = normalizedValue.coerceIn(0, 255)

                // Create grayscale color
                val color = Color.argb(255, clampedValue, clampedValue, clampedValue)
                bitmap.setPixel(x, y, color)
            }
        }

        return Bitmap.createBitmap(bitmap, 0, 0, dim, dim, rotateTransform, false)
    }

    private fun processQuantizedOutput(outputBuffer: ByteBuffer, dim: Int): Bitmap {
        val pixels = ByteArray(dim * dim)
        outputBuffer.get(pixels)

        // Create depth map
        val bitmap = Bitmap.createBitmap(dim, dim, Bitmap.Config.ARGB_8888)

        for (y in 0 until dim) {
            for (x in 0 until dim) {
                // Convert to unsigned byte
                val value = pixels[y * dim + x].toInt() and 0xFF
                // Apply inverse mapping if needed
                val invertedValue = 255 - value
                val color = Color.argb(255, invertedValue, invertedValue, invertedValue)
                bitmap.setPixel(x, y, color)
            }
        }

        return Bitmap.createBitmap(bitmap, 0, 0, dim, dim, rotateTransform, false)
    }

    // Create a magma colormap for more visually appealing depth maps
    private fun applyMagmaColormap(value: Int): Int {
        // Simplified magma colormap (blue to red)
        val r = (value / 255.0f * 255).toInt()
        val g = (value / 255.0f * 100).toInt()
        val b = (255 - value / 255.0f * 255).toInt()
        return Color.argb(255, r.coerceIn(0, 255), g.coerceIn(0, 255), b.coerceIn(0, 255))
    }

    // Close the interpreter when done
    fun close() {
        tflite.close()
    }
}