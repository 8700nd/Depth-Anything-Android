package com.ml.shubham0204.depthanything

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.ClickableText
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.ml.shubham0204.depthanything.ui.theme.DepthAnythingTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import net.engawapg.lib.zoomable.rememberZoomState
import net.engawapg.lib.zoomable.zoomable
import java.io.File
import java.io.IOException

private const val TAG = "MainActivity"

class MainActivity : ComponentActivity() {

    private var depthImageState = mutableStateOf<Bitmap?>(null)
    private var inferenceTimeState = mutableLongStateOf(0)
    private var progressState = mutableStateOf(false)
    private var cameraXState = mutableStateOf(false)
    private var osCameraState = mutableStateOf(false)
    private lateinit var depthAnything: DepthAnything
    private var currentPhotoPath: String = ""
    private var selectedModelState = mutableStateOf("")
    private var useNnapiState = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent { ActivityUI() }
    }

    @Composable
    private fun ActivityUI() {
        var showPermissionDialog by remember { mutableStateOf(false) }
        var permissionDenied by remember { mutableStateOf(false) }

        val cameraPermissionLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            if (isGranted) {
                permissionDenied = false
                showPermissionDialog = false
            } else {
                permissionDenied = true
                showPermissionDialog = true
            }
        }

        LaunchedEffect(Unit) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        DepthAnythingTheme {
            Surface(
                modifier = Modifier.fillMaxSize(),
                color = MaterialTheme.colorScheme.background
            ) {
                val depthImage by remember { depthImageState }
                val showCameraX by remember { cameraXState }
                ProgressDialog()

                if (showPermissionDialog) {
                    PermissionDeniedDialog(
                        onDismiss = { showPermissionDialog = false },
                        onOpenSettings = {
                            val intent =
                                Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                                    data = Uri.fromParts("package", packageName, null)
                                }
                            startActivity(intent)
                        }
                    )
                }

                if (depthImage != null) {
                    DepthImageUI(depthImage = depthImage!!)
                } else if (showCameraX) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                        CameraModeUI()
                    } else {
                        // Show permission dialog and reset camera state
                        cameraXState.value = false
                        showPermissionDialog = true
                    }
                } else {
                    ImageSelectionUI(
                        useNnapi = useNnapiState.value,
                        onNnapiCheckedChange = { useNnapiState.value = it },
                        onPermissionDenied = { showPermissionDialog = true }
                    )
                }
            }
        }
    }

    @Composable
    private fun ImageSelectionUI(
        useNnapi: Boolean,
        onNnapiCheckedChange: (Boolean) -> Unit,
        onPermissionDenied: () -> Unit
    ) {
        val pickMediaLauncher =
            rememberLauncherForActivityResult(
                contract = ActivityResultContracts.PickVisualMedia()
            ) {
                if (it != null) {
                    progressState.value = true
                    val bitmap = getFixedBitmap(it)
                    CoroutineScope(Dispatchers.Default).launch {
                        val (depthMap, inferenceTime) = depthAnything.predict(bitmap)
                        depthImageState.value = colormapInferno(depthMap)
                        inferenceTimeState.longValue = inferenceTime
                        withContext(Dispatchers.Main) { progressState.value = false }
                    }
                }
            }

        val models = remember { listModelsInAssets() }

        LaunchedEffect(models) {
            if (selectedModelState.value.isEmpty() && models.isNotEmpty()) {
                selectedModelState.value = models.first()
            }
        }

        LaunchedEffect(selectedModelState.value, useNnapi) {
            if (selectedModelState.value.isNotEmpty()) {
                withContext(Dispatchers.IO) {
                    Log.d(
                        TAG,
                        "ImageSelectionUI: Create depthAnything selectedModelState = ${selectedModelState.value}, useNnapi = $useNnapi"
                    )
                    depthAnything = DepthAnything(
                        this@MainActivity,
                        selectedModelState.value,
                        useNnapi
                    )
                }
            }
        }

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .padding(horizontal = 8.dp)
                .fillMaxWidth()
        ) {
            Text(
                text = getString(R.string.model_name),
                style = MaterialTheme.typography.displaySmall,
                modifier = Modifier.align(Alignment.CenterHorizontally)
            )
            Text(
                text = getString(R.string.model_description),
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                modifier = Modifier.align(Alignment.CenterHorizontally)
            )

            // Hyperlink-style text
            val annotatedString = buildAnnotatedString {
                pushStringAnnotation(
                    tag = "paper",
                    annotation = getString(R.string.model_paper_url)
                )
                withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.primary)) {
                    append("View Paper")
                }
                pop()
                append("   ")
                pushStringAnnotation(
                    tag = "github",
                    annotation = getString(R.string.model_github_url)
                )
                withStyle(style = SpanStyle(color = MaterialTheme.colorScheme.primary)) {
                    append("GitHub")
                }
                pop()
            }
            ClickableText(
                text = annotatedString,
                style = MaterialTheme.typography.bodyMedium,
                onClick = { offset ->
                    annotatedString
                        .getStringAnnotations(tag = "paper", start = offset, end = offset)
                        .firstOrNull()
                        ?.let {
                            Intent(Intent.ACTION_VIEW, Uri.parse(it.item)).apply {
                                startActivity(this)
                            }
                        }
                    annotatedString
                        .getStringAnnotations(tag = "github", start = offset, end = offset)
                        .firstOrNull()
                        ?.let {
                            Intent(Intent.ACTION_VIEW, Uri.parse(it.item)).apply {
                                startActivity(this)
                            }
                        }
                }
            )

            // Model selection dropdown
            var expanded by remember { mutableStateOf(false) }
            Row(
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Checkbox(
                    checked = useNnapi,
                    onCheckedChange = { onNnapiCheckedChange(it) }
                )
                Text(text = "Enable NNAPI")
            }

            Box {
                OutlinedButton(
                    onClick = { expanded = true },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(selectedModelState.value.ifEmpty { "No Model" })
                }
                DropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    models.forEach { model ->
                        DropdownMenuItem(
                            text = { Text(model) },
                            onClick = {
                                selectedModelState.value = model
                                expanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = {
                if (this@MainActivity.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    osCameraState.value = true
                    dispatchTakePictureIntent()
                } else {
                    onPermissionDenied()
                }
            }) { Text(text = "Take A Picture") }

            Button(
                onClick = {
                    pickMediaLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                }
            ) {
                Text(text = "Select From Gallery")
            }

            Button(onClick = {
                if (this@MainActivity.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    cameraXState.value = true
                } else {
                    onPermissionDenied()
                }
            }) {
                Text(text = "Use Camera for live inference")
            }
        }
    }

    private fun listModelsInAssets(): List<String> {
        return assets.list("")?.filter { it.endsWith(".onnx") } ?: emptyList()
    }

    @Composable
    private fun CameraModeUI() {
        var depthBitmap by remember { mutableStateOf<Bitmap?>(null) }
        var isProcessing by remember { mutableStateOf(false) }

        val fps = if (inferenceTimeState.longValue > 0) {
            1000f / inferenceTimeState.longValue.toFloat()
        } else {
            0f
        }

        Column(modifier = Modifier.fillMaxSize()) {
            // Raw frames view
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .clip(RectangleShape)
            ) {
                CameraPreview(
                    onFrameCaptured = { bitmap ->
                        if (!isProcessing) {
                            isProcessing = true
                            CoroutineScope(Dispatchers.Default).launch {
                                val (depthMap, inferenceTime) = depthAnything.predict(bitmap)
                                val matrix = Matrix().apply {
                                    postRotate(-90f)
                                    postScale(-1f, -1f)
                                }
                                depthBitmap = Bitmap.createBitmap(
                                    colormapInferno(depthMap),
                                    0, 0,
                                    colormapInferno(depthMap).width,
                                    colormapInferno(depthMap).height,
                                    matrix,
                                    true
                                )
                                inferenceTimeState.longValue = inferenceTime
                                withContext(Dispatchers.Main) {
                                    isProcessing = false
                                }
                            }
                        }
                    },
                    isProcessing = isProcessing,
                    modifier = Modifier.fillMaxSize()
                )
            }

            // Depth map view
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .clip(RectangleShape)
            ) {
                if (depthBitmap != null) {
                    Image(
                        modifier = Modifier.fillMaxSize(),
                        bitmap = depthBitmap!!.asImageBitmap(),
                        contentDescription = "Depth Map",
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("Processing depth...")
                    }
                }
            }

            // Controls row
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Button(onClick = { cameraXState.value = false }) {
                    Text("Back")
                }

                Text(
                    text = "FPS: ${"%.2f".format(fps)}",
                    modifier = Modifier.align(Alignment.CenterVertically)
                )
                Text(
                    modifier = Modifier.align(Alignment.CenterVertically),
                    text = "Inference time: ${inferenceTimeState.longValue} ms"
                )
            }
            Text(
                modifier = Modifier.padding(start = 16.dp),
                text = "Model: ${depthAnything.modelName}"
            )
        }
    }

    @Composable
    private fun DepthImageUI(depthImage: Bitmap) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row {
                Text(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(2f),
                    text = "Depth Image",
                    style = MaterialTheme.typography.headlineSmall
                )
                Button(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    onClick = { depthImageState.value = null }
                ) {
                    Text(text = "Close")
                }
            }
            Image(
                modifier =
                    Modifier
                        .aspectRatio(depthImage.width.toFloat() / depthImage.height.toFloat())
                        .zoomable(rememberZoomState()),
                bitmap = depthImage.asImageBitmap(),
                contentDescription = "Depth Image"
            )
            Text(text = "Inference time: ${inferenceTimeState.longValue} ms")
            Text(text = "Model used: ${depthAnything.modelName}")
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun PermissionDeniedDialog(
        onDismiss: () -> Unit,
        onOpenSettings: () -> Unit
    ) {
        AlertDialog(
            onDismissRequest = onDismiss,
            title = { Text("Camera Permission Required") },
            text = { Text("Camera permission is required to use the camera feature. Please grant permission in app settings.") },
            confirmButton = {
                Button(
                    onClick = onOpenSettings
                ) {
                    Text("Open Settings")
                }
            },
            dismissButton = {
                Button(
                    onClick = onDismiss
                ) {
                    Text("Dismiss")
                }
            }
        )
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    private fun ProgressDialog() {
        val isShowingProgress by remember { progressState }
        if (isShowingProgress) {
            BasicAlertDialog(onDismissRequest = { /* ProgressDialog is not cancellable */ }) {
                Surface(color = androidx.compose.ui.graphics.Color.White) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        CircularProgressIndicator()
                        Text(text = "Processing image ...")
                    }
                }
            }
        }
    }

    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, false)
    }

    private fun getFixedBitmap(imageFileUri: Uri): Bitmap {
        var imageBitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(imageFileUri))
        val exifInterface = ExifInterface(contentResolver.openInputStream(imageFileUri)!!)
        imageBitmap =
            when (
                exifInterface.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_UNDEFINED
                )
            ) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(imageBitmap, 90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(imageBitmap, 180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(imageBitmap, 270f)
                else -> imageBitmap
            }
        return imageBitmap
    }

    // Dispatch an Intent which opens the camera application for the user.
    // The code is from -> https://developer.android.com/training/camera/photobasics#TaskPath
    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (takePictureIntent.resolveActivity(packageManager) != null) {
            val photoFile: File? =
                try {
                    val imagesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
                    File.createTempFile("image", ".jpg", imagesDir).apply {
                        currentPhotoPath = absolutePath
                    }
                } catch (ex: IOException) {
                    null
                }
            photoFile?.also {
                val photoURI =
                    FileProvider.getUriForFile(this, "com.ml.shubham0204.depthanything", it)
                takePictureLauncher.launch(photoURI)
            }
        }
    }

    private val takePictureLauncher =
        registerForActivityResult(ActivityResultContracts.TakePicture()) {
            if (it) {
                var bitmap = BitmapFactory.decodeFile(currentPhotoPath)
                val exifInterface = ExifInterface(currentPhotoPath)
                bitmap =
                    when (
                        exifInterface.getAttributeInt(
                            ExifInterface.TAG_ORIENTATION,
                            ExifInterface.ORIENTATION_UNDEFINED
                        )
                    ) {
                        ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(bitmap, 90f)
                        ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(bitmap, 180f)
                        ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(bitmap, 270f)
                        else -> bitmap
                    }
                progressState.value = true
                CoroutineScope(Dispatchers.Default).launch {
                    val (depthMap, inferenceTime) = depthAnything.predict(bitmap)
                    depthImageState.value = colormapInferno(depthMap)
                    inferenceTimeState.longValue = inferenceTime
                    withContext(Dispatchers.Main) { progressState.value = false }
                }
            }
        }
}
