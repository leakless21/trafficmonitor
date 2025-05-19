# Fully Manual YOLO Inference: From Scratch with ONNX and TensorRT

This document provides a comprehensive guide to performing inference with YOLO models in ONNX (`.onnx`) or TensorRT (`.engine`) formats **without using any Ultralytics library code, including utilities.** This approach requires you to implement all preprocessing and postprocessing steps manually.

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Prerequisites](#2-prerequisites)
3.  [Loading the Model](#3-loading-the-model)
    *   [3.1 Using ONNX Runtime for `.onnx` files](#31-using-onnx-runtime-for-onnx-files)
    *   [3.2 Using TensorRT Python API for `.engine` files](#32-using-tensorrt-python-api-for-engine-files)
    *   [3.3 Using PyTorch for `.pt` files](#33-using-pytorch-for-pt-files)
    *   [3.4 Using OpenVINO™ for IR format (.xml + .bin)](#34-using-openvino-for-ir-format-xml--bin)
    *   [3.5 Using PaddlePaddle for `.pdmodel` + `.pdiparams`](#35-using-paddlepaddle-for-pdmodel--pdiparams)
4.  [Manual Image Preprocessing](#4-manual-image-preprocessing)
    *   [4.1 Image Loading](#41-image-loading)
    *   [4.2 Letterboxing (Resize and Pad)](#42-letterboxing-resize-and-pad)
    *   [4.3 Color Conversion, Normalization, and Tensor Preparation](#43-color-conversion-normalization-and-tensor-preparation)
5.  [Performing Inference](#5-performing-inference)
    *   [5.1 ONNX Runtime Inference](#51-onnx-runtime-inference)
    *   [5.2 TensorRT Inference](#52-tensorrt-inference)
    *   [5.3 PyTorch Inference](#53-pytorch-inference)
    *   [5.4 OpenVINO™ Inference](#54-openvino-inference)
    *   [5.5 PaddlePaddle Inference](#55-paddlepaddle-inference)
6.  [Manual Postprocessing (Detection Example)](#6-manual-postprocessing-detection-example)
    *   [6.1 Understanding Raw Model Output](#61-understanding-raw-model-output)
    *   [6.2 Non-Maximum Suppression (NMS) from Scratch](#62-non-maximum-suppression-nms-from-scratch)
    *   [6.3 Scaling Coordinates to Original Image Size](#63-scaling-coordinates-to-original-image-size)
    *   [6.4 Processing Other Tasks (Segmentation, Pose)](#64-processing-other-tasks-segmentation-pose)
7.  [Putting It All Together (Conceptual Script)](#7-putting-it-all-together-conceptual-script)
8.  [Important Considerations and Limitations](#8-important-considerations-and-limitations)

## 1. Introduction
This guide is for advanced users who need full control over the inference pipeline or want to integrate YOLO models into environments where the Ultralytics library is not available or desired. We will focus on object detection as the primary example for postprocessing.

## 2. Prerequisites

*   **Python:** Version 3.8+ is recommended.
*   **Core Libraries:**
    *   `numpy`: For numerical operations, especially array manipulation.
    *   `opencv-python` (cv2): For image loading and basic image manipulations (if not handled by another specific library section).
*   **Framework-Specific Libraries:**
    *   **ONNX Runtime:** `onnxruntime` (or `onnxruntime-gpu` for CUDA/DirectML execution). Install with `pip install onnxruntime` or `pip install onnxruntime-gpu`.
    *   **TensorRT:** The `tensorrt` Python package (usually installed as part of the TensorRT SDK). You will also likely need `pycuda` (`pip install pycuda`) for direct CUDA memory management if you go deep into manual buffer handling.
    *   **PyTorch:** `torch` and `torchvision` (if using torchvision models for architecture definition). Install with `pip install torch torchvision` (see PyTorch official website for CUDA-specific versions if needed).
    *   **OpenVINO™:** `openvino` package. Install with `pip install openvino`.
    *   **PaddlePaddle:** `paddlepaddle` (or `paddlepaddle-gpu` for CUDA). Install with `pip install paddlepaddle` or `pip install paddlepaddle-gpu`.
*   **Your Model Files:**
    *   For ONNX: Your `.onnx` model file.
    *   For TensorRT: Your `.engine` file (pre-built using TensorRT tools like `trtexec` or the TensorRT API).
    *   For PyTorch: Your `.pt` or `.pth` file (preferably a `state_dict`) AND the Python script defining your `torch.nn.Module` architecture.
    *   For OpenVINO™: Your `.xml` (topology) and `.bin` (weights) IR model files.
    *   For PaddlePaddle: Your `.pdmodel` (model structure) and `.pdiparams` (weights) inference model files.
*   **Knowledge of your Model:**
    *   **Input Shape:** Expected dimensions, e.g., `(1, 3, 640, 640)` (batch, channels, height, width).
    *   **Input Preprocessing Steps:** How raw images need to be transformed (resizing, normalization, color space conversion, channel order BGR->RGB, etc.).
    *   **Output Structure:** The shape and meaning of the output tensor(s) (e.g., `(batch_size, num_proposals, 5 + num_classes)` for detections).

This guide assumes you have these prerequisites set up and have access to your model files.

---

## 3. Loading the Model

This section describes how to load your `.onnx` or `.engine` model file using the respective runtime libraries.

### 3.1 Using ONNX Runtime for `.onnx` files

ONNX Runtime is a cross-platform inference and training accelerator.

```python
import onnxruntime
import numpy as np

# Path to your ONNX model
onnx_model_path = "path/to/your/model.yolov8n.onnx" # Example

# Create an inference session
# You can specify providers like 'CUDAExecutionProvider', 'CPUExecutionProvider'
# If 'CUDAExecutionProvider' is chosen, ensure onnxruntime-gpu is installed and CUDA is configured
providers = ['CPUExecutionProvider'] # Or ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    print(f"ONNX model loaded successfully from {onnx_model_path}")
    
    # Get model input details
    input_details = session.get_inputs()
    input_name = input_details[0].name
    input_shape = input_details[0].shape # e.g., [1, 3, 640, 640] for batch, channels, height, width
    input_type = input_details[0].type   # e.g., 'tensor(float)'
    print(f"Input Name: {input_name}, Input Shape: {input_shape}, Input Type: {input_type}")

    # Get model output details
    output_details = session.get_outputs()
    # For YOLO models, there might be one or more outputs
    # Example: one output tensor [batch_size, num_detections, 4_coords + 1_score + num_classes]
    # Or separate outputs for boxes, scores, classes.
    output_names = [output.name for output in output_details]
    output_shapes = [output.shape for output in output_details]
    print(f"Output Names: {output_names}, Output Shapes: {output_shapes}")

except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None # Or handle error as appropriate

# Later, you'll use session.run() for inference
# Example dummy input (replace with actual preprocessed image tensor)
# dummy_input = np.random.randn(*input_shape).astype(np.float32) 
# outputs = session.run(output_names, {input_name: dummy_input})
```

**Key points for ONNX Runtime:**
*   `InferenceSession(model_path, providers)`: Loads the model. The `providers` list determines the execution backend (CPU, GPU). ONNX Runtime will try them in order.
*   `get_inputs()`: Returns a list of `NodeArg` objects describing model inputs (name, shape, type). For most YOLO models, there's one image input.
*   `get_outputs()`: Returns a list of `NodeArg` objects for model outputs. You need to know what these outputs represent (e.g., raw detections, feature maps).

### 3.2 Using TensorRT Python API for `.engine` files

TensorRT is an SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications.

**Note:** Loading TensorRT engines and performing inference typically involves more boilerplate code than ONNX Runtime, especially regarding defining input/output tensor names, allocating GPU memory, and managing execution contexts. The `trt` Python module is usually required, often along with `pycuda` for memory management if working directly with CUDA.

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Important for initializing CUDA driver
import numpy as np

# Path to your TensorRT engine file
trt_engine_path = "path/to/your/model.engine" # Example

# Create a logger (TensorRT requires one)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # Or trt.Logger.INFO, trt.Logger.ERROR

# Deserialize the engine
runtime = trt.Runtime(TRT_LOGGER)
try:
    with open(trt_engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
except Exception as e:
    print(f"Error deserializing TensorRT engine: {e}")
    # Handle error appropriately
    exit()

# Create an execution context
# An engine can have multiple execution contexts, allowing multiple inferences to run in parallel on the same engine.
context = engine.create_execution_context()

# Allocate input and output buffers (host and device)
# This is a crucial and often complex part. You need to know the exact names, shapes, and dtypes of your model's inputs and outputs.
# These names are defined when you convert your model to a TensorRT engine.

# Example (highly dependent on your specific model):
# Let's assume your model has one input named "images" and one output named "output".
# These names must match what the TensorRT engine expects.

input_binding_idx = engine.get_binding_index("images") # Or your model's input tensor name
output_binding_idx = engine.get_binding_index("output") # Or your model's output tensor name

# Get expected input/output shapes and dtypes
input_shape = engine.get_binding_shape(input_binding_idx)
output_shape = engine.get_binding_shape(output_binding_idx)
input_dtype = trt.nptype(engine.get_binding_dtype(input_binding_idx))
output_dtype = trt.nptype(engine.get_binding_dtype(output_binding_idx))

# Allocate host (CPU) memory
h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=input_dtype)
h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=output_dtype)

# Allocate device (GPU) memory
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

# Create a CUDA stream for asynchronous execution
stream = cuda.Stream()

print(f"TensorRT engine loaded. Input: {input_shape}, Output: {output_shape}")
# The 'engine', 'context', 'h_input', 'h_output', 'd_input', 'd_output', 'stream',
# 'input_binding_idx', 'output_binding_idx' are now ready for inference.
# Remember to also store input_shape, output_shape, input_dtype, output_dtype.
```

**Important Considerations for TensorRT:**
*   **Tensor Names:** You *must* know the exact names of the input and output tensors as defined during the TensorRT engine creation. These are used to get binding indices.
*   **Dynamic Shapes:** If your engine was built with dynamic shapes, the process of setting input shapes and allocating buffers might be more involved, using `context.set_binding_shape()`.
*   **Memory Management:** You are responsible for allocating and freeing CUDA memory. `pycuda.autoinit` helps with initialization, but explicit deallocation (e.g., `d_input.free()`) might be needed in long-running applications, though `pycuda.autoinit` often handles cleanup on exit.
*   **Batching:** The `input_shape` and `output_shape` obtained from the engine will include the batch dimension.

This section has provided the boilerplate for loading ONNX and TensorRT models. The TensorRT part, in particular, requires careful attention to detail regarding tensor names and memory management, and often involves using `pycuda`.

### 3.3 Using PyTorch for `.pt` files

PyTorch models (`.pt` or `.pth` files) usually contain either the entire model structure and its learned weights, or just the `state_dict` (a dictionary of learned weights and buffers). The recommended practice is to save and load the `state_dict`.

To load a `.pt` file containing a `state_dict`, you first need to define your model architecture in Python using PyTorch's `torch.nn.Module`.

```python
import torch
import torchvision.models as models # Example: if using a standard torchvision model

# --- Step 1: Define your Model Architecture ---
# This MUST be the same architecture as the one whose state_dict was saved.
# If it's a custom model, you need its class definition.
# Example using a pretrained ResNet18 architecture (replace with your actual model)
# For YOLO models, you would need the specific YOLO architecture class definition.
# This is a placeholder; for a YOLO model, you'd define or import its nn.Module class.

# Let's assume you have a MyYOLOModel class defined elsewhere:
# from my_yolo_model_definition import MyYOLOModel
# model = MyYOLOModel(num_classes=80) # Or however it's instantiated

# For demonstration, let's use a generic ResNet as a placeholder.
# In a real YOLO scenario, this would be your YOLO model's class.
model_architecture = models.resnet18(weights=None) # Load architecture without pretrained weights
# If your .pt file saves the whole model (not just state_dict), and you trust the source:
# model = torch.load("path/to/your/model.pt")
# However, loading state_dict is safer and more common.

# --- Step 2: Load the state_dict ---
pt_model_path = "path/to/your/yolo_model_state_dict.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Load the state_dict
    state_dict = torch.load(pt_model_path, map_location=device)
    
    # If the state_dict was saved from a model wrapped in nn.DataParallel or nn.DistributedDataParallel,
    # keys might have a "module." prefix. You might need to remove it.
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] if k.startswith('module.') else k # remove `module.`
    #     new_state_dict[name] = v
    # model_architecture.load_state_dict(new_state_dict)

    model_architecture.load_state_dict(state_dict)
    print(f"PyTorch model state_dict loaded successfully from {pt_model_path}")

except Exception as e:
    print(f"Error loading PyTorch state_dict: {e}")
    # Handle error appropriately
    exit()

# --- Step 3: Set the model to evaluation mode ---
# This is crucial for layers like Dropout and BatchNorm to behave correctly during inference.
model_architecture.eval()

# --- Step 4: Move the model to the desired device ---
model_architecture.to(device)

print(f"PyTorch model '{pt_model_path}' loaded and set to eval mode on {device}.")
# The 'model_architecture' is now ready for inference.
# Ensure you also know the expected input preprocessing steps for this model.
```

**Important Considerations for PyTorch `.pt` files:**
*   **Model Definition:** You *must* have the Python class definition of the model whose `state_dict` you are loading. The architecture must match exactly.
*   **`state_dict` vs. Full Model:**
    *   Loading `state_dict` (as shown) is safer and more flexible.
    *   `torch.load("model.pt")` can load an entire model object (if saved that way), but this is less common for sharing and can be a security risk if the source is untrusted, as it involves unpickling arbitrary code. For `weights_only=True` (default in PyTorch 2.6+ if `pickle_module` is not passed), `torch.load` will restrict unpickling.
*   **Device Mapping:** Use `map_location` in `torch.load()` to control which device the tensors are loaded onto initially (e.g., `map_location=torch.device('cpu')` if loading a GPU-trained model on a CPU-only machine). Then, use `model.to(device)` to move the model to your target inference device.
*   **`model.eval()`:** Always call `model.eval()` before inference to set dropout and batch normalization layers to evaluation mode.

### 3.4 Using OpenVINO™ for IR format (.xml + .bin)

OpenVINO™ (Open Visual Inference & Neural network Optimization) toolkit optimizes and deploys deep learning models. Its Intermediate Representation (IR) format consists of an `.xml` file (describing the network topology) and a `.bin` file (containing the weights and biases).

**Note on Model Formats:**
*   **IR Format (Primary focus here):** This section primarily details loading pre-converted IR models (`.xml` + `.bin`). These are typically generated using OpenVINO's Model Converter (`ovc` CLI tool) or the `openvino.convert_model()` Python function from other formats like ONNX, PyTorch, TensorFlow, etc. Converting to IR first can allow for ahead-of-time optimizations and is a common deployment practice.
*   **Direct ONNX Loading:** OpenVINO™ Core (`ov.Core()`) can also directly load ONNX models. You can use `model = core.read_model(model="path/to/your/model.onnx")` and then proceed to compile it with `core.compile_model()`. This can simplify the workflow if you are starting with an ONNX file and do not require a separate IR conversion step.
*   **Direct PyTorch/TensorFlow Loading:** The `openvino.convert_model()` function can also take a PyTorch or TensorFlow model object directly, convert it, and return an `ov.Model` object ready for compilation, e.g., `ov_model = ov.convert_model(pytorch_model_object, example_input=...)`.

This guide focuses on the scenario where you have existing IR files or want to understand the IR-based deployment.

```python
import openvino as ov # Preferred new API (formerly openvino.runtime)
import numpy as np

# --- Step 1: Initialize OpenVINO Core ---
core = ov.Core()

# --- Step 2: Read the IR Model ---
# The .xml file describes the model topology, and the .bin file contains the weights.
# The .bin file is usually inferred if it has the same base name and is in the same directory.
ir_model_xml_path = "path/to/your/model.xml"
# ir_model_bin_path = "path/to/your/model.bin" # Usually not needed if names match

try:
    model = core.read_model(model=ir_model_xml_path) # Optionally, weights=ir_model_bin_path
    print(f"OpenVINO IR model loaded successfully from {ir_model_xml_path}")
except Exception as e:
    print(f"Error loading OpenVINO IR model: {e}")
    # Handle error appropriately
    exit()

# --- Step 3: (Optional) Inspect Model Inputs/Outputs ---
# This helps in understanding the expected input tensor name, shape, and type.
# And similarly for outputs.
input_tensor = model.input(0) # Get the first input tensor (or by name: model.input("input_name"))
output_tensor = model.output(0) # Get the first output tensor (or by name: model.output("output_name"))

input_name = input_tensor.any_name
input_shape = input_tensor.shape # e.g., [1, 3, 640, 640]
input_type = input_tensor.element_type # e.g., <Type: 'float32'>

print(f"Model input name: {input_name}, shape: {input_shape}, type: {input_type}")
print(f"Model output name: {output_tensor.any_name}, shape: {output_tensor.shape}, type: {output_tensor.element_type}")


# --- Step 4: Compile the Model for a Target Device ---
# Available devices: "CPU", "GPU", "NPU", "AUTO", etc.
# "AUTO" selects the best available device automatically.
device_name = "CPU" # Or "GPU", "AUTO"
try:
    compiled_model = core.compile_model(model=model, device_name=device_name)
    print(f"Model compiled successfully for device: {device_name}")
except Exception as e:
    print(f"Error compiling model for device {device_name}: {e}")
    exit()

# The 'compiled_model' is now ready for inference.
# It's good practice to also retrieve the actual input layer name from the compiled_model
# as it might differ slightly or be more specific after compilation.
# For single-input models, compiled_model.input(0) or compiled_model.inputs[0] works.
# For named inputs: compiled_model.input("input_layer_name_from_ir")

# Ensure you have the actual input tensor name and shape for preprocessing
compiled_input_tensor = compiled_model.input(0) # Or by name
compiled_output_tensor = compiled_model.output(0) # Or by name

print(f"Compiled model input: {compiled_input_tensor.any_name}, shape: {compiled_input_tensor.shape}")
print(f"Compiled model output: {compiled_output_tensor.any_name}, shape: {compiled_output_tensor.shape}")

# Store the compiled_model and potentially the input_name/shape for the inference step.
# The compiled_model object handles the specifics of the target device.
```

**Important Considerations for OpenVINO™ IR:**
*   **`openvino` package:** Ensure you have the `openvino` Python package installed (`pip install openvino openvino-dev`). The `openvino-dev` package includes tools like Model Converter.
*   **IR Format / Direct Loading:** Decide whether to convert your model to IR format first (`.xml` & `.bin`) for potential ahead-of-time optimizations, or load an ONNX model directly using `core.read_model()`.
*   **`ov.Core()`:** This is the entry point to OpenVINO runtime functionality.
*   **`core.read_model()`:** Loads the model from `.xml` and `.bin` files (for IR) or directly from an `.onnx` file.
*   **`core.read_model()`:** Loads the model from the `.xml` and `.bin` files.
*   **`core.compile_model()`:** Compiles the model for a specific target device (e.g., "CPU", "GPU"). This step performs device-specific optimizations. The `compiled_model` object is what you use for inference.
*   **Input/Output Tensors:** You can inspect the model's input and output tensors using `model.inputs` and `model.outputs` (or `model.input(index_or_name)`, `model.output(index_or_name)`). This is crucial for knowing the tensor names, shapes, and data types required for preprocessing and expected from postprocessing. The `compiled_model` also has similar properties.
*   **Inference Request (Implicit):** The modern API often uses `compiled_model(inputs_dict)` directly for synchronous inference, which implicitly creates an inference request. For more control (e.g., asynchronous inference), you can create an `InferRequest` object: `infer_request = compiled_model.create_infer_request()`.

### 3.5 Using PaddlePaddle for `.pdmodel` + `.pdiparams`

PaddlePaddle is an open-source deep learning platform. For inference, models are typically saved in a two-file format: a `.pdmodel` file containing the model structure and a `.pdiparams` file for the weights. Sometimes, a single `.pdmodel` can also be used if weights are embedded (less common for separate saving/loading).

The `paddle.inference` API is used for this.

```python
import paddle.inference as paddle_inf
import numpy as np

# --- Step 1: Create an Inference Config ---
# Path to your inference model files
model_file_path = "path/to/your/inference.pdmodel"
params_file_path = "path/to/your/inference.pdiparams"

config = paddle_inf.Config(model_file_path, params_file_path)

# --- Step 2: Configure the Predictor (Optional but Recommended) ---
# Enable GPU (if available and desired)
# if paddle.is_compiled_with_cuda():
#     config.enable_use_gpu(100, 0) # 100MB GPU memory, device_id 0
# else:
#     config.disable_gpu()

# CPU threads (example)
# config.set_cpu_math_library_num_threads(4)

# Enable MKLDNN for CPU acceleration (if applicable)
# config.enable_mkldnn()

# Other options: config.enable_tensorrt_engine(...), config.switch_ir_optim(), etc.
# For basic CPU inference, often the defaults are fine or minimal config is needed.
print("PaddlePaddle Config initialized.")

# --- Step 3: Create the Predictor ---
try:
    predictor = paddle_inf.create_predictor(config)
    print("PaddlePaddle Predictor created successfully.")
except Exception as e:
    print(f"Error creating PaddlePaddle Predictor: {e}")
    # Handle error appropriately
    exit()

# --- Step 4: (Optional) Get Input/Output Names and Handles ---
# This helps understand how to provide input and retrieve output.
input_names = predictor.get_input_names()
output_names = predictor.get_output_names()

print(f"Model input names: {input_names}") # e.g., ['x']
print(f"Model output names: {output_names}") # e.g., ['save_infer_model/scale_0.tmp_1']

# Get input handle (assuming first input)
if not input_names:
    print("Error: No input names found in the model.")
    exit()
input_handle = predictor.get_input_handle(input_names[0])
# input_shape = input_handle.shape() # Shape can be dynamic here, set during run
# print(f"Input '{input_names[0]}' expected shape (can be dynamic): {input_shape}")

# The 'predictor', 'input_names', 'output_names', and 'input_handle' (and similar for outputs)
# are now ready for the inference step. The predictor object encapsulates the loaded model.
```

**Important Considerations for PaddlePaddle Inference:**
*   **`paddlepaddle` package:** Ensure `paddlepaddle` (or `paddlepaddle-gpu`) is installed (`pip install paddlepaddle`).
*   **Model Format:** You need the inference model files (`.pdmodel` and `.pdiparams`), which are typically exported from a trained model using `paddle.jit.save` or specialized export tools within model suites like PaddleOCR/PaddleDetection.
*   **`paddle.inference` API:** This is the standard Python API for PaddlePaddle inference. It provides classes like `Config` and `Predictor` (created via `create_predictor`). While underlying C++ APIs might evolve (e.g., namespace changes to `paddle_infer`), the Python `paddle.inference` module remains the consistent entry point.
*   **`paddle.inference.Config`:** This object is crucial for specifying model paths and configuring the inference engine (CPU/GPU, MKLDNN, TensorRT, etc.).
*   **`paddle.inference.create_predictor()`:** Creates the inference engine instance.
*   **Input/Output Handles:** You interact with the model's inputs and outputs via handles obtained using `predictor.get_input_handle(name)` and `predictor.get_output_handle(name)`.
*   **Data Transfer:** Data is copied to the input tensor handle using `input_handle.copy_from_cpu(numpy_array)` (or `copy_from_gpu` if applicable).

## 4. Manual Image Preprocessing

Before feeding an image to the YOLO model, it must be preprocessed to match the model's expected input format. This typically involves loading the image, resizing/padding it (letterboxing), converting color formats, arranging dimensions, and normalizing pixel values.

Let's assume your model expects a fixed input size, e.g., `(640, 640)` pixels, with 3 color channels (RGB), and pixel values normalized to `[0.0, 1.0]`. The input tensor shape would be `(batch_size, 3, height, width)`.

### 4.1 Image Loading

We'll use OpenCV (`cv2`) to load images.

```python
import cv2
import numpy as np

image_path = "path/to/your/image.jpg"
original_image = cv2.imread(image_path) # Reads in BGR format by default

if original_image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

original_height, original_width = original_image.shape[:2]
print(f"Loaded image: {image_path}, Original Dims: ({original_width}W x {original_height}H)")
```

### 4.2 Letterboxing (Resize and Pad)

Letterboxing resizes an image to the target input dimensions while maintaining its aspect ratio. The remaining area is filled with padding (typically a neutral color like gray).

**Parameters:**
*   `img`: The original image (NumPy array).
*   `new_shape`: The target dimensions (height, width) for the model, e.g., `(640, 640)`.
*   `color`: The padding color (B, G, R), e.g., `(114, 114, 114)` for gray.
*   `auto`: If `True`, computes padding to achieve minimal rectangular inference (not fully implemented here for simplicity, focuses on fixed `new_shape`).
*   `scaleFill`: If `True`, stretches image to fill `new_shape` without maintaining aspect ratio (not typical for YOLO letterboxing).
*   `scaleup`: If `True`, allows upscaling the image if its dimensions are smaller than `new_shape`. If `False`, only downscales.
*   `stride`: The model stride (e.g., 32). Ensures padded dimensions are multiples of the stride. (Simplified here, full stride alignment adds complexity).

```python
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)  # only scale down, do not scale up (for better test mAP)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    # For minimal rectangular inference (auto=True), dw and dh should be multiples of stride
    # This simplified version assumes fixed new_shape and basic padding
    # if auto:
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride) # wh padding

    if scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # wh ratios
    else:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh) # image, ratio, padding

# Example Usage:
model_input_shape = (640, 640) # Desired model input (height, width)
padded_image, ratio, (pad_w, pad_h) = letterbox(original_image.copy(), model_input_shape, auto=False) # Set auto to True for stride alignment if implemented

padded_height, padded_width = padded_image.shape[:2]
print(f"Padded image Dims: ({padded_width}W x {padded_height}H)")
# cv2.imshow("Padded Image", padded_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
```
**Note on `auto=True` and `stride`:**
The `LetterBox` utility in Ultralytics, when `auto=True`, adjusts padding to ensure the dimensions are multiples of the model's `stride` (e.g., 32). This is optimal for some model architectures. The simplified version above primarily focuses on fitting the image into `new_shape`. For perfect replication of Ultralytics `auto=True` behavior, you'd need to implement the stride alignment logic for `dw` and `dh`.

### 4.3 Color Conversion, Normalization, and Tensor Preparation

After letterboxing, the image needs further processing:
1.  **BGR to RGB:** OpenCV loads images as BGR. Most models expect RGB.
2.  **HWC to CHW:** Transpose dimensions from (Height, Width, Channels) to (Channels, Height, Width).
3.  **Normalization:** Convert pixel values from `0-255` (uint8) to `0.0-1.0` (float32 or float16).
4.  **Contiguous Array:** Ensure the NumPy array is C-contiguous in memory for efficient processing.
5.  **Batch Dimension:** Add a batch dimension at the beginning (e.g., `(1, C, H, W)` for a single image).

```python
# 1. BGR to RGB
img_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

# 2. HWC to CHW (Height, Width, Channels) to (Channels, Height, Width)
img_chw = np.transpose(img_rgb, (2, 0, 1))

# 3. Normalization (0-255 to 0.0-1.0) and Type Conversion
# Assuming model expects float32 input
img_normalized = img_chw.astype(np.float32) / 255.0

# 4. Ensure Contiguous Array
img_contiguous = np.ascontiguousarray(img_normalized)

# 5. Add Batch Dimension
# Input tensor shape for the model is typically (batch_size, channels, height, width)
input_tensor = np.expand_dims(img_contiguous, axis=0) # Adds batch dimension: (1, C, H, W)

print(f"Final input tensor shape: {input_tensor.shape}, Data type: {input_tensor.dtype}")

# This input_tensor is now ready to be fed into the ONNX Runtime or TensorRT session.
```

This completes the manual preprocessing. The `input_tensor` is now in the format expected by many YOLO models.

---

## 5. Performing Inference

Once the model is loaded (Section 3) and the input image is preprocessed (Section 4) into `input_tensor`, you can perform inference.

### 5.1 ONNX Runtime Inference

Using the `session` object created in Section 3.1:

```python
# Assuming 'session', 'input_name', 'output_names', and 'input_tensor' are defined
# from previous sections.

# Ensure input_tensor matches the expected type (e.g., float32)
# input_tensor was already converted to np.float32 in preprocessing.

# Perform inference
try:
    # The input to session.run is a dictionary mapping input names to NumPy arrays.
    # The output is a list of NumPy arrays, corresponding to the output_names.
    raw_outputs = session.run(output_names, {input_name: input_tensor})
    print(f"ONNX inference successful. Received {len(raw_outputs)} output tensor(s).")
    
    # Example: if you have one output tensor (common for many YOLO models)
    # raw_prediction = raw_outputs[0]
    # print(f"Shape of raw prediction: {raw_prediction.shape}")
    # Example output shape: (batch_size, num_proposals, 4_coords + 1_conf + num_classes)
    # e.g., (1, 8400, 85) for YOLOv8 with 80 classes (8400 proposals, 4 bbox + 1 conf + 80 classes)

except Exception as e:
    print(f"Error during ONNX inference: {e}")
    raw_outputs = [] # Or handle error as appropriate
```

The `raw_outputs` will be a list of NumPy arrays. The structure and meaning of these arrays depend on how your specific YOLO model was exported to ONNX. You need to understand this structure for postprocessing.

### 5.2 TensorRT Inference

Using the `engine`, `context`, host/device buffers (`h_input`, `d_input`, `h_output`, `d_output`), and `bindings` prepared in Section 3.2. You'll also need `pycuda.driver` if you haven't imported it as `cuda`.

```python
import pycuda.driver as cuda # Ensure PyCUDA is imported
# import pycuda.autoinit # Typically called once at the start of your script if using PyCUDA

# Assuming 'context', 'h_input', 'd_input', 'h_output', 'd_output',
# 'bindings', 'input_tensor', 'engine' are defined from previous sections.
# And 'stream' if you are using asynchronous execution.

# Ensure input_tensor is correctly shaped and typed for h_input
# h_input was created with the correct shape and dtype from engine properties.
np.copyto(h_input, input_tensor.ravel()) # Copy preprocessed data into pinned host memory buffer
                                        # .ravel() flattens, ensure h_input expects this if not (1,C,H,W)
                                        # If h_input is already (1,C,H,W), direct assignment is fine:
# h_input[:] = input_tensor 

try:
    # 1. Transfer input data from Host (CPU) to Device (GPU)
    cuda.memcpy_htod(d_input, h_input) # For synchronous copy
    # cuda.memcpy_htod_async(d_input, h_input, stream) # For asynchronous copy

    # 2. Execute Inference
    # For synchronous execution:
    # context.execute_v2(bindings=bindings) 
    # For newer TRT versions. For older, it might be context.execute(batch_size=1, bindings=bindings)
    # Check your TensorRT version documentation if execute_v2 is not found.
    # If your engine has dynamic shapes, ensure context.set_binding_shape was called if needed.
    if not context.execute_v2(bindings=bindings):
        print("TensorRT inference execution failed")
        # Handle error

    # For asynchronous execution:
    # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # stream.synchronize() # Wait for the stream to complete

    # 3. Transfer predictions from Device (GPU) to Host (CPU)
    cuda.memcpy_dtoh(h_output, d_output) # For synchronous copy
    # cuda.memcpy_dtoh_async(h_output, d_output, stream) # For asynchronous copy
    # stream.synchronize() # If using async copy

    # h_output now contains the raw model predictions as a flat NumPy array.
    # You need to reshape it based on the known output shape of your TensorRT engine.
    # Example output_shape obtained in Section 3.2: (1, 25200, 85)
    # raw_prediction_trt = h_output.reshape(output_shape_from_engine) # output_shape_from_engine was determined at load time
    # print(f"TensorRT inference successful. Raw output shape: {raw_prediction_trt.shape}")

except Exception as e:
    print(f"Error during TensorRT inference: {e}")
    # Potentially clean up CUDA resources if error is unrecoverable

# Do not forget to free CUDA memory when done (e.g., at the end of your application)
# d_input.free()
# d_output.free()
# if stream: stream.detach()
```

**Key points for TensorRT Inference:**
*   **Data Transfer:** Explicitly copy data between CPU (host) and GPU (device) memory using `cuda.memcpy_htod` (host to device) and `cuda.memcpy_dtoh` (device to host).
*   **Execution:** `context.execute_v2(bindings=bindings)` is the common way to run inference. For older TensorRT versions, it might be `context.execute(batch_size=..., bindings=bindings)`.
*   **Asynchronous Execution:** For better performance, especially with multiple streams or models, use asynchronous copies (`memcpy_htod_async`, `memcpy_dtoh_async`) and execution (`execute_async_v2`) along with CUDA streams (`cuda.Stream()`). Remember to synchronize the stream (`stream.synchronize()`) before accessing the output data on the host.
*   **Output Reshaping:** The `h_output` buffer will be a flat 1D array after `memcpy_dtoh`. You must reshape it to the actual multi-dimensional output tensor shape that your model produces (e.g., `(batch_size, num_proposals, num_attributes)`).
*   **Memory Management:** Remember to free allocated GPU memory (`d_input.free()`, `d_output.free()`) when it's no longer needed to prevent memory leaks.

After these steps, `raw_outputs` (from ONNX) or the reshaped `h_output` (from TensorRT) will hold the raw numerical predictions from the model. The next step is postprocessing.

### 5.3 PyTorch Inference

Using the `model_architecture` (which is an `nn.Module`) loaded and prepared in Section 3.3:

```python
# Assuming 'model_architecture', 'device', and 'preprocessed_input_tensor' (from Section 4.2) are defined.
# 'preprocessed_input_tensor' should be a PyTorch tensor on the correct 'device'.

# Ensure model is on the correct device and in eval mode (already done in Section 3.3)
# model_architecture.to(device)
# model_architecture.eval()

# Convert preprocessed NumPy array (from Section 4.2, which was (1, H, W, C) or (1, C, H, W))
# to a PyTorch tensor if it's not already.
# The preprocessing in 4.2 created 'input_tensor' as a NumPy array with shape (1, 3, H, W)
# and type float32. This is suitable for PyTorch.

# input_torch_tensor = torch.from_numpy(input_tensor).to(device)
# If your preprocessing already produces a PyTorch tensor on the correct device, use that directly.
# Let's assume 'input_tensor' from section 4.2 is what we need to convert.
# It was: input_tensor = np.expand_dims(transposed_image, axis=0).astype(np.float32)

pytorch_input_tensor = torch.from_numpy(input_tensor).to(device)


# Perform inference
try:
    with torch.no_grad(): # Important: disable gradient calculations for inference
        raw_outputs = model_architecture(pytorch_input_tensor)
    
    # Depending on your model, raw_outputs might be a single tensor, a tuple of tensors, or a dict.
    # For many YOLO models, it's often a tensor or a list/tuple of tensors.
    # Example: if it's a single tensor:
    # output_data_pytorch = raw_outputs.cpu().numpy() # Move to CPU and convert to NumPy for postprocessing
    
    # If raw_outputs is a tuple (e.g., some YOLO versions might return multiple heads):
    if isinstance(raw_outputs, tuple) or isinstance(raw_outputs, list):
        # Process or concatenate them as needed. For now, let's assume the first one is primary.
        # This is highly model-specific.
        # For simplicity, let's take the first output if it's a list/tuple and convert to NumPy.
        # You'll need to adapt this based on your specific YOLO model's output structure.
        if raw_outputs:
            output_data_pytorch = raw_outputs[0].cpu().numpy() 
            if len(raw_outputs) > 1:
                print(f"PyTorch model returned multiple output tensors ({len(raw_outputs)}). Using the first one for this example.")
                # You might need to handle other tensors as well for full postprocessing.
        else:
            raise ValueError("PyTorch model returned empty outputs.")

    else: # Assuming it's a single tensor
        output_data_pytorch = raw_outputs.cpu().numpy()

    print(f"PyTorch inference successful. Output shape: {output_data_pytorch.shape}")
    # 'output_data_pytorch' (NumPy array) now holds the raw model output.

except Exception as e:
    print(f"Error during PyTorch inference: {e}")
    # Handle error appropriately
    # output_data_pytorch = None
    exit()

# 'output_data_pytorch' can now be used in Section 6: Manual Postprocessing.
# Ensure its shape and content match what your postprocessing logic expects.
```

**Key Points for PyTorch Inference:**
*   **`torch.no_grad()`:** Wrap the inference call in this context manager to disable gradient computations, which saves memory and speeds up execution.
*   **Input Tensor:** Ensure your input data is a PyTorch tensor (`torch.Tensor`), on the same device as the model, and has the correct shape and data type.
*   **Output:** The output from the model will also be a PyTorch tensor (or a tuple/list of tensors). Convert it to a NumPy array using `.cpu().numpy()` if you plan to use NumPy-based postprocessing functions. Moving to CPU first (`.cpu()`) is important if the model was on a GPU.

### 5.4 Using OpenVINO™ Compiled Model for Inference

Using the `compiled_model` from Section 3.4 and the preprocessed `input_tensor` (NumPy array) from Section 4.2:

```python
# Assuming 'compiled_model' (from Section 3.4) and 'input_tensor' (NumPy array from Section 4.2) are defined.
# 'input_tensor' from Section 4.2 is a NumPy array: (1, 3, H, W), float32. This is suitable.

# OpenVINO expects input data as a dictionary where keys are input tensor names (or indices)
# and values are the NumPy arrays.

# Get the input tensor name from the compiled model (safer)
# In Section 3.4, we stored: compiled_input_tensor = compiled_model.input(0)
# input_name_ov = compiled_input_tensor.any_name # Or use the actual name if known

# For a single input model, you can also get the input layer directly:
# (This was shown in section 3.4, let's assume compiled_input_tensor is available)
# input_layer_ov = compiled_model.input(0) # Gets the first input layer

# Perform inference
# The modern API allows calling the compiled_model directly like a function.
# This performs synchronous inference.
try:
    # Create the input dictionary.
    # For models with a single input, the key can be the input tensor object, its name, or index 0.
    # For multiple inputs, provide a dictionary for all of them.
    # input_dict_ov = {input_layer_ov.any_name: input_tensor} # Using the name
    input_dict_ov = {compiled_model.input(0): input_tensor} # Using the input tensor object from compiled model

    results_ov = compiled_model(inputs=input_dict_ov)
    
    # 'results_ov' is a dictionary where keys are output tensor objects (or names)
    # and values are NumPy arrays.
    # For a single output model:
    # output_data_ov = results_ov[compiled_model.output(0)] # Access by output tensor object
    
    # Let's assume compiled_output_tensor was obtained in section 3.4
    # compiled_output_tensor = compiled_model.output(0)
    output_data_ov = results_ov[compiled_model.output(0)] # Or use its actual name
    
    print(f"OpenVINO inference successful. Output shape: {output_data_ov.shape}")
    # 'output_data_ov' (NumPy array) now holds the raw model output.

except Exception as e:
    print(f"Error during OpenVINO inference: {e}")
    # Handle error appropriately
    # output_data_ov = None
    exit()

# 'output_data_ov' can now be used in Section 6: Manual Postprocessing.
# Ensure its shape and content match what your postprocessing logic expects.

# For asynchronous inference or more control, you would create an infer request:
# infer_request = compiled_model.create_infer_request()
# infer_request.infer(inputs={compiled_model.input(0): input_tensor})
# output_data_ov = infer_request.get_output_tensor().data
# Or for multiple outputs:
# results_ov = infer_request.results # Dictionary of results
```

**Key Points for OpenVINO™ Inference:**
*   **Input Data:** Input data is typically a dictionary mapping input tensor names (or input tensor objects from `compiled_model.inputs`) to NumPy arrays.
*   **Synchronous Inference:** Calling `compiled_model(inputs_dict)` performs synchronous inference.
*   **Output Data:** The result is a dictionary where keys are output tensor objects (from `compiled_model.outputs`) and values are the corresponding NumPy arrays.
*   **Asynchronous Inference:** For higher performance in applications that can benefit from it (e.g., processing multiple streams), use `compiled_model.create_infer_request()` and then `infer_request.start_async()` or `infer_request.infer()` for subsequent calls on the request object.

### 5.5 Using PaddlePaddle for Inference

Using the `predictor` created in Section 3.5 and the preprocessed `input_tensor` (NumPy array) from Section 4.2:

```python
# Assuming 'predictor' (from Section 3.5) and 'input_tensor' (NumPy array from Section 4.2) are defined.
# 'input_tensor' from Section 4.2 is a NumPy array: (1, 3, H, W), float32. This is suitable.

# Get input/output names (already done in Section 3.5, assuming they are stored)
# input_names = predictor.get_input_names()
# output_names = predictor.get_output_names()

if not input_names or not output_names:
    print("Error: Input or output names not available from predictor.")
    # Handle error (this check should ideally be in the loading phase)
    exit()

input_name = input_names[0] # Assuming single input
output_name = output_names[0] # Assuming single output, adapt if multiple

# Get input handle
input_handle = predictor.get_input_handle(input_name)

# Set input data
# The shape of the input_tensor must match what the model expects for this handle.
# For dynamic shapes, the model adapts to the shape of the data set here.
input_handle.reshape(input_tensor.shape) # Reshape if necessary, or ensure it matches
input_handle.copy_from_cpu(input_tensor)

# Perform inference
try:
    predictor.run() # This runs the inference
    print("PaddlePaddle inference executed.")

except Exception as e:
    print(f"Error during PaddlePaddle predictor.run(): {e}")
    # Handle error appropriately
    exit()

# Get output data
output_handle = predictor.get_output_handle(output_name)
output_data_paddle = output_handle.copy_to_cpu() # Returns a NumPy array

print(f"PaddlePaddle inference successful. Output shape: {output_data_paddle.shape}")
# 'output_data_paddle' (NumPy array) now holds the raw model output.

# 'output_data_paddle' can now be used in Section 6: Manual Postprocessing.
# Ensure its shape and content match what your postprocessing logic expects.
```

**Key Points for PaddlePaddle Inference:**
*   **Predictor Object:** The `predictor` object created from `paddle.inference.Config` is used for inference.
*   **Input Handles:** Get an input handle using `predictor.get_input_handle(input_name)`.
*   **`copy_from_cpu()`:** Use `input_handle.copy_from_cpu(numpy_array)` to feed your preprocessed NumPy array to the model. You might need to call `input_handle.reshape()` if your model supports dynamic input shapes and your input tensor shape varies.
*   **`predictor.run()`:** Executes the inference.
*   **Output Handles:** Get an output handle using `predictor.get_output_handle(output_name)`.
*   **`copy_to_cpu()`:** Use `output_handle.copy_to_cpu()` to retrieve the inference results as a NumPy array.

---

## 6. Manual Postprocessing (Detection Example)

Postprocessing converts the raw numerical output from the model into meaningful detections (bounding boxes, class labels, and confidence scores).

### 6.1 Understanding Raw Model Output

This is the most critical and model-dependent step. **You MUST know the exact structure of your model's output tensor(s).**

Common output formats for YOLO detection models (after any necessary reshaping):
*   **Single Tensor Output:** A single tensor of shape `(batch_size, num_proposals, 4_coordinates + 1_objectness_score + num_classes_scores)`. 
    *   `num_proposals`: Number of potential detections (e.g., 8400, 25200). This often corresponds to the total number of anchor boxes or grid cells considered by the model across different feature map scales.
    *   `4_coordinates`: Typically `[center_x, center_y, width, height]` (xywh) or `[x_min, y_min, x_max, y_max]` (xyxy). These are usually normalized to the model's input image size (e.g., 640x640).
    *   `1_objectness_score`: Confidence that an object is present in this proposal.
    *   `num_classes_scores`: Scores for each class. The highest score indicates the predicted class.
*   **Multiple Tensor Outputs:** Some models might output boxes, scores, and class IDs in separate tensors.

**Example: YOLOv8-like output (single tensor)**
Let's assume the output tensor `P` has shape `(1, 8400, 85)` for a batch size of 1, 8400 proposals, and a model trained on COCO (80 classes). So, 85 = 4 (bbox) + 1 (confidence) + 80 (class scores).

*   `P[0, :, 0:4]` would be the bounding box coordinates (e.g., cx, cy, w, h).
*   `P[0, :, 4]` would be the objectness/confidence score.
*   `P[0, :, 5:]` would be the 80 class scores.

To get the final class and its score for each proposal:
*   `box_confidence = P[0, :, 4]`
*   `class_scores = P[0, :, 5:]`
*   `final_confidence = box_confidence * class_scores.max(axis=-1)` (or just `box_confidence` if class scores are already scaled by it, model dependent)
*   `predicted_class_id = class_scores.argmax(axis=-1)`

**You must verify this with your specific model export settings and documentation.** Tools like Netron can help visualize ONNX model structures and identify output tensor names and shapes.

### 6.2 Non-Maximum Suppression (NMS) from Scratch

NMS filters overlapping bounding boxes, keeping only the ones with the highest confidence for each detected object.

**Algorithm:**
1.  Select all detection proposals with confidence above a certain threshold (`conf_threshold`).
2.  For each class:
    a.  Sort the remaining proposals for that class by their confidence scores in descending order.
    b.  Take the proposal with the highest score and add it to the list of final detections.
    c.  Calculate the Intersection over Union (IoU) of this proposal with all other remaining proposals for the same class.
    d.  Discard (suppress) any proposals that have an IoU greater than a certain threshold (`iou_threshold`) with the selected proposal.
    e.  Repeat from step 2b until no proposals are left for the current class.
3.  Combine final detections from all classes.

**6.2.1 Calculate Intersection over Union (IoU)**

IoU measures the overlap between two bounding boxes.

```python
def calculate_iou(box1, box2):
    """
    Calculate IoU of two bounding boxes.
    Boxes are in [x_min, y_min, x_max, y_max] format.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou
```

**6.2.2 NMS Implementation**

```python
def non_max_suppression_scratch(predictions, conf_threshold, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on raw model predictions.

    Args:
        predictions (np.array): Model output, assumed to be [num_proposals, 4_coords + 1_conf + num_classes].
                                 Coordinates are assumed to be [cx, cy, w, h] initially.
        conf_threshold (float): Minimum confidence score to consider a detection.
        iou_threshold (float): IoU threshold for suppressing overlapping boxes.

    Returns:
        list: List of final detections, each detection is [x1, y1, x2, y2, conf, class_id].
    """
    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    # x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2
    boxes_cxcywh = predictions[:, :4]
    confidences_obj = predictions[:, 4] # Objectness score
    class_probs = predictions[:, 5:]    # Class probabilities

    # Calculate final confidences and class IDs
    # Method 1: Multiply objectness by class probability
    # class_confidences = confidences_obj[:, np.newaxis] * class_probs 
    # Method 2: Use objectness score directly if class_probs are independent scores for boxes already past an objectness check in model
    # Or, if class_probs already factor in objectness (depends on model export)
    # For simplicity, let's find max class prob and multiply by objectness
    max_class_probs = np.max(class_probs, axis=1)
    class_ids = np.argmax(class_probs, axis=1)
    final_confidences = confidences_obj * max_class_probs # Element-wise if necessary

    # Filter by confidence threshold
    keep = final_confidences > conf_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    final_confidences = final_confidences[keep]
    class_ids = class_ids[keep]

    if not len(boxes_cxcywh):
        return []

    # Convert to [x1, y1, x2, y2] for IoU calculation
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    final_detections = []
    unique_class_ids = np.unique(class_ids)

    for class_id in unique_class_ids:
        class_mask = (class_ids == class_id)
        class_boxes = boxes_xyxy[class_mask]
        class_confs = final_confidences[class_mask]

        if not len(class_boxes):
            continue

        # Sort by confidence (descending)
        sorted_indices = np.argsort(class_confs)[::-1]
        class_boxes = class_boxes[sorted_indices]
        class_confs = class_confs[sorted_indices]

        current_class_detections = []
        while len(class_boxes) > 0:
            # Pick the box with highest confidence
            best_box = class_boxes[0]
            best_conf = class_confs[0]
            current_class_detections.append([*best_box, best_conf, class_id])

            # Remove it from the list
            class_boxes = class_boxes[1:]
            class_confs = class_confs[1:]

            if not len(class_boxes):
                break

            # Calculate IoU with remaining boxes
            ious = np.array([calculate_iou(best_box, other_box) for other_box in class_boxes])
            
            # Keep boxes with IoU <= threshold
            keep_mask = ious <= iou_threshold
            class_boxes = class_boxes[keep_mask]
            class_confs = class_confs[keep_mask]
        
        final_detections.extend(current_class_detections)
        
    return final_detections

# Example Usage (assuming 'raw_prediction' from ONNX/TensorRT output):
# raw_prediction shape e.g. (1, 8400, 85) -> take first batch element raw_prediction[0]
# filtered_detections = non_max_suppression_scratch(raw_prediction[0], conf_threshold=0.25, iou_threshold=0.45)
# for det in filtered_detections:
#     print(f"Box: {det[0:4]}, Conf: {det[4]:.2f}, Class: {int(det[5])}")
```
**Important considerations for NMS:**
*   **Coordinate Format:** Ensure your box coordinates are in `[x_min, y_min, x_max, y_max]` format for the `calculate_iou` function. If your model outputs `[center_x, center_y, width, height]`, convert them first.
*   **Efficiency:** The NMS implementation above is for clarity. For a large number of proposals, it can be slow due to Python loops. Vectorized NumPy operations or specialized libraries (if allowed) would be much faster.
*   **Multi-Class NMS:** The example performs NMS class-by-class. Agnostic NMS (ignoring class) is another variant where all boxes are processed together.
*   **Confidence Calculation:** The way `final_confidences` are calculated (`objectness_score * class_score` or just one of them) depends heavily on the model architecture and how it was trained/exported. Consult your model's specifics.

### 6.3 Scaling Coordinates to Original Image Size

The bounding boxes from NMS are relative to the letterboxed/padded input image (e.g., 640x640). They need to be scaled back to the original image dimensions.

```python
def scale_boxes_scratch(padded_img_shape, boxes_xyxy, original_img_shape, ratio, pad_wh):
    """
    Rescale bounding boxes from the padded image size back to the original image size.

    Args:
        padded_img_shape (tuple): Shape of the padded image (height, width) used for model input.
        boxes_xyxy (np.array): Array of bounding boxes [[x1, y1, x2, y2], ...]
                               relative to the padded_img_shape.
        original_img_shape (tuple): Shape of the original image (height, width).
        ratio (tuple): Scaling ratios (ratio_w, ratio_h) used during letterboxing.
                       Usually (r, r) where r = min(new_h/orig_h, new_w/orig_w).
        pad_wh (tuple): Padding amounts (pad_w, pad_h) added during letterboxing.
                        These are total padding (dw, dh) from letterbox, not per side.

    Returns:
        np.array: Array of rescaled bounding boxes.
    """
    if not len(boxes_xyxy):
        return np.array([])

    pad_w, pad_h = pad_wh[0], pad_wh[1] # dw, dh from letterbox are total width/height padding
    # We used dw/2 and dh/2 for padding on each side in letterbox. So actual padding on top-left is (pad_w/2, pad_h/2)
    # However, the scaling should be based on the original image content within the padded image.
    
    # Remove padding from coordinates
    # boxes are x1,y1,x2,y2 relative to padded image (e.g. 640x640)
    # The letterbox function returned total dw, dh. Padding on each side is dw/2, dh/2.
    boxes_xyxy[:, 0] -= (pad_w / 2)  # x1
    boxes_xyxy[:, 2] -= (pad_w / 2)  # x2
    boxes_xyxy[:, 1] -= (pad_h / 2)  # y1
    boxes_xyxy[:, 3] -= (pad_h / 2)  # y2

    # Scale coordinates by the inverse of the ratio used during letterboxing
    # ratio was (new_dim / old_dim). We need to multiply by (old_dim / new_dim) = 1/r.
    # ratio_w, ratio_h = ratio
    boxes_xyxy[:, 0] /= ratio[0] # x1
    boxes_xyxy[:, 2] /= ratio[0] # x2
    boxes_xyxy[:, 1] /= ratio[1] # y1
    boxes_xyxy[:, 3] /= ratio[1] # y2

    # Clip boxes to original image dimensions
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, original_img_shape[1])  # x1, clip to width
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, original_img_shape[0])  # y1, clip to height
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, original_img_shape[1])  # x2, clip to width
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, original_img_shape[0])  # y2, clip to height

    return boxes_xyxy.round().astype(np.int32)

# Example Usage:
# Assuming 'filtered_detections' from NMS (list of [x1,y1,x2,y2,conf,cls])
# 'padded_image.shape', 'original_image.shape', 'ratio', 'pad_w', 'pad_h' from preprocessing

# if filtered_detections:
#     nms_boxes_xyxy = np.array([det[:4] for det in filtered_detections])
#     scaled_boxes = scale_boxes_scratch(
#         padded_image.shape[:2], 
#         nms_boxes_xyxy.copy(), # Pass a copy to avoid modifying NMS results directly 
#         original_image.shape[:2],
#         ratio, # ratio = (r_w, r_h) from letterbox
#         (pad_w, pad_h) # pad_wh = (dw, dh) total padding from letterbox
#     )
#     for i, s_box in enumerate(scaled_boxes):
#         conf = filtered_detections[i][4]
#         cls_id = int(filtered_detections[i][5])
#         print(f"Scaled Box: {s_box}, Conf: {conf:.2f}, Class: {cls_id}")
```

### 6.4 Processing Other Tasks (Segmentation, Pose)

*   **Segmentation:** In addition to bounding boxes, segmentation models output masks. These masks are also relative to the padded input size and need to be scaled back to the original image dimensions. The process involves:
    1.  Resizing the raw mask output (often low-resolution, e.g., 160x160) to the padded input image size (e.g., 640x640).
    2.  Applying a threshold to create a binary mask.
    3.  Cropping the mask using the scaled bounding box.
    4.  Resizing the cropped mask to the dimensions of the bounding box in the original image.
    This is more complex than box scaling.
*   **Pose Estimation:** Keypoints (x, y coordinates, and possibly visibility) are outputted. These keypoints are also relative to the padded input image and need to be scaled similarly to bounding box coordinates (remove padding offset, then scale by inverse ratio).

Detailed from-scratch implementations for segmentation and pose postprocessing are extensive and beyond this guide's scope but follow similar principles of coordinate transformation.

---

## 7. Putting It All Together (Conceptual Script)

This conceptual script outlines how the pieces from previous sections would fit together for a complete inference pipeline from scratch using ONNX Runtime as an example. Error handling and TensorRT-specific memory management (like freeing CUDA buffers) would need to be more robust in a production script.

```python
import cv2
import numpy as np
import onnxruntime # Or tensorrt and pycuda for .engine files

# --- Configuration ---
ONNX_MODEL_PATH = "path/to/your/model.yolov8n.onnx"
IMAGE_PATH = "path/to/your/image.jpg"
MODEL_INPUT_SHAPE = (640, 640) # Expected (height, width) by the model
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
# Names of classes if not embedded in the model or if you want to override
CLASS_NAMES = [f'class_{i}' for i in range(80)] # Example for COCO

# --- Helper Functions (from previous sections) ---

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    else:
        dw /= 2
        dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw*2, dh*2) # Return total padding dw, dh

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def non_max_suppression_scratch(predictions, conf_threshold, iou_threshold):
    boxes_cxcywh = predictions[:, :4]
    confidences_obj = predictions[:, 4]
    class_probs = predictions[:, 5:]
    max_class_probs = np.max(class_probs, axis=1)
    class_ids = np.argmax(class_probs, axis=1)
    final_confidences = confidences_obj * max_class_probs
    keep = final_confidences > conf_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    final_confidences = final_confidences[keep]
    class_ids = class_ids[keep]
    if not len(boxes_cxcywh):
        return []
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    final_detections = []
    unique_class_ids = np.unique(class_ids)
    for class_id in unique_class_ids:
        class_mask = (class_ids == class_id)
        class_boxes = boxes_xyxy[class_mask]
        class_confs = final_confidences[class_mask]
        if not len(class_boxes):
            continue
        sorted_indices = np.argsort(class_confs)[::-1]
        class_boxes = class_boxes[sorted_indices]
        class_confs = class_confs[sorted_indices]
        current_class_detections = []
        while len(class_boxes) > 0:
            best_box = class_boxes[0]
            best_conf = class_confs[0]
            current_class_detections.append([*best_box, best_conf, class_id])
            class_boxes = class_boxes[1:]
            class_confs = class_confs[1:]
            if not len(class_boxes):
                break
            ious = np.array([calculate_iou(best_box, other_box) for other_box in class_boxes])
            keep_mask = ious <= iou_threshold
            class_boxes = class_boxes[keep_mask]
            class_confs = class_confs[keep_mask]
        final_detections.extend(current_class_detections)
    return final_detections

def scale_boxes_scratch(padded_img_shape, boxes_xyxy, original_img_shape, ratio_xy, pad_wh_total):
    if not len(boxes_xyxy):
        return np.array([])
    pad_w_total, pad_h_total = pad_wh_total
    # Remove padding (coordinates are relative to top-left of padded image)
    # Padding was added (dw/2, dh/2) to each side. Here pad_w_total = dw, pad_h_total = dh.
    boxes_xyxy[:, 0] -= (pad_w_total / 2)
    boxes_xyxy[:, 2] -= (pad_w_total / 2)
    boxes_xyxy[:, 1] -= (pad_h_total / 2)
    boxes_xyxy[:, 3] -= (pad_h_total / 2)
    # Scale to original image size
    boxes_xyxy[:, 0] /= ratio_xy[0] # ratio_w
    boxes_xyxy[:, 2] /= ratio_xy[0] # ratio_w
    boxes_xyxy[:, 1] /= ratio_xy[1] # ratio_h
    boxes_xyxy[:, 3] /= ratio_xy[1] # ratio_h
    # Clip to original image dimensions
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, original_img_shape[1])
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, original_img_shape[0])
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, original_img_shape[1])
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, original_img_shape[0])
    return boxes_xyxy.round().astype(np.int32)

# --- Main Inference Pipeline ---
def main():
    # 1. Load Model (ONNX Example)
    print("Loading ONNX model...")
    try:
        session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        input_details = session.get_inputs()[0]
        input_name = input_details.name
        # Model expected input shape (e.g., 1, 3, 640, 640), use this if MODEL_INPUT_SHAPE is not reliable
        model_expected_h = input_details.shape[2] 
        model_expected_w = input_details.shape[3]
        current_model_input_shape = (model_expected_h, model_expected_w)
        output_names = [output.name for output in session.get_outputs()]
        print(f"Model loaded. Input: {input_name}{input_details.shape}, Outputs: {output_names}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # 2. Load and Preprocess Image
    print(f"Loading image: {IMAGE_PATH}...")
    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None:
        print(f"Error: Image not found at {IMAGE_PATH}")
        return
    original_h, original_w = original_image.shape[:2]

    print(f"Preprocessing for model input shape {current_model_input_shape}...")
    padded_image, letterbox_ratio_xy, letterbox_pad_wh_total = letterbox(original_image.copy(), new_shape=current_model_input_shape)
    
    img_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_normalized = img_chw.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(np.ascontiguousarray(img_normalized), axis=0)
    print(f"Input tensor ready with shape: {input_tensor.shape}")

    # 3. Perform Inference
    print("Running inference...")
    try:
        raw_outputs = session.run(output_names, {input_name: input_tensor})
        # Assuming first output is the main detection tensor
        # Shape might be (1, num_proposals, 4_coords + 1_conf + num_classes) e.g., (1, 8400, 85)
        raw_prediction_data = raw_outputs[0] 
        print(f"Raw prediction output shape: {raw_prediction_data.shape}")
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # 4. Postprocess
    print("Postprocessing...")
    # Assuming raw_prediction_data is (batch_size, num_proposals, attributes)
    # Taking the first batch element for NMS
    detections_after_nms = non_max_suppression_scratch(raw_prediction_data[0], CONF_THRESHOLD, IOU_THRESHOLD)

    if not detections_after_nms:
        print("No detections found after NMS.")
    else:
        print(f"Found {len(detections_after_nms)} detections after NMS.")
        # Extract boxes for scaling
        boxes_to_scale = np.array([det[:4] for det in detections_after_nms])
        
        scaled_boxes_xyxy = scale_boxes_scratch(
            padded_image.shape[:2], 
            boxes_to_scale.copy(), 
            (original_h, original_w),
            letterbox_ratio_xy, 
            letterbox_pad_wh_total
        )

        print("Final Detections (scaled to original image):")
        for i, scaled_box in enumerate(scaled_boxes_xyxy):
            conf = detections_after_nms[i][4]
            class_id = int(detections_after_nms[i][5])
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"ClassID_{class_id}"
            print(f"  Box: {scaled_box.tolist()}, Conf: {conf:.2f}, Class: {class_name} (ID: {class_id})")
            
            # Draw on original image for visualization
            x1, y1, x2, y2 = scaled_box
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_image, f"{class_name} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # cv2.imshow("Detections", original_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite("output_manual_inference.jpg", original_image)

if __name__ == "__main__":
    main()

```

# --- Conceptual Script for PaddlePaddle Inference (using paddle.inference) ---

import cv2
import numpy as np
import paddle.inference as paddle_inf

# --- Configuration ---
PADDLE_MODEL_FILE = "path/to/your/inference.pdmodel"
PADDLE_PARAMS_FILE = "path/to/your/inference.pdiparams"
IMAGE_PATH = "path/to/your/image.jpg"
MODEL_INPUT_SHAPE = (640, 640) # Expected (height, width) by the model
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
# Number of classes your model was trained on (excluding background)
# For COCO, this is typically 80 for YOLO models.
NUM_CLASSES = 80

def preprocess_image_paddle(image_path, input_shape):
    img = cv2.imread(image_path)
    original_shape = img.shape[:2] # (height, width)
    
    target_h, target_w = input_shape
    
    # Calculate ratio and padding (letterbox/aspect-ratio preserving resize)
    ratio_h = target_h / original_shape[0]
    ratio_w = target_w / original_shape[1]
    ratio = min(ratio_h, ratio_w)
    
    new_unpad_h, new_unpad_w = int(round(original_shape[0] * ratio)), int(round(original_shape[1] * ratio))
    resized_img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    
    pad_h = (target_h - new_unpad_h) // 2
    pad_w = (target_w - new_unpad_w) // 2
    
    # Create a new image with padding
    padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8) # Common padding color
    padded_img[pad_h:pad_h + new_unpad_h, pad_w:pad_w + new_unpad_w] = resized_img
    
    # Normalize (0-1) and transpose (HWC to CHW) BGR to RGB if needed by model
    # PaddlePaddle models often expect BGR, HWC, and normalization specific to their training.
    # Assuming the model expects BGR, CHW, and normalized to [0,1]
    # This step is HIGHLY model-dependent. Check your model's requirements.
    input_image = padded_img.astype(np.float32) / 255.0
    input_image = input_image.transpose((2, 0, 1)) # HWC to CHW
    input_tensor = np.expand_dims(input_image, axis=0) # Add batch dimension (1, C, H, W)
    
    return input_tensor.astype(np.float32), original_shape, ratio, (pad_w, pad_h)

def postprocess_paddle_detections(raw_output, original_image_shape, input_tensor_shape, ratio, pad, conf_thresh, iou_thresh, num_classes):
    # This function needs to be adapted based on the *exact* output format of your PaddlePaddle YOLO model.
    # Assuming raw_output is a NumPy array from predictor.get_output_handle().copy_to_cpu()
    # And a common YOLO output format (e.g., [batch_size, num_proposals, 4_coords + 1_obj_score + num_class_scores])
    
    # Example: if raw_output shape is (1, N, 5 + num_classes)
    # where N is number of proposals (e.g. 8400 for 640x640 input)
    # Coords are typically [cx, cy, w, h] or [x1, y1, x2, y2]
    
    predictions = raw_output[0] # Assuming batch size 1
    
    # Filter by confidence
    obj_scores = predictions[:, 4] # Objectness score column index
    class_scores_all = predictions[:, 5:] # Class scores start from column 5
    
    boxes = []
    scores = []
    class_ids = []

    for i in range(predictions.shape[0]):
        proposal_obj_score = obj_scores[i]
        proposal_class_scores = class_scores_all[i]
        
        if proposal_obj_score < conf_thresh:
            continue
            
        class_id = np.argmax(proposal_class_scores)
        confidence = proposal_obj_score * proposal_class_scores[class_id]
        
        if confidence < conf_thresh:
            continue
            
        # Extract box coordinates (cx, cy, w, h) - assuming this format
        cx, cy, w, h = predictions[i, 0:4]
        x1 = (cx - w / 2)
        y1 = (cy - h / 2)
        x2 = (cx + w / 2)
        y2 = (cy + h / 2)
        
        # Scale back to original image coordinates (before letterboxing/padding)
        # Remove padding
        x1 = (x1 - pad[0]) / ratio
        y1 = (y1 - pad[1]) / ratio
        x2 = (x2 - pad[0]) / ratio
        y2 = (y2 - pad[1]) / ratio
        
        # Clip to original image dimensions
        x1 = np.clip(x1, 0, original_image_shape[1])
        y1 = np.clip(y1, 0, original_image_shape[0])
        x2 = np.clip(x2, 0, original_image_shape[1])
        y2 = np.clip(y2, 0, original_image_shape[0])
        
        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return [], [], []

    # Apply Non-Maximum Suppression (NMS)
    # OpenCV's NMSBoxes expects boxes as (x, y, w, h) for cv2.dnn.NMSBoxes
    # Or (x1,y1,x2,y2) for tf.image.non_max_suppression which is more common with pure numpy approaches
    # For simplicity, let's use a basic NMS idea here (a proper one is more involved)
    # This is a placeholder for a real NMS implementation.
    # You'd typically use a library function or implement NMS from scratch.
    # Example using a simplified approach (not full NMS):
    # For this conceptual script, we assume a library like torchvision.ops.nms or similar would be used
    # or a from-scratch implementation. We'll just return the filtered boxes for now.
    # A full NMS implementation is beyond this conceptual script snippet.
    # For a real NMS, you'd use something like:
    # indices = cv2.dnn.NMSBoxes([ [b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes], scores, conf_thresh, iou_thresh)
    # if len(indices) > 0:
    #     selected_boxes = [boxes[i] for i in indices.flatten()]
    #     selected_scores = [scores[i] for i in indices.flatten()]
    #     selected_class_ids = [class_ids[i] for i in indices.flatten()]
    # else: 
    #     selected_boxes, selected_scores, selected_class_ids = [], [], []
    # return selected_boxes, selected_scores, selected_class_ids
    
    # Placeholder: returning boxes before NMS for simplicity of this conceptual script
    # In a real scenario, implement or call NMS here.
    # For demonstration, let's assume NMS is applied and these are the results.
    # This part requires a proper NMS function.
    # For now, we return all boxes that passed confidence threshold.
    final_boxes = [[int(coord) for coord in box] for box in boxes]
    return final_boxes, scores, class_ids

def main_paddle_inference():
    # 1. Load Model
    config = paddle_inf.Config(PADDLE_MODEL_FILE, PADDLE_PARAMS_FILE)
    # config.enable_use_gpu(100, 0) # If using GPU
    predictor = paddle_inf.create_predictor(config)
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_handle = predictor.get_output_handle(output_names[0])

    # 2. Preprocess Image
    input_tensor, original_shape, ratio, pad = preprocess_image_paddle(IMAGE_PATH, MODEL_INPUT_SHAPE)

    # 3. Perform Inference
    input_handle.reshape(input_tensor.shape) # Important for dynamic shapes
    input_handle.copy_from_cpu(input_tensor)
    predictor.run()
    raw_output = output_handle.copy_to_cpu()

    # 4. Postprocess results
    boxes, scores, class_ids = postprocess_paddle_detections(
        raw_output, 
        original_shape, 
        MODEL_INPUT_SHAPE, 
        ratio, 
        pad, 
        CONF_THRESHOLD, 
        IOU_THRESHOLD, 
        NUM_CLASSES
    )

    # 5. Visualize or use results
    print(f"Detected {len(boxes)} objects:")
    for i, box in enumerate(boxes):
        print(f"  Box: {box}, Class ID: {class_ids[i]}, Score: {scores[i]:.2f}")
        # You can draw these boxes on the original image using cv2.rectangle and cv2.putText

    # Example visualization (optional)
    # img_display = cv2.imread(IMAGE_PATH)
    # for i, box in enumerate(boxes):
    #     x1, y1, x2, y2 = box
    #     cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(img_display, f"Cls {class_ids[i]}: {scores[i]:.2f}", (x1, y1 - 10), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imshow("Detections - PaddlePaddle", img_display)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # main_onnx_inference() # Uncomment to run ONNX example
    # main_tensorrt_inference() # Uncomment to run TensorRT example
    # main_pytorch_inference() # Uncomment to run PyTorch example
    # main_openvino_inference() # Uncomment to run OpenVINO example
    main_paddle_inference() # Uncomment to run PaddlePaddle example

## 8. Important Considerations and Limitations

*   **Model Specifics are Key:** This entire pipeline hinges on your exact knowledge of the model's input requirements (shape, normalization, color order) and output format (tensor structure, coordinate system, confidence/class representation). **There is no universal solution.**
*   **Efficiency:** The Python/NumPy implementations for NMS and other utilities are for clarity, not speed. For production, consider:
    *   Vectorizing operations more heavily with NumPy.
    *   Using C++/CUDA extensions for critical parts.
    *   For TensorRT, ensure your engine is optimized. Some postprocessing (like NMS) can sometimes be included in the TensorRT graph itself during conversion for maximum performance.
*   **TensorRT Complexity:** Direct TensorRT API usage (Section 3.2 and 5.2) is significantly more complex due to manual memory management and interaction with CUDA. Errors can be cryptic. Thoroughly test each step.
*   **Output Parsing Variations:** YOLO models can have different output layer structures, especially across versions (v3, v4, v5, v7, v8, etc.) or if custom-exported. The example assumed a common format (e.g., `[batch, num_proposals, cxcywh+conf+classes]`). Adapt `non_max_suppression_scratch` and output interpretation accordingly.
*   **Coordinate Systems:** Be meticulous about coordinate systems (`xywh` vs. `xyxy`, normalized vs. absolute) at each step.
*   **Batch Processing:** The conceptual script processes a single image. For batch processing, you would batch images in `input_tensor` and loop through the batch dimension in the `raw_outputs` during postprocessing.
*   **Error Handling:** Production code needs more robust error handling and resource management (e.g., freeing CUDA memory in `finally` blocks for TensorRT).
*   **Class Names:** Ensure you have the correct mapping of class IDs to human-readable names, matching what the model was trained on.
*   **Other Tasks (Segmentation/Pose):** Postprocessing for these tasks is significantly more involved than object detection and requires custom logic for mask manipulation or keypoint handling, including their own scaling and NMS-like filtering if applicable.

This from-scratch approach offers maximum control but also demands a deep understanding of both the model and the inference runtimes. 