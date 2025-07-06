python scripts/benchmark_models.py \
       --half          # enable FP16
[BENCH] YOLOv10n (vehicles) → Vehicles/YOLOv10n/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4354.4±2840.6 MB/s, size: 247.7 KB)
                   all        535       5428      0.873      0.819      0.881       0.64
Speed: 0.2ms preprocess, 5.6ms inference, 0.0ms loss, 0.1ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.8s, saved as 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.torchscript' (9.1 MB)

Export complete (1.0s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 5975.9±2500.2 MB/s, size: 298.3 KB)
                   all        535       5428      0.859      0.823      0.878       0.63
Speed: 0.2ms preprocess, 2.4ms inference, 0.0ms loss, 0.1ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.9s, saved as 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.onnx' (4.5 MB)

Export complete (1.0s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4317.8±2847.2 MB/s, size: 131.7 KB)
                   all        535       5428      0.861       0.82      0.879       0.63
Speed: 0.2ms preprocess, 4.8ms inference, 0.0ms loss, 0.2ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.9s, saved as 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.onnx' (8.9 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:05:50] [TRT] [I] [MemUsageChange] Init CUDA: CPU -2, GPU +0, now: CPU 2102, GPU 797 (MiB)
[07/06/2025-14:05:51] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1563, GPU +6, now: CPU 3743, GPU 803 (MiB)
[07/06/2025-14:05:51] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:05:51] [TRT] [I] Input filename:   Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.onnx
[07/06/2025-14:05:51] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:05:51] [TRT] [I] Opset version:    19
[07/06/2025-14:05:51] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:05:51] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:05:51] [TRT] [I] Domain:           
[07/06/2025-14:05:51] [TRT] [I] Model version:    0
[07/06/2025-14:05:51] [TRT] [I] Doc string:       
[07/06/2025-14:05:51] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 300, 6) DataType.FLOAT
TensorRT: building FP16 engine as Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.engine
[07/06/2025-14:05:51] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:07:14] [TRT] [I] Compiler backend is used during engine build.
[07/06/2025-14:08:31] [TRT] [I] Detected 1 inputs and 1 output network tensors.
[07/06/2025-14:08:32] [TRT] [I] Total Host Persistent Memory: 503248 bytes
[07/06/2025-14:08:32] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:08:32] [TRT] [I] Max Scratch Memory: 1382400 bytes
[07/06/2025-14:08:32] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 156 steps to complete.
[07/06/2025-14:08:32] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 6.84674ms to assign 11 blocks to 156 nodes requiring 9524224 bytes.
[07/06/2025-14:08:32] [TRT] [I] Total Activation Memory: 9523200 bytes
[07/06/2025-14:08:32] [TRT] [I] Total Weights Memory: 4616736 bytes
[07/06/2025-14:08:32] [TRT] [I] Compiler backend is used during engine execution.
[07/06/2025-14:08:32] [TRT] [I] Engine generation completed in 161.128 seconds.
[07/06/2025-14:08:32] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 138 MiB
TensorRT: export success ✅ 163.9s, saved as 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.engine' (8.1 MB)

Export complete (163.9s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:08:33] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:08:33] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 13 (MiB)
Loading Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:08:33] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:08:33] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:08:33] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 1, GPU 26 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2209.7±989.5 MB/s, size: 217.8 KB)
                   all        535       5428      0.858      0.823      0.879       0.63
Speed: 0.3ms preprocess, 1.3ms inference, 0.0ms loss, 0.2ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: End-to-end models not supported by PaddlePaddle yet

PyTorch: starting from 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 1.3s, saved as 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.onnx' (8.9 MB)
CPU Group: [ 11  8  6  4  2  0  9  10  7  5  3  1 ], 800000 - 4400000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
CPU Group: [ 11  8  6  4  2  0  9  10  7  5  3  1 ], 800000 - 4400000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
try 'pip install -U aliyun-log-python-sdk'
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.
    
    If you wish to install a non-Debian-packaged Python package,
    create a virtual environment using python3 -m venv path/to/venv.
    Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
    sure you have python3-full installed.
    
    If you wish to install a non-Debian packaged Python application,
    it may be easiest to use pipx install xyz, which will manage a
    virtual environment for you. Make sure you have pipx installed.
    
    See /usr/share/doc/python3.12/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
try 'pip install -U aliyun-log-python-sdk'
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.
    
    If you wish to install a non-Debian-packaged Python package,
    create a virtual environment using python3 -m venv path/to/venv.
    Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
    sure you have python3-full installed.
    
    If you wish to install a non-Debian packaged Python application,
    it may be easiest to use pipx install xyz, which will manage a
    virtual environment for you. Make sure you have pipx installed.
    
    See /usr/share/doc/python3.12/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:08:37] :46: ONNX Model ir version: 9
[14:08:37] :47: ONNX Model opset version: 19
[14:08:37] :148: Check it out ==> /model.11/Resize_output_0 has empty input, the index is 1
[14:08:37] :148: Check it out ==> /model.14/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 2.0s, saved as 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.mnn' (4.4 MB)

Export complete (2.0s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1871.5±896.7 MB/s, size: 219.9 KB)
                   all        535       5428      0.861      0.821      0.878       0.63
Speed: 1.0ms preprocess, 28.1ms inference, 0.0ms loss, 0.1ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.8s, saved as 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.torchscript' (9.1 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.torchscript ncnnparam=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.Attention
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2f
inline module = ultralytics.nn.modules.block.C2fCIB
inline module = ultralytics.nn.modules.block.CIB
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.PSA
inline module = ultralytics.nn.modules.block.RepVGGDW
inline module = ultralytics.nn.modules.block.SCDown
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.v10Detect
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.Attention
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2f
inline module = ultralytics.nn.modules.block.C2fCIB
inline module = ultralytics.nn.modules.block.CIB
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.PSA
inline module = ultralytics.nn.modules.block.RepVGGDW
inline module = ultralytics.nn.modules.block.SCDown
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.v10Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
BinaryOp floor_divide not supported yet
BinaryOp remainder not supported yet
ignore torch.topk torch.topk_10 param dim=-1
ignore torch.topk torch.topk_10 param k=300
ignore torch.topk torch.topk_10 param largest=True
ignore torch.topk torch.topk_10 param sorted=True
ignore torch.gather torch.gather_68 param dim=1
ignore torch.gather torch.gather_69 param dim=1
ignore torch.topk torch.topk_11 param dim=-1
ignore torch.topk torch.topk_11 param k=300
ignore torch.topk torch.topk_11 param largest=True
ignore torch.topk torch.topk_11 param sorted=True
ignore pnnx.Expression pnnx_expr_3 param expr=[@0,floor_divide(@1,6)]
ignore Tensor.to Tensor.to_16 param copy=False
ignore Tensor.to Tensor.to_16 param dtype=torch.float
NCNN: export success ✅ 1.2s, saved as 'Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model' (4.5 MB)

Export complete (2.1s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv10n/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
ERROR ❌ Benchmark failure for NCNN: End-to-end torch.topk operation is not supported for NCNN prediction yet
ERROR ❌ Benchmark failure for IMX: 
ERROR ❌ Benchmark failure for RKNN: End-to-end models not supported by RKNN yet
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/Vehicles/YOLOv10n/runs/detect/train/weights/best.pt on Results/datasets/vehicles/data.yaml at imgsz=640 (205.97s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.5              0.6401                   5.64  177.24
1             TorchScript       ✅        9.1              0.6301                   2.44  409.49
2                    ONNX       ✅        4.5              0.6301                   4.76  210.09
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        8.1              0.6302                   1.26  792.31
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        4.4              0.6302                  28.11   35.57
13                   NCNN       ❎        4.5                   -                      -       -
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to Vehicles/YOLOv10n/benchmark_vehicles.log
      ↳ saved Vehicles/YOLOv10n/benchmark_vehicles.json
[BENCH] YOLOv5u (vehicles) → Vehicles/YOLOv5u/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2376.0±1015.5 MB/s, size: 252.2 KB)
                   all        535       5428      0.879      0.819      0.883      0.632
Speed: 0.2ms preprocess, 4.3ms inference, 0.0ms loss, 0.8ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.0 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.7s, saved as 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.torchscript' (10.0 MB)

Export complete (0.7s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2139.7±651.5 MB/s, size: 261.3 KB)
                   all        535       5428      0.856      0.834       0.88      0.619
Speed: 0.2ms preprocess, 2.1ms inference, 0.0ms loss, 0.8ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.5s, saved as 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.onnx' (4.9 MB)

Export complete (0.5s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2179.2±785.0 MB/s, size: 319.0 KB)
                   all        535       5428       0.86      0.832       0.88      0.619
Speed: 0.2ms preprocess, 3.5ms inference, 0.0ms loss, 0.8ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.5s, saved as 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.onnx' (9.8 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:09:14] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1537, GPU +0, now: CPU 5987, GPU 843 (MiB)
[07/06/2025-14:09:14] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:09:14] [TRT] [I] Input filename:   Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.onnx
[07/06/2025-14:09:14] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:09:14] [TRT] [I] Opset version:    19
[07/06/2025-14:09:14] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:09:14] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:09:14] [TRT] [I] Domain:           
[07/06/2025-14:09:14] [TRT] [I] Model version:    0
[07/06/2025-14:09:14] [TRT] [I] Doc string:       
[07/06/2025-14:09:14] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 10, 8400) DataType.FLOAT
TensorRT: building FP16 engine as Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.engine
[07/06/2025-14:09:14] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:11:33] [TRT] [I] Detected 1 inputs and 3 output network tensors.
[07/06/2025-14:11:34] [TRT] [I] Total Host Persistent Memory: 428688 bytes
[07/06/2025-14:11:34] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:11:34] [TRT] [I] Max Scratch Memory: 0 bytes
[07/06/2025-14:11:34] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 128 steps to complete.
[07/06/2025-14:11:34] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 4.39756ms to assign 8 blocks to 128 nodes requiring 8756224 bytes.
[07/06/2025-14:11:34] [TRT] [I] Total Activation Memory: 8755200 bytes
[07/06/2025-14:11:34] [TRT] [I] Total Weights Memory: 5136640 bytes
[07/06/2025-14:11:34] [TRT] [I] Engine generation completed in 139.798 seconds.
[07/06/2025-14:11:34] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 1 MiB, GPU 138 MiB
TensorRT: export success ✅ 141.8s, saved as 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.engine' (8.6 MB)

Export complete (141.8s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:11:34] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:11:34] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 13 (MiB)
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:11:34] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:11:34] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:11:34] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +8, now: CPU 1, GPU 26 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4064.8±1319.2 MB/s, size: 108.4 KB)
                   all        535       5428      0.858      0.833       0.88      0.619
Speed: 0.3ms preprocess, 1.0ms inference, 0.0ms loss, 0.9ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: ERROR ❌️ argument 'half' is not supported for format='paddle'

PyTorch: starting from 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.8s, saved as 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.onnx' (9.8 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:11:38] :46: ONNX Model ir version: 9
[14:11:38] :47: ONNX Model opset version: 19
[14:11:38] :148: Check it out ==> /model.11/Resize_output_0 has empty input, the index is 1
[14:11:38] :148: Check it out ==> /model.15/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 0.9s, saved as 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.mnn' (4.9 MB)

Export complete (0.9s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2373.4±1691.8 MB/s, size: 280.1 KB)
                   all        535       5428      0.856      0.835       0.88      0.619
Speed: 1.0ms preprocess, 26.9ms inference, 0.0ms loss, 0.7ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.0 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.7s, saved as 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.torchscript' (10.0 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.torchscript ncnnparam=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C3
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.Detect
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C3
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
NCNN: export success ✅ 1.2s, saved as 'Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model' (5.0 MB)

Export complete (1.9s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Loading Results/Vehicles/YOLOv5u/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2495.9±1000.3 MB/s, size: 129.1 KB)
                   all        535       5428      0.856      0.834       0.88      0.619
Speed: 1.0ms preprocess, 43.1ms inference, 0.0ms loss, 0.7ms postprocess per image
ERROR ❌ Benchmark failure for IMX: IMX only supported for YOLOv8
ERROR ❌ Benchmark failure for RKNN: inference not supported on GPU
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/Vehicles/YOLOv5u/runs/detect/train/weights/best.pt on Results/datasets/vehicles/data.yaml at imgsz=640 (208.43s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.0              0.6319                   4.26  234.53
1             TorchScript       ✅       10.0              0.6188                   2.09   479.1
2                    ONNX       ✅        4.9              0.6187                   3.45  289.56
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        8.6              0.6186                    1.0  999.85
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        4.9              0.6187                  26.93   37.13
13                   NCNN       ✅        5.0              0.6189                  43.06   23.22
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to Vehicles/YOLOv5u/benchmark_vehicles.log
      ↳ saved Vehicles/YOLOv5u/benchmark_vehicles.json
[BENCH] YOLOv8n (vehicles) → Vehicles/YOLOv8n/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3040.6±1557.5 MB/s, size: 259.4 KB)
                   all        535       5428      0.879      0.818      0.891       0.64
Speed: 0.2ms preprocess, 3.3ms inference, 0.0ms loss, 0.7ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (6.0 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.6s, saved as 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.torchscript' (11.9 MB)

Export complete (0.6s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2083.6±875.8 MB/s, size: 236.7 KB)
                   all        535       5428      0.858      0.836      0.889      0.632
Speed: 0.2ms preprocess, 2.1ms inference, 0.0ms loss, 0.8ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (6.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.6s, saved as 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.onnx' (5.9 MB)

Export complete (0.6s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2137.5±978.4 MB/s, size: 186.6 KB)
                   all        535       5428      0.857      0.837      0.889      0.632
Speed: 0.2ms preprocess, 4.6ms inference, 0.0ms loss, 1.0ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (6.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.5s, saved as 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.onnx' (11.7 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:12:43] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1537, GPU +0, now: CPU 5927, GPU 891 (MiB)
[07/06/2025-14:12:43] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:12:43] [TRT] [I] Input filename:   Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.onnx
[07/06/2025-14:12:43] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:12:43] [TRT] [I] Opset version:    19
[07/06/2025-14:12:43] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:12:43] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:12:43] [TRT] [I] Domain:           
[07/06/2025-14:12:43] [TRT] [I] Model version:    0
[07/06/2025-14:12:43] [TRT] [I] Doc string:       
[07/06/2025-14:12:43] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 10, 8400) DataType.FLOAT
TensorRT: building FP16 engine as Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.engine
[07/06/2025-14:12:43] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:14:37] [TRT] [I] Detected 1 inputs and 3 output network tensors.
[07/06/2025-14:14:37] [TRT] [I] Total Host Persistent Memory: 386720 bytes
[07/06/2025-14:14:37] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:14:37] [TRT] [I] Max Scratch Memory: 0 bytes
[07/06/2025-14:14:37] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 112 steps to complete.
[07/06/2025-14:14:37] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 3.74368ms to assign 7 blocks to 112 nodes requiring 9062912 bytes.
[07/06/2025-14:14:37] [TRT] [I] Total Activation Memory: 9062400 bytes
[07/06/2025-14:14:37] [TRT] [I] Total Weights Memory: 6089536 bytes
[07/06/2025-14:14:37] [TRT] [I] Engine generation completed in 114.247 seconds.
[07/06/2025-14:14:37] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 1 MiB, GPU 138 MiB
TensorRT: export success ✅ 116.2s, saved as 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.engine' (9.0 MB)

Export complete (116.2s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:14:37] [TRT] [I] Loaded engine size: 9 MiB
[07/06/2025-14:14:37] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 14 (MiB)
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:14:37] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:14:37] [TRT] [I] Loaded engine size: 9 MiB
[07/06/2025-14:14:37] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +8, now: CPU 1, GPU 28 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4267.5±2314.8 MB/s, size: 279.3 KB)
                   all        535       5428      0.856      0.839      0.889      0.632
Speed: 0.3ms preprocess, 1.1ms inference, 0.0ms loss, 0.9ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: ERROR ❌️ argument 'half' is not supported for format='paddle'

PyTorch: starting from 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (6.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.6s, saved as 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.onnx' (11.7 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:14:41] :46: ONNX Model ir version: 9
[14:14:41] :47: ONNX Model opset version: 19
[14:14:41] :148: Check it out ==> /model.10/Resize_output_0 has empty input, the index is 1
[14:14:41] :148: Check it out ==> /model.13/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 0.7s, saved as 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.mnn' (5.8 MB)

Export complete (0.7s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2101.5±172.1 MB/s, size: 342.6 KB)
                   all        535       5428      0.858      0.836      0.889      0.631
Speed: 1.0ms preprocess, 29.1ms inference, 0.0ms loss, 0.6ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (6.0 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.6s, saved as 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.torchscript' (11.9 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.torchscript ncnnparam=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2f
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.Detect
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2f
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
NCNN: export success ✅ 1.2s, saved as 'Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model' (5.9 MB)

Export complete (1.8s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Loading Results/Vehicles/YOLOv8n/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3978.7±2705.6 MB/s, size: 246.2 KB)
                   all        535       5428      0.858      0.836      0.889      0.631
Speed: 1.0ms preprocess, 39.2ms inference, 0.0ms loss, 0.6ms postprocess per image
ERROR ❌ Benchmark failure for IMX: ERROR ❌️ argument 'half' is not supported for format='imx'
ERROR ❌ Benchmark failure for RKNN: inference not supported on GPU
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/Vehicles/YOLOv8n/runs/detect/train/weights/best.pt on Results/datasets/vehicles/data.yaml at imgsz=640 (181.65s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        6.0              0.6398                   3.27  305.84
1             TorchScript       ✅       11.9              0.6315                    2.1   476.0
2                    ONNX       ✅        5.9              0.6321                   4.65  215.21
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        9.0              0.6323                    1.1  904.93
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        5.8              0.6311                  29.11   34.35
13                   NCNN       ✅        5.9              0.6313                  39.16   25.54
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to Vehicles/YOLOv8n/benchmark_vehicles.log
      ↳ saved Vehicles/YOLOv8n/benchmark_vehicles.json
[BENCH] GhostYOLO (vehicles) → Vehicles/GhostYOLO/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3099.1±2178.7 MB/s, size: 267.9 KB)
                   all        535       5428      0.865      0.785      0.857      0.598
Speed: 0.2ms preprocess, 6.8ms inference, 0.0ms loss, 0.7ms postprocess per image

PyTorch: starting from 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (3.7 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 1.4s, saved as 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.torchscript' (7.5 MB)

Export complete (1.4s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1673.5±223.6 MB/s, size: 237.9 KB)
                   all        535       5428      0.857      0.796      0.857      0.591
Speed: 0.2ms preprocess, 2.9ms inference, 0.0ms loss, 0.7ms postprocess per image

PyTorch: starting from 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (3.7 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.9s, saved as 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.onnx' (3.5 MB)

Export complete (0.9s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2260.0±1475.0 MB/s, size: 250.2 KB)
                   all        535       5428      0.858      0.796      0.857       0.59
Speed: 0.2ms preprocess, 4.9ms inference, 0.0ms loss, 1.0ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (3.7 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.8s, saved as 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.onnx' (6.9 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:15:50] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1537, GPU +0, now: CPU 5746, GPU 891 (MiB)
[07/06/2025-14:15:50] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:15:50] [TRT] [I] Input filename:   Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.onnx
[07/06/2025-14:15:50] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:15:50] [TRT] [I] Opset version:    19
[07/06/2025-14:15:50] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:15:50] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:15:50] [TRT] [I] Domain:           
[07/06/2025-14:15:50] [TRT] [I] Model version:    0
[07/06/2025-14:15:50] [TRT] [I] Doc string:       
[07/06/2025-14:15:50] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 10, 8400) DataType.FLOAT
TensorRT: building FP16 engine as Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.engine
[07/06/2025-14:15:50] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:18:26] [TRT] [I] Detected 1 inputs and 3 output network tensors.
[07/06/2025-14:18:27] [TRT] [I] Total Host Persistent Memory: 695616 bytes
[07/06/2025-14:18:27] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:18:27] [TRT] [I] Max Scratch Memory: 0 bytes
[07/06/2025-14:18:27] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 239 steps to complete.
[07/06/2025-14:18:27] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 12.0919ms to assign 10 blocks to 239 nodes requiring 9422336 bytes.
[07/06/2025-14:18:27] [TRT] [I] Total Activation Memory: 9420800 bytes
[07/06/2025-14:18:27] [TRT] [I] Total Weights Memory: 3618560 bytes
[07/06/2025-14:18:27] [TRT] [I] Engine generation completed in 156.814 seconds.
[07/06/2025-14:18:27] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 1 MiB, GPU 138 MiB
TensorRT: export success ✅ 159.0s, saved as 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.engine' (7.8 MB)

Export complete (159.1s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:18:27] [TRT] [I] Loaded engine size: 7 MiB
[07/06/2025-14:18:27] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +9, now: CPU 1, GPU 12 (MiB)
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:18:27] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:18:27] [TRT] [I] Loaded engine size: 7 MiB
[07/06/2025-14:18:27] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +9, now: CPU 2, GPU 24 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3743.5±1404.4 MB/s, size: 233.5 KB)
                   all        535       5428      0.857      0.795      0.857       0.59
Speed: 0.3ms preprocess, 1.4ms inference, 0.0ms loss, 0.9ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: ERROR ❌️ argument 'half' is not supported for format='paddle'

PyTorch: starting from 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (3.7 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 1.1s, saved as 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.onnx' (6.9 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:18:31] :46: ONNX Model ir version: 9
[14:18:31] :47: ONNX Model opset version: 19
[14:18:31] :148: Check it out ==> /model.10/Resize_output_0 has empty input, the index is 1
[14:18:31] :148: Check it out ==> /model.13/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 1.2s, saved as 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.mnn' (3.5 MB)

Export complete (1.3s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1535.9±380.5 MB/s, size: 342.1 KB)
                   all        535       5428      0.856      0.796      0.857      0.591
Speed: 1.0ms preprocess, 25.2ms inference, 0.0ms loss, 0.7ms postprocess per image

PyTorch: starting from 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (3.7 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 1.4s, saved as 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.torchscript' (7.5 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.torchscript ncnnparam=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.C3Ghost
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.GhostBottleneck
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.conv.GhostConv
inline module = ultralytics.nn.modules.head.Detect
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.C3Ghost
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.GhostBottleneck
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.conv.GhostConv
inline module = ultralytics.nn.modules.head.Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
NCNN: export success ✅ 1.4s, saved as 'Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model' (3.5 MB)

Export complete (2.8s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Loading Results/Vehicles/GhostYOLO/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2895.0±2277.3 MB/s, size: 239.9 KB)
                   all        535       5428      0.856      0.795      0.857      0.591
Speed: 1.0ms preprocess, 39.4ms inference, 0.0ms loss, 0.6ms postprocess per image
ERROR ❌ Benchmark failure for IMX: IMX only supported for YOLOv8
ERROR ❌ Benchmark failure for RKNN: inference not supported on GPU
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/Vehicles/GhostYOLO/runs/detect/train/weights/best.pt on Results/datasets/vehicles/data.yaml at imgsz=640 (229.27s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        3.7              0.5979                   6.78   147.4
1             TorchScript       ✅        7.5              0.5913                   2.93  340.97
2                    ONNX       ✅        3.5              0.5901                   4.88  204.88
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        7.8              0.5899                   1.36  735.16
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        3.5              0.5907                  25.23   39.63
13                   NCNN       ✅        3.5              0.5909                  39.43   25.36
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to Vehicles/GhostYOLO/benchmark_vehicles.log
      ↳ saved Vehicles/GhostYOLO/benchmark_vehicles.json
[BENCH] YOLO11n (vehicles) → Vehicles/YOLO11n/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3729.0±1769.6 MB/s, size: 367.0 KB)
                   all        535       5428      0.865      0.829       0.88      0.639
Speed: 0.2ms preprocess, 5.0ms inference, 0.0ms loss, 0.7ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.2 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 1.0s, saved as 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.torchscript' (10.4 MB)

Export complete (1.0s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2705.8±606.8 MB/s, size: 261.6 KB)
                   all        535       5428      0.863       0.82      0.882       0.63
Speed: 0.2ms preprocess, 2.6ms inference, 0.0ms loss, 0.8ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.2 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.6s, saved as 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.onnx' (5.1 MB)

Export complete (0.6s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4223.8±2781.4 MB/s, size: 153.0 KB)
                   all        535       5428      0.844      0.836      0.881      0.628
Speed: 0.2ms preprocess, 4.7ms inference, 0.0ms loss, 1.0ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.2 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.8s, saved as 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.onnx' (10.1 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:19:37] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1538, GPU +0, now: CPU 5775, GPU 795 (MiB)
[07/06/2025-14:19:37] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:19:37] [TRT] [I] Input filename:   Results/Vehicles/YOLO11n/runs/detect/train/weights/best.onnx
[07/06/2025-14:19:37] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:19:37] [TRT] [I] Opset version:    19
[07/06/2025-14:19:37] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:19:37] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:19:37] [TRT] [I] Domain:           
[07/06/2025-14:19:37] [TRT] [I] Model version:    0
[07/06/2025-14:19:37] [TRT] [I] Doc string:       
[07/06/2025-14:19:37] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 10, 8400) DataType.FLOAT
TensorRT: building FP16 engine as Results/Vehicles/YOLO11n/runs/detect/train/weights/best.engine
[07/06/2025-14:19:37] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:20:51] [TRT] [I] Compiler backend is used during engine build.
[07/06/2025-14:22:04] [TRT] [I] Detected 1 inputs and 1 output network tensors.
[07/06/2025-14:22:05] [TRT] [I] Total Host Persistent Memory: 536016 bytes
[07/06/2025-14:22:05] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:22:05] [TRT] [I] Max Scratch Memory: 1382400 bytes
[07/06/2025-14:22:05] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 184 steps to complete.
[07/06/2025-14:22:05] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 9.44302ms to assign 11 blocks to 184 nodes requiring 9524224 bytes.
[07/06/2025-14:22:05] [TRT] [I] Total Activation Memory: 9523200 bytes
[07/06/2025-14:22:05] [TRT] [I] Total Weights Memory: 5252610 bytes
[07/06/2025-14:22:05] [TRT] [I] Compiler backend is used during engine execution.
[07/06/2025-14:22:05] [TRT] [I] Engine generation completed in 148.108 seconds.
[07/06/2025-14:22:05] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 2 MiB, GPU 138 MiB
TensorRT: export success ✅ 150.6s, saved as 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.engine' (8.8 MB)

Export complete (150.6s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:22:05] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:22:05] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 14 (MiB)
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:22:05] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:22:05] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:22:05] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 1, GPU 28 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3779.5±843.4 MB/s, size: 192.8 KB)
                   all        535       5428      0.846      0.835      0.882      0.628
Speed: 0.3ms preprocess, 1.2ms inference, 0.0ms loss, 0.9ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: ERROR ❌️ argument 'half' is not supported for format='paddle'

PyTorch: starting from 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.2 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.7s, saved as 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.onnx' (10.1 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:22:08] :46: ONNX Model ir version: 9
[14:22:08] :47: ONNX Model opset version: 19
[14:22:08] :148: Check it out ==> /model.11/Resize_output_0 has empty input, the index is 1
[14:22:09] :148: Check it out ==> /model.14/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 0.9s, saved as 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.mnn' (5.0 MB)

Export complete (0.9s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2087.4±654.6 MB/s, size: 186.6 KB)
                   all        535       5428      0.844      0.836      0.882       0.63
Speed: 1.0ms preprocess, 26.1ms inference, 0.0ms loss, 0.6ms postprocess per image

PyTorch: starting from 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.2 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.8s, saved as 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best.torchscript' (10.4 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/Vehicles/YOLO11n/runs/detect/train/weights/best.torchscript ncnnparam=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.Attention
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2PSA
inline module = ultralytics.nn.modules.block.C3k
inline module = ultralytics.nn.modules.block.C3k2
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.PSABlock
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.conv.DWConv
inline module = ultralytics.nn.modules.head.Detect
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.Attention
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2PSA
inline module = ultralytics.nn.modules.block.C3k
inline module = ultralytics.nn.modules.block.C3k2
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.PSABlock
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.conv.DWConv
inline module = ultralytics.nn.modules.head.Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
NCNN: export success ✅ 1.2s, saved as 'Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model' (5.1 MB)

Export complete (2.1s)
Results saved to /home/cetech/trafficmonitor/Results/Vehicles/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/traffic-flow-test1-bicycle-vehicles-2/data.yaml half 
Visualize:       https://netron.app
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Loading Results/Vehicles/YOLO11n/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 4636.5±2306.8 MB/s, size: 285.8 KB)
                   all        535       5428      0.845      0.836      0.882       0.63
Speed: 1.0ms preprocess, 42.5ms inference, 0.0ms loss, 0.6ms postprocess per image
ERROR ❌ Benchmark failure for IMX: IMX only supported for YOLOv8
ERROR ❌ Benchmark failure for RKNN: inference not supported on GPU
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/Vehicles/YOLO11n/runs/detect/train/weights/best.pt on Results/datasets/vehicles/data.yaml at imgsz=640 (219.09s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.2              0.6392                   4.97  201.09
1             TorchScript       ✅       10.4              0.6302                   2.65  377.77
2                    ONNX       ✅        5.1              0.6275                   4.72  211.79
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        8.8              0.6281                   1.24  805.26
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        5.0                0.63                  26.13   38.27
13                   NCNN       ✅        5.1                0.63                  42.48   23.54
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to Vehicles/YOLO11n/benchmark_vehicles.log
      ↳ saved Vehicles/YOLO11n/benchmark_vehicles.json
[BENCH] YOLOv10n (plates) → License Plate/YOLOv10n/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1874.2±767.9 MB/s, size: 20.0 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.983      0.942      0.979       0.72
Speed: 0.2ms preprocess, 4.6ms inference, 0.0ms loss, 0.1ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 1.0s, saved as 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.torchscript' (9.1 MB)

Export complete (1.0s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv10n/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/License Plate/YOLOv10n/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1645.0±799.1 MB/s, size: 17.3 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189       0.98      0.946      0.978      0.713
Speed: 0.2ms preprocess, 2.4ms inference, 0.0ms loss, 0.1ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.8s, saved as 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.onnx' (4.5 MB)

Export complete (0.9s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv10n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/License Plate/YOLOv10n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2519.0±141.1 MB/s, size: 30.2 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.981      0.945      0.978      0.713
Speed: 0.2ms preprocess, 4.6ms inference, 0.0ms loss, 0.2ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 1.0s, saved as 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.onnx' (8.9 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:23:38] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1537, GPU +0, now: CPU 6501, GPU 895 (MiB)
[07/06/2025-14:23:38] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:23:38] [TRT] [I] Input filename:   Results/License Plate/YOLOv10n/runs/detect/train/weights/best.onnx
[07/06/2025-14:23:38] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:23:38] [TRT] [I] Opset version:    19
[07/06/2025-14:23:38] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:23:38] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:23:38] [TRT] [I] Domain:           
[07/06/2025-14:23:38] [TRT] [I] Model version:    0
[07/06/2025-14:23:38] [TRT] [I] Doc string:       
[07/06/2025-14:23:38] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 300, 6) DataType.FLOAT
TensorRT: building FP16 engine as Results/License Plate/YOLOv10n/runs/detect/train/weights/best.engine
[07/06/2025-14:23:38] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:24:32] [TRT] [I] Compiler backend is used during engine build.
[07/06/2025-14:25:44] [TRT] [I] Detected 1 inputs and 1 output network tensors.
[07/06/2025-14:25:44] [TRT] [I] Total Host Persistent Memory: 501296 bytes
[07/06/2025-14:25:44] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:25:44] [TRT] [I] Max Scratch Memory: 1382400 bytes
[07/06/2025-14:25:44] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 157 steps to complete.
[07/06/2025-14:25:44] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 6.85978ms to assign 10 blocks to 157 nodes requiring 9523712 bytes.
[07/06/2025-14:25:44] [TRT] [I] Total Activation Memory: 9523200 bytes
[07/06/2025-14:25:44] [TRT] [I] Total Weights Memory: 4616608 bytes
[07/06/2025-14:25:44] [TRT] [I] Compiler backend is used during engine execution.
[07/06/2025-14:25:44] [TRT] [I] Engine generation completed in 126.127 seconds.
[07/06/2025-14:25:44] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 2 MiB, GPU 138 MiB
TensorRT: export success ✅ 128.7s, saved as 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.engine' (7.8 MB)

Export complete (128.8s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv10n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:25:45] [TRT] [I] Loaded engine size: 7 MiB
[07/06/2025-14:25:45] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 13 (MiB)
Loading Results/License Plate/YOLOv10n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:25:45] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:25:45] [TRT] [I] Loaded engine size: 7 MiB
[07/06/2025-14:25:45] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 1, GPU 26 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2396.9±965.5 MB/s, size: 24.5 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189       0.98      0.945      0.978      0.713
Speed: 0.3ms preprocess, 1.2ms inference, 0.0ms loss, 0.2ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: End-to-end models not supported by PaddlePaddle yet

PyTorch: starting from 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 1.3s, saved as 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.onnx' (8.9 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:25:53] :46: ONNX Model ir version: 9
[14:25:53] :47: ONNX Model opset version: 19
[14:25:53] :148: Check it out ==> /model.11/Resize_output_0 has empty input, the index is 1
[14:25:53] :148: Check it out ==> /model.14/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 1.5s, saved as 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.mnn' (4.4 MB)

Export complete (1.5s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv10n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/License Plate/YOLOv10n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1524.4±476.9 MB/s, size: 360.1 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189       0.98      0.945      0.978      0.713
Speed: 1.0ms preprocess, 26.3ms inference, 0.0ms loss, 0.1ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.5 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.8s, saved as 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best.torchscript' (9.1 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/License Plate/YOLOv10n/runs/detect/train/weights/best.torchscript ncnnparam=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.Attention
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2f
inline module = ultralytics.nn.modules.block.C2fCIB
inline module = ultralytics.nn.modules.block.CIB
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.PSA
inline module = ultralytics.nn.modules.block.RepVGGDW
inline module = ultralytics.nn.modules.block.SCDown
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.v10Detect
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.Attention
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2f
inline module = ultralytics.nn.modules.block.C2fCIB
inline module = ultralytics.nn.modules.block.CIB
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.PSA
inline module = ultralytics.nn.modules.block.RepVGGDW
inline module = ultralytics.nn.modules.block.SCDown
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.v10Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
BinaryOp floor_divide not supported yet
BinaryOp remainder not supported yet
ignore torch.topk torch.topk_10 param dim=-1
ignore torch.topk torch.topk_10 param k=300
ignore torch.topk torch.topk_10 param largest=True
ignore torch.topk torch.topk_10 param sorted=True
ignore torch.gather torch.gather_68 param dim=1
ignore torch.gather torch.gather_69 param dim=1
ignore torch.topk torch.topk_11 param dim=-1
ignore torch.topk torch.topk_11 param k=300
ignore torch.topk torch.topk_11 param largest=True
ignore torch.topk torch.topk_11 param sorted=True
ignore pnnx.Expression pnnx_expr_3 param expr=[@0,floor_divide(@1,1)]
ignore Tensor.to Tensor.to_16 param copy=False
ignore Tensor.to Tensor.to_16 param dtype=torch.float
NCNN: export success ✅ 1.2s, saved as 'Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model' (4.5 MB)

Export complete (2.1s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv10n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv10n/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
ERROR ❌ Benchmark failure for NCNN: End-to-end torch.topk operation is not supported for NCNN prediction yet
ERROR ❌ Benchmark failure for IMX: 
ERROR ❌ Benchmark failure for RKNN: End-to-end models not supported by RKNN yet
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/License Plate/YOLOv10n/runs/detect/train/weights/best.pt on Results/datasets/plates/data.yaml at imgsz=640 (241.84s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.5              0.7195                   4.61  216.87
1             TorchScript       ✅        9.1              0.7132                   2.39  418.12
2                    ONNX       ✅        4.5              0.7126                   4.65  215.04
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        7.8              0.7126                   1.22  819.12
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        4.4              0.7132                  26.34   37.96
13                   NCNN       ❎        4.5                   -                      -       -
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to License Plate/YOLOv10n/benchmark_plates.log
      ↳ saved License Plate/YOLOv10n/benchmark_plates.json
[BENCH] YOLOv5u (plates) → License Plate/YOLOv5u/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2149.6±925.3 MB/s, size: 32.6 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.985      0.954      0.978      0.712
Speed: 0.2ms preprocess, 3.7ms inference, 0.0ms loss, 0.8ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.1 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.7s, saved as 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.torchscript' (10.0 MB)

Export complete (0.7s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2420.3±1217.9 MB/s, size: 23.5 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.953      0.978      0.706
Speed: 0.2ms preprocess, 2.1ms inference, 0.0ms loss, 0.6ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.1 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.5s, saved as 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.onnx' (4.9 MB)

Export complete (0.5s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1823.2±675.8 MB/s, size: 18.9 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.953      0.977      0.706
Speed: 0.2ms preprocess, 3.3ms inference, 0.0ms loss, 1.0ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.1 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.5s, saved as 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.onnx' (9.8 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:31:19] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1537, GPU +0, now: CPU 6993, GPU 897 (MiB)
[07/06/2025-14:31:19] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:31:19] [TRT] [I] Input filename:   Results/License Plate/YOLOv5u/runs/detect/train/weights/best.onnx
[07/06/2025-14:31:19] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:31:19] [TRT] [I] Opset version:    19
[07/06/2025-14:31:19] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:31:19] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:31:19] [TRT] [I] Domain:           
[07/06/2025-14:31:19] [TRT] [I] Model version:    0
[07/06/2025-14:31:19] [TRT] [I] Doc string:       
[07/06/2025-14:31:19] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 5, 8400) DataType.FLOAT
TensorRT: building FP16 engine as Results/License Plate/YOLOv5u/runs/detect/train/weights/best.engine
[07/06/2025-14:31:19] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:33:15] [TRT] [I] Detected 1 inputs and 3 output network tensors.
[07/06/2025-14:33:16] [TRT] [I] Total Host Persistent Memory: 427440 bytes
[07/06/2025-14:33:16] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:33:16] [TRT] [I] Max Scratch Memory: 0 bytes
[07/06/2025-14:33:16] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 121 steps to complete.
[07/06/2025-14:33:16] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 3.78775ms to assign 7 blocks to 121 nodes requiring 8755712 bytes.
[07/06/2025-14:33:16] [TRT] [I] Total Activation Memory: 8755200 bytes
[07/06/2025-14:33:16] [TRT] [I] Total Weights Memory: 5085952 bytes
[07/06/2025-14:33:16] [TRT] [I] Engine generation completed in 116.955 seconds.
[07/06/2025-14:33:16] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 2 MiB, GPU 138 MiB
TensorRT: export success ✅ 118.9s, saved as 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.engine' (8.3 MB)

Export complete (119.0s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:33:16] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:33:16] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 13 (MiB)
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:33:16] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:33:16] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:33:16] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +8, now: CPU 1, GPU 26 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2083.3±1100.6 MB/s, size: 25.6 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.953      0.977      0.706
Speed: 0.2ms preprocess, 1.0ms inference, 0.0ms loss, 0.8ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: ERROR ❌️ argument 'half' is not supported for format='paddle'

PyTorch: starting from 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.1 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.6s, saved as 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.onnx' (9.8 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:33:24] :46: ONNX Model ir version: 9
[14:33:24] :47: ONNX Model opset version: 19
[14:33:24] :148: Check it out ==> /model.11/Resize_output_0 has empty input, the index is 1
[14:33:24] :148: Check it out ==> /model.15/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 0.7s, saved as 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.mnn' (4.9 MB)

Export complete (0.7s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2443.6±1990.3 MB/s, size: 58.2 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.953      0.978      0.706
Speed: 1.0ms preprocess, 23.7ms inference, 0.0ms loss, 0.5ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.1 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.7s, saved as 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best.torchscript' (10.0 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/License Plate/YOLOv5u/runs/detect/train/weights/best.torchscript ncnnparam=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C3
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.Detect
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C3
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
NCNN: export success ✅ 1.2s, saved as 'Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model' (5.0 MB)

Export complete (1.9s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv5u/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Loading Results/License Plate/YOLOv5u/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3506.4±2658.6 MB/s, size: 60.3 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.953      0.978      0.706
Speed: 1.0ms preprocess, 42.5ms inference, 0.0ms loss, 0.6ms postprocess per image
ERROR ❌ Benchmark failure for IMX: IMX only supported for YOLOv8
ERROR ❌ Benchmark failure for RKNN: inference not supported on GPU
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/License Plate/YOLOv5u/runs/detect/train/weights/best.pt on Results/datasets/plates/data.yaml at imgsz=640 (544.33s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)      FPS
0                 PyTorch       ✅        5.1              0.7117                   3.68   272.01
1             TorchScript       ✅       10.0               0.706                   2.06   485.32
2                    ONNX       ✅        4.9              0.7061                   3.34   299.36
3                OpenVINO       ❌        0.0                   -                      -        -
4                TensorRT       ✅        8.3              0.7057                   0.96  1036.38
5                  CoreML       ❌        0.0                   -                      -        -
6   TensorFlow SavedModel       ❌        0.0                   -                      -        -
7     TensorFlow GraphDef       ❌        0.0                   -                      -        -
8         TensorFlow Lite       ❌        0.0                   -                      -        -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -        -
10          TensorFlow.js       ❌        0.0                   -                      -        -
11           PaddlePaddle       ❌        0.0                   -                      -        -
12                    MNN       ✅        4.9               0.706                  23.74    42.12
13                   NCNN       ✅        5.0              0.7058                  42.54    23.51
14                    IMX       ❌        0.0                   -                      -        -
15                   RKNN       ❌        0.0                   -                      -        -

      ↳ saved logs to License Plate/YOLOv5u/benchmark_plates.log
      ↳ saved License Plate/YOLOv5u/benchmark_plates.json
[BENCH] YOLOv8n (plates) → License Plate/YOLOv8n/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1973.4±1381.9 MB/s, size: 98.8 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.988      0.949      0.977      0.718
Speed: 0.2ms preprocess, 3.1ms inference, 0.0ms loss, 0.8ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (6.0 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.6s, saved as 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.torchscript' (11.9 MB)

Export complete (0.6s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2815.3±1075.5 MB/s, size: 33.9 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.956      0.977      0.709
Speed: 0.2ms preprocess, 2.0ms inference, 0.0ms loss, 0.7ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (6.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.5s, saved as 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.onnx' (5.9 MB)

Export complete (0.5s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2234.4±624.5 MB/s, size: 22.0 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.956      0.977      0.709
Speed: 0.2ms preprocess, 4.5ms inference, 0.0ms loss, 0.9ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (6.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.7s, saved as 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.onnx' (11.7 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:36:43] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1538, GPU +0, now: CPU 6963, GPU 801 (MiB)
[07/06/2025-14:36:43] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:36:43] [TRT] [I] Input filename:   Results/License Plate/YOLOv8n/runs/detect/train/weights/best.onnx
[07/06/2025-14:36:43] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:36:43] [TRT] [I] Opset version:    19
[07/06/2025-14:36:43] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:36:43] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:36:43] [TRT] [I] Domain:           
[07/06/2025-14:36:43] [TRT] [I] Model version:    0
[07/06/2025-14:36:43] [TRT] [I] Doc string:       
[07/06/2025-14:36:43] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 5, 8400) DataType.FLOAT
TensorRT: building FP16 engine as Results/License Plate/YOLOv8n/runs/detect/train/weights/best.engine
[07/06/2025-14:36:43] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:38:40] [TRT] [I] Detected 1 inputs and 3 output network tensors.
[07/06/2025-14:38:41] [TRT] [I] Total Host Persistent Memory: 388672 bytes
[07/06/2025-14:38:41] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:38:41] [TRT] [I] Max Scratch Memory: 0 bytes
[07/06/2025-14:38:41] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 128 steps to complete.
[07/06/2025-14:38:41] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 4.96477ms to assign 9 blocks to 128 nodes requiring 9063936 bytes.
[07/06/2025-14:38:41] [TRT] [I] Total Activation Memory: 9062400 bytes
[07/06/2025-14:38:41] [TRT] [I] Total Weights Memory: 6089536 bytes
[07/06/2025-14:38:41] [TRT] [I] Engine generation completed in 117.988 seconds.
[07/06/2025-14:38:41] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 2 MiB, GPU 138 MiB
TensorRT: export success ✅ 120.1s, saved as 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.engine' (9.3 MB)

Export complete (120.2s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:38:41] [TRT] [I] Loaded engine size: 9 MiB
[07/06/2025-14:38:41] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 14 (MiB)
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:38:41] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:38:41] [TRT] [I] Loaded engine size: 9 MiB
[07/06/2025-14:38:41] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +8, now: CPU 1, GPU 28 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1642.0±516.1 MB/s, size: 19.1 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.956      0.977      0.708
Speed: 0.2ms preprocess, 1.1ms inference, 0.0ms loss, 0.8ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: ERROR ❌️ argument 'half' is not supported for format='paddle'

PyTorch: starting from 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (6.0 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.6s, saved as 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.onnx' (11.7 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:38:49] :46: ONNX Model ir version: 9
[14:38:49] :47: ONNX Model opset version: 19
[14:38:49] :148: Check it out ==> /model.10/Resize_output_0 has empty input, the index is 1
[14:38:49] :148: Check it out ==> /model.13/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 0.7s, saved as 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.mnn' (5.8 MB)

Export complete (0.8s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2382.6±1962.0 MB/s, size: 48.5 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.956      0.977      0.709
Speed: 1.0ms preprocess, 29.2ms inference, 0.0ms loss, 0.5ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (6.0 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.6s, saved as 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best.torchscript' (11.9 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/License Plate/YOLOv8n/runs/detect/train/weights/best.torchscript ncnnparam=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2f
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.Detect
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2f
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.head.Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
NCNN: export success ✅ 1.2s, saved as 'Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model' (5.9 MB)

Export complete (1.8s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLOv8n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Loading Results/License Plate/YOLOv8n/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2883.5±1769.2 MB/s, size: 50.7 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.987      0.956      0.977      0.709
Speed: 1.0ms preprocess, 38.6ms inference, 0.0ms loss, 0.5ms postprocess per image
ERROR ❌ Benchmark failure for IMX: ERROR ❌️ argument 'half' is not supported for format='imx'
ERROR ❌ Benchmark failure for RKNN: inference not supported on GPU
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/License Plate/YOLOv8n/runs/detect/train/weights/best.pt on Results/datasets/plates/data.yaml at imgsz=640 (327.39s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        6.0              0.7181                    3.1  322.94
1             TorchScript       ✅       11.9              0.7093                   2.03   492.7
2                    ONNX       ✅        5.9              0.7085                   4.47  223.66
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        9.3              0.7083                   1.07  929.98
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        5.8              0.7091                  29.24    34.2
13                   NCNN       ✅        5.9              0.7092                  38.64   25.88
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to License Plate/YOLOv8n/benchmark_plates.log
      ↳ saved License Plate/YOLOv8n/benchmark_plates.json
[BENCH] GhostYOLO (plates) → License Plate/GhostYOLO/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1656.2±493.4 MB/s, size: 33.1 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.984      0.943      0.971      0.699
Speed: 0.2ms preprocess, 5.6ms inference, 0.0ms loss, 0.6ms postprocess per image

PyTorch: starting from 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (3.8 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 1.4s, saved as 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.torchscript' (7.5 MB)

Export complete (1.4s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1916.8±338.9 MB/s, size: 24.5 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.985      0.945      0.971      0.689
Speed: 0.2ms preprocess, 2.7ms inference, 0.0ms loss, 0.7ms postprocess per image

PyTorch: starting from 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (3.8 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.9s, saved as 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.onnx' (3.5 MB)

Export complete (1.0s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2364.8±1053.4 MB/s, size: 34.9 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.984      0.945       0.97       0.69
Speed: 0.2ms preprocess, 4.7ms inference, 0.0ms loss, 0.8ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (3.8 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.8s, saved as 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.onnx' (6.9 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:42:20] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1538, GPU +0, now: CPU 6677, GPU 897 (MiB)
[07/06/2025-14:42:20] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:42:20] [TRT] [I] Input filename:   Results/License Plate/GhostYOLO/runs/detect/train/weights/best.onnx
[07/06/2025-14:42:20] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:42:20] [TRT] [I] Opset version:    19
[07/06/2025-14:42:20] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:42:20] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:42:20] [TRT] [I] Domain:           
[07/06/2025-14:42:20] [TRT] [I] Model version:    0
[07/06/2025-14:42:20] [TRT] [I] Doc string:       
[07/06/2025-14:42:20] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 5, 8400) DataType.FLOAT
TensorRT: building FP16 engine as Results/License Plate/GhostYOLO/runs/detect/train/weights/best.engine
[07/06/2025-14:42:20] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:44:51] [TRT] [I] Detected 1 inputs and 3 output network tensors.
[07/06/2025-14:44:52] [TRT] [I] Total Host Persistent Memory: 692480 bytes
[07/06/2025-14:44:52] [TRT] [I] Total Device Persistent Memory: 1024 bytes
[07/06/2025-14:44:52] [TRT] [I] Max Scratch Memory: 0 bytes
[07/06/2025-14:44:52] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 240 steps to complete.
[07/06/2025-14:44:52] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 11.8296ms to assign 9 blocks to 240 nodes requiring 9434624 bytes.
[07/06/2025-14:44:52] [TRT] [I] Total Activation Memory: 9433600 bytes
[07/06/2025-14:44:52] [TRT] [I] Total Weights Memory: 3669248 bytes
[07/06/2025-14:44:52] [TRT] [I] Engine generation completed in 152.31 seconds.
[07/06/2025-14:44:52] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 2 MiB, GPU 138 MiB
TensorRT: export success ✅ 154.5s, saved as 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.engine' (8.0 MB)

Export complete (154.5s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:44:52] [TRT] [I] Loaded engine size: 7 MiB
[07/06/2025-14:44:52] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +9, now: CPU 1, GPU 12 (MiB)
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:44:52] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:44:52] [TRT] [I] Loaded engine size: 7 MiB
[07/06/2025-14:44:52] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +8, now: CPU 2, GPU 24 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1546.4±359.4 MB/s, size: 18.2 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.984      0.945      0.971      0.689
Speed: 0.2ms preprocess, 1.3ms inference, 0.0ms loss, 0.7ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: ERROR ❌️ argument 'half' is not supported for format='paddle'

PyTorch: starting from 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (3.8 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.9s, saved as 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.onnx' (6.9 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:45:01] :46: ONNX Model ir version: 9
[14:45:01] :47: ONNX Model opset version: 19
[14:45:01] :148: Check it out ==> /model.10/Resize_output_0 has empty input, the index is 1
[14:45:01] :148: Check it out ==> /model.13/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 1.0s, saved as 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.mnn' (3.5 MB)

Export complete (1.1s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1182.9±462.2 MB/s, size: 19.1 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.985      0.945      0.971      0.689
Speed: 1.0ms preprocess, 24.5ms inference, 0.0ms loss, 0.5ms postprocess per image

PyTorch: starting from 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (3.8 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 1.5s, saved as 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best.torchscript' (7.5 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/License Plate/GhostYOLO/runs/detect/train/weights/best.torchscript ncnnparam=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.C3Ghost
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.GhostBottleneck
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.conv.GhostConv
inline module = ultralytics.nn.modules.head.Detect
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.C3Ghost
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.GhostBottleneck
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.conv.GhostConv
inline module = ultralytics.nn.modules.head.Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
NCNN: export success ✅ 1.4s, saved as 'Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model' (3.5 MB)

Export complete (2.9s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/GhostYOLO/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Loading Results/License Plate/GhostYOLO/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2012.0±804.5 MB/s, size: 22.6 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.985      0.945      0.971       0.69
Speed: 0.9ms preprocess, 39.2ms inference, 0.0ms loss, 0.5ms postprocess per image
ERROR ❌ Benchmark failure for IMX: IMX only supported for YOLOv8
ERROR ❌ Benchmark failure for RKNN: inference not supported on GPU
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/License Plate/GhostYOLO/runs/detect/train/weights/best.pt on Results/datasets/plates/data.yaml at imgsz=640 (365.06s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        3.8              0.6991                   5.64  177.25
1             TorchScript       ✅        7.5              0.6893                   2.73  365.64
2                    ONNX       ✅        3.5              0.6896                   4.69  212.96
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        8.0              0.6893                   1.32  756.66
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        3.5              0.6895                  24.54   40.75
13                   NCNN       ✅        3.5              0.6895                  39.24   25.48
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to License Plate/GhostYOLO/benchmark_plates.log
      ↳ saved License Plate/GhostYOLO/benchmark_plates.json
[BENCH] YOLO11n (plates) → License Plate/YOLO11n/runs/detect/train/weights/best.pt
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2112.0±651.6 MB/s, size: 18.9 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.986      0.954      0.979      0.722
Speed: 0.2ms preprocess, 4.4ms inference, 0.0ms loss, 0.6ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.2 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.8s, saved as 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.torchscript' (10.4 MB)

Export complete (0.9s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best.torchscript imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best.torchscript imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best.torchscript for TorchScript inference...
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best.torchscript for TorchScript inference...
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2116.8±497.8 MB/s, size: 19.3 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.984      0.955      0.978      0.715
Speed: 0.2ms preprocess, 2.4ms inference, 0.0ms loss, 0.6ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.2 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.8s, saved as 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.onnx' (5.1 MB)

Export complete (0.8s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best.onnx imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best.onnx imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime CUDAExecutionProvider
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 3192.8±2152.8 MB/s, size: 57.5 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.985      0.956      0.978      0.713
Speed: 0.2ms preprocess, 4.7ms inference, 0.0ms loss, 0.9ms postprocess per image
ERROR ❌ Benchmark failure for OpenVINO: inference not supported on GPU

PyTorch: starting from 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.2 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 0.6s, saved as 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.onnx' (10.1 MB)

TensorRT: starting export with TensorRT 10.12.0.36...
[07/06/2025-14:48:20] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU -1537, GPU +0, now: CPU 6600, GPU 897 (MiB)
[07/06/2025-14:48:20] [TRT] [I] ----------------------------------------------------------------
[07/06/2025-14:48:20] [TRT] [I] Input filename:   Results/License Plate/YOLO11n/runs/detect/train/weights/best.onnx
[07/06/2025-14:48:20] [TRT] [I] ONNX IR version:  0.0.9
[07/06/2025-14:48:20] [TRT] [I] Opset version:    19
[07/06/2025-14:48:20] [TRT] [I] Producer name:    pytorch
[07/06/2025-14:48:20] [TRT] [I] Producer version: 2.7.1
[07/06/2025-14:48:20] [TRT] [I] Domain:           
[07/06/2025-14:48:20] [TRT] [I] Model version:    0
[07/06/2025-14:48:20] [TRT] [I] Doc string:       
[07/06/2025-14:48:20] [TRT] [I] ----------------------------------------------------------------
TensorRT: input "images" with shape(1, 3, 640, 640) DataType.FLOAT
TensorRT: output "output0" with shape(1, 5, 8400) DataType.FLOAT
TensorRT: building FP16 engine as Results/License Plate/YOLO11n/runs/detect/train/weights/best.engine
[07/06/2025-14:48:20] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/06/2025-14:49:35] [TRT] [I] Compiler backend is used during engine build.
[07/06/2025-14:50:45] [TRT] [I] Detected 1 inputs and 1 output network tensors.
[07/06/2025-14:50:46] [TRT] [I] Total Host Persistent Memory: 536016 bytes
[07/06/2025-14:50:46] [TRT] [I] Total Device Persistent Memory: 0 bytes
[07/06/2025-14:50:46] [TRT] [I] Max Scratch Memory: 1382400 bytes
[07/06/2025-14:50:46] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 188 steps to complete.
[07/06/2025-14:50:46] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 9.89081ms to assign 11 blocks to 188 nodes requiring 9524224 bytes.
[07/06/2025-14:50:46] [TRT] [I] Total Activation Memory: 9523200 bytes
[07/06/2025-14:50:46] [TRT] [I] Total Weights Memory: 5303298 bytes
[07/06/2025-14:50:46] [TRT] [I] Compiler backend is used during engine execution.
[07/06/2025-14:50:46] [TRT] [I] Engine generation completed in 145.895 seconds.
[07/06/2025-14:50:46] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 2 MiB, GPU 138 MiB
TensorRT: export success ✅ 148.1s, saved as 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.engine' (8.8 MB)

Export complete (148.2s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best.engine imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best.engine imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:50:46] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:50:46] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 14 (MiB)
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best.engine for TensorRT inference...
[07/06/2025-14:50:46] [TRT] [I] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
[07/06/2025-14:50:46] [TRT] [I] Loaded engine size: 8 MiB
[07/06/2025-14:50:46] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 1, GPU 28 (MiB)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1724.2±514.5 MB/s, size: 19.5 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.985      0.956      0.978      0.715
Speed: 0.2ms preprocess, 1.2ms inference, 0.0ms loss, 0.7ms postprocess per image
ERROR ❌ Benchmark failure for CoreML: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow SavedModel: ERROR ❌️ argument 'half' is not supported for format='saved_model'
ERROR ❌ Benchmark failure for TensorFlow GraphDef: ERROR ❌️ argument 'half' is not supported for format='pb'
ERROR ❌ Benchmark failure for TensorFlow Lite: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow Edge TPU: inference not supported on GPU
ERROR ❌ Benchmark failure for TensorFlow.js: inference not supported on GPU
ERROR ❌ Benchmark failure for PaddlePaddle: ERROR ❌️ argument 'half' is not supported for format='paddle'

PyTorch: starting from 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.2 MB)

ONNX: starting export with onnx 1.17.0 opset 19...
ONNX: slimming with onnxslim 0.1.59...
ONNX: export success ✅ 1.0s, saved as 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.onnx' (10.1 MB)

MNN: starting export with MNN 3.2.1...
Start to Convert Other Model Format To MNN Model..., target version: 3.2
[14:50:55] :46: ONNX Model ir version: 9
[14:50:55] :47: ONNX Model opset version: 19
[14:50:55] :148: Check it out ==> /model.11/Resize_output_0 has empty input, the index is 1
[14:50:55] :148: Check it out ==> /model.14/Resize_output_0 has empty input, the index is 1
Start to Optimize the MNN Net...
inputTensors : [ images, ]
outputTensors: [ output0, ]
Converted Success!
MNN: export success ✅ 1.1s, saved as 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.mnn' (5.0 MB)

Export complete (1.2s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best.mnn imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best.mnn imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best.mnn for MNN inference...
MNN use low precision
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1206.0±487.3 MB/s, size: 17.0 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.984      0.955      0.978      0.714
Speed: 1.0ms preprocess, 27.7ms inference, 0.0ms loss, 0.6ms postprocess per image

PyTorch: starting from 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 5, 8400) (5.2 MB)

TorchScript: starting export with torch 2.7.1+cu126...
TorchScript: export success ✅ 0.8s, saved as 'Results/License Plate/YOLO11n/runs/detect/train/weights/best.torchscript' (10.4 MB)

NCNN: starting export with NCNN 1.0.20250503...
NCNN: running '/home/cetech/trafficmonitor/.venv/lib/python3.11/site-packages/ultralytics/pnnx Results/License Plate/YOLO11n/runs/detect/train/weights/best.torchscript ncnnparam=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param ncnnbin=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin ncnnpy=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py pnnxparam=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param pnnxbin=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin pnnxpy=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py pnnxonnx=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx fp16=1 device=cuda inputshape="[1, 3, 640, 640]"'
pnnxparam = Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.param
pnnxbin = Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.bin
pnnxpy = Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model_pnnx.py
pnnxonnx = Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.pnnx.onnx
ncnnparam = Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.ncnn.param
ncnnbin = Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model.ncnn.bin
ncnnpy = Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model/model_ncnn.py
fp16 = 1
optlevel = 2
device = cuda
inputshape = [1,3,640,640]f32
inputshape2 = 
customop = 
moduleop = 
############# pass_level0
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.Attention
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2PSA
inline module = ultralytics.nn.modules.block.C3k
inline module = ultralytics.nn.modules.block.C3k2
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.PSABlock
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.conv.DWConv
inline module = ultralytics.nn.modules.head.Detect
inline module = torch.nn.modules.linear.Identity
inline module = ultralytics.nn.modules.block.Attention
inline module = ultralytics.nn.modules.block.Bottleneck
inline module = ultralytics.nn.modules.block.C2PSA
inline module = ultralytics.nn.modules.block.C3k
inline module = ultralytics.nn.modules.block.C3k2
inline module = ultralytics.nn.modules.block.DFL
inline module = ultralytics.nn.modules.block.PSABlock
inline module = ultralytics.nn.modules.block.SPPF
inline module = ultralytics.nn.modules.conv.Concat
inline module = ultralytics.nn.modules.conv.Conv
inline module = ultralytics.nn.modules.conv.DWConv
inline module = ultralytics.nn.modules.head.Detect

----------------

############# pass_level1
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
NCNN: export success ✅ 1.2s, saved as 'Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model' (5.1 MB)

Export complete (2.1s)
Results saved to /home/cetech/trafficmonitor/Results/License Plate/YOLO11n/runs/detect/train/weights
Predict:         yolo predict task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model imgsz=640 half 
Validate:        yolo val task=detect model=Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model imgsz=640 data=/kaggle/working/datasets/License-Plate-Recognition-5/data.yaml half 
Visualize:       https://netron.app
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Loading Results/License Plate/YOLO11n/runs/detect/train/weights/best_ncnn_model for NCNN inference...
Setting batch=1 input of shape (1, 3, 640, 640)
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1676.6±636.3 MB/s, size: 31.6 KB)
val: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corruval: Scanning /home/cetech/trafficmonitor/Results/datasets/plates/valid/labels.cache... 2042 images, 3 backgrounds, 0 corrupt: 100%|##########| 2042/2042 [00:00<?, ?it/s]
                   all       2042       2189      0.984      0.955      0.978      0.715
Speed: 1.0ms preprocess, 41.9ms inference, 0.0ms loss, 0.5ms postprocess per image
ERROR ❌ Benchmark failure for IMX: IMX only supported for YOLOv8
ERROR ❌ Benchmark failure for RKNN: inference not supported on GPU
Setup complete ✅ (12 CPUs, 31.2 GB RAM, 198.5/231.2 GB disk)

Benchmarks complete for Results/License Plate/YOLO11n/runs/detect/train/weights/best.pt on Results/datasets/plates/data.yaml at imgsz=640 (365.61s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
                   Format Status❔  Size (MB) metrics/mAP50-95(B) Inference time (ms/im)     FPS
0                 PyTorch       ✅        5.2              0.7218                   4.44  225.16
1             TorchScript       ✅       10.4              0.7146                   2.41  414.29
2                    ONNX       ✅        5.1              0.7132                   4.66  214.47
3                OpenVINO       ❌        0.0                   -                      -       -
4                TensorRT       ✅        8.8              0.7146                   1.24  808.62
5                  CoreML       ❌        0.0                   -                      -       -
6   TensorFlow SavedModel       ❌        0.0                   -                      -       -
7     TensorFlow GraphDef       ❌        0.0                   -                      -       -
8         TensorFlow Lite       ❌        0.0                   -                      -       -
9     TensorFlow Edge TPU       ❌        0.0                   -                      -       -
10          TensorFlow.js       ❌        0.0                   -                      -       -
11           PaddlePaddle       ❌        0.0                   -                      -       -
12                    MNN       ✅        5.0              0.7144                  27.72   36.07
13                   NCNN       ✅        5.1              0.7146                  41.89   23.87
14                    IMX       ❌        0.0                   -                      -       -
15                   RKNN       ❌        0.0                   -                      -       -

      ↳ saved logs to License Plate/YOLO11n/benchmark_plates.log
      ↳ saved License Plate/YOLO11n/benchmark_plates.json