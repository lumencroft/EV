# depthModule.py

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TENSORRT_ENGINE_PATH = "depth.engine"
MODEL_INPUT_SHAPE = (518, 518)
PREPROCESS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PREPROCESS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class DepthEstimator:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine.")
            
        self.context = self.engine.create_execution_context()
        
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        input_shape = self.engine.get_tensor_shape(self.input_name)
        output_shape = self.engine.get_tensor_shape(self.output_name)
        self.output_shape = tuple(output_shape)
        
        self.h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

    def __del__(self):
        self.d_input.free()
        self.d_output.free()

    def run_inference(self, image_bgr):
        tensor = self._preprocess_image(image_bgr)
        np.copyto(self.h_input, tensor.ravel())
        
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        self.context.set_tensor_address(self.input_name, self.d_input)
        self.context.set_tensor_address(self.output_name, self.d_output)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        depth_map = self.h_output.reshape(self.output_shape)
        return np.squeeze(depth_map)

    def _preprocess_image(self, image_bgr):
        resized = cv2.resize(image_bgr, (MODEL_INPUT_SHAPE[1], MODEL_INPUT_SHAPE[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - PREPROCESS_MEAN) / PREPROCESS_STD
        tensor = np.expand_dims(normalized.transpose(2, 0, 1), 0)
        return tensor