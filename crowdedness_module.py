import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# --- 1. 설정 ---
TENSORRT_ENGINE_PATH = "depth.engine"
MODEL_INPUT_H, MODEL_INPUT_W = 518, 518

# 점유율 계산을 위한 파라미터
DEPTH_LEVEL_START = 0.5
DEPTH_LEVEL_END = 2.1
DEPTH_LEVEL_STEP = 0.2
MIN_CONTOUR_AREA = 1000
DECISION_THRESHOLD = 0.5

class DepthEstimator:
    """TensorRT 엔진을 로드하고 깊이 추론을 수행하는 클래스"""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"TensorRT 엔진 로드 중: {engine_path}")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if not self.engine:
            raise RuntimeError("엔진을 로드하는 데 실패했습니다.")
            
        self.context = self.engine.create_execution_context()
        print("✅ 엔진 로드 및 컨텍스트 생성 완료.")
        
        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
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
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        output_shape = (1, self.engine.get_binding_shape(1)[2], self.engine.get_binding_shape(1)[3])
        depth_map = self.h_output.reshape(output_shape)
        return np.squeeze(depth_map)

    def _preprocess_image(self, image_bgr):
        resized_frame = cv2.resize(image_bgr, (MODEL_INPUT_W, MODEL_INPUT_H))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = (rgb_frame.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return np.expand_dims(normalized_frame.transpose(2, 0, 1), 0).astype(np.float32)

def calculate_occupancy_score(depth_map, original_shape):
    h, w = original_shape
    depth_map_resized = cv2.resize(depth_map, (w, h))
    ratios_per_level = []
    for level in np.arange(DEPTH_LEVEL_START, DEPTH_LEVEL_END, DEPTH_LEVEL_STEP):
        mask = (depth_map_resized <= level).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) < MIN_CONTOUR_AREA: continue
        hull = cv2.convexHull(main_contour)
        area_contour = cv2.contourArea(main_contour)
        area_hull = cv2.contourArea(hull)
        if area_hull > 0:
            ratios_per_level.append((area_hull - area_contour) / area_hull)
    return np.sum(ratios_per_level)

try:
    depth_model = DepthEstimator(TENSORRT_ENGINE_PATH)
except Exception as e:
    print(f"모델 초기화 중 심각한 오류 발생: {e}")
    depth_model = None

def get_crowdedness_decision(frame):
    if depth_model is None:
        print("모델이 초기화되지 않았습니다.")
        return -1 # 에러를 의미하는 값
    if frame is None or frame.size == 0:
        return 1 # 빈 프레임은 안전(GO)
    
    depth_map = depth_model.run_inference(frame)
    score = calculate_occupancy_score(depth_map, frame.shape[:2])
    
    # ✨✨ 수정된 부분: "STOP" -> 2, "GO" -> 1 ✨✨
    if score > DECISION_THRESHOLD:
        decision = 2  # STOP
    else:
        decision = 1  # GO
    
    return decision