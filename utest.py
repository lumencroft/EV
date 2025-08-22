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

# --- 2. 클래스 및 함수 정의 ---

class DepthEstimator:
    """TensorRT 10 API에 맞춰 수정된 최종 버전 클래스"""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"TensorRT 엔진 로드 중: {engine_path}")
        try:
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        except Exception as e:
            raise RuntimeError(f"엔진 파일 로드 실패: {e}")
        
        if not self.engine:
            raise RuntimeError("엔진을 디코딩하는 데 실패했습니다.")
            
        self.context = self.engine.create_execution_context()
        print("✅ 엔진 로드 및 컨텍스트 생성 완료.")

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
        # 자원 해제 시 에러 방지를 위해 hasattr로 확인
        if hasattr(self, 'd_input') and self.d_input:
            self.d_input.free()
        if hasattr(self, 'd_output') and self.d_output:
            self.d_output.free()

    def run_inference(self, image_bgr):
        tensor = self._preprocess_image(image_bgr)
        np.copyto(self.h_input, tensor.ravel())
        
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        depth_map = self.h_output.reshape(self.output_shape)
        return np.squeeze(depth_map)

    def _preprocess_image(self, image_bgr):
        resized_frame = cv2.resize(image_bgr, (MODEL_INPUT_W, MODEL_INPUT_H))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = (rgb_frame.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return np.expand_dims(normalized_frame.transpose(2, 0, 1), 0).astype(np.float32)

def calculate_occupancy_and_visualize(depth_map, original_shape):
    """점유율 계산과 시각화 데이터를 함께 반환하는 함수"""
    h, w = original_shape
    depth_map_resized = cv2.resize(depth_map, (w, h))
    
    contour_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    ratios_per_level = []
    
    for level in np.arange(DEPTH_LEVEL_START, DEPTH_LEVEL_END, DEPTH_LEVEL_STEP):
        mask = (depth_map_resized <= level).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        main_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(main_contour) < MIN_CONTOUR_AREA:
            continue
            
        hull = cv2.convexHull(main_contour)
        
        # 시각화: 등고선(초록색), Convex Hull(노란색) 그리기
        cv2.drawContours(contour_canvas, [main_contour], -1, (0, 255, 0), 1)
        cv2.drawContours(contour_canvas, [hull], -1, (0, 255, 255), 1)

        area_contour = cv2.contourArea(main_contour)
        area_hull = cv2.contourArea(hull)
        
        if area_hull > 0:
            ratios_per_level.append((area_hull - area_contour) / area_hull)
            
    score = np.sum(ratios_per_level)
    return score, contour_canvas

def main():
    """웹캠을 실행하고 실시간으로 혼잡도를 분석 및 시각화합니다."""
    # 모델 초기화
    try:
        depth_model = DepthEstimator(TENSORRT_ENGINE_PATH)
    except Exception as e:
        print(f"모델 초기화 중 심각한 오류 발생: {e}")
        return

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return

    print("웹캠 시작... 'q'를 누르면 종료됩니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 프레임을 읽을 수 없습니다.")
            break
        
        # 1. Depth Map 추론
        depth_map = depth_model.run_inference(frame)
        
        # 2. 점수 계산 및 시각화용 데이터 생성
        score, contour_canvas = calculate_occupancy_and_visualize(depth_map, frame.shape[:2])
        
        # 3. 최종 결정 (GO / STOP)
        if score > DECISION_THRESHOLD:
            decision = 2  # STOP
            status_text = "STOP"
            color = (0, 0, 255)  # 빨간색
        else:
            decision = 1  # GO
            status_text = "GO"
            color = (0, 255, 0)  # 초록색
        
        # 4. 화면에 결과 표시
        # 메인 화면에 점수와 상태 표시
        score_text = f"Score: {score:.2f}"
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, score_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Depth Map 시각화 (컬러맵 적용)
        depth_viz = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        depth_viz_resized = cv2.resize(depth_viz, (frame.shape[1], frame.shape[0]))

        # 5. 창 보여주기
        cv2.imshow("Crowdedness Cam", frame)
        cv2.imshow("Depth Map", depth_viz_resized)
        cv2.imshow("Contours", contour_canvas)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    del depth_model
    print("프로그램 종료.")

# --- 3. 메인 코드 실행 ---
if __name__ == '__main__':
    main()