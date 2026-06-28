import carla
import cv2
import numpy as np
import queue
import math
import matplotlib.pyplot as plt
from line_detection import get_bird_eye_view, detect_edges, fit_inner_lanes, draw_lane_lines, LaneTracker
from steering_control import PIDController 


#전역 큐 설정(카메라 데이터 수신용)
img_queue = queue.Queue(maxsize=1)
img_queue_top = queue.Queue(maxsize=1)

#탑뷰 카메라 데이터 콜백
def process_top_camera_data(image):
    img_bgra = np.array(image.raw_data).reshape((image.height, image.width, 4))
    frame = img_bgra[:, :, :3].copy()
    if img_queue_top.full(): img_queue_top.get()
    img_queue_top.put(frame)

#메인 ADAS(Advanced Driver Assistance Systems) 카메라 데이터 콜백
def process_camera_data(image):
    img_bgra = np.array(image.raw_data).reshape((image.height, image.width, 4))
    frame = img_bgra[:, :, :3].copy()
    if img_queue.full(): img_queue.get()
    img_queue.put(frame)

#슬라이더(Trackbar) 인터페이스용 더미 함수
def nothing(x): pass

#차선 변경 조향 제어를 위한 5차 다항식 함수(Quintic Polynomial), counter steering, 3차로는 not robustness --> 5차로 robustness하게
def get_quintic_steering(progress, direction, power=0.18):
    #1. (progress**2): 시작 시 조향 가속도를 0으로 만들어 '부드러운 진입' 유도
    #2. (1 - progress)**2: 종료 시 조향 가속도를 0으로 만들어 '부드러운 복귀' 유도
    #3. (0.5 - progress): 0.5초를 기점으로 값의 부호(+/-)가 반전됨 (가장 핵심!)
    #    - 0.0~0.5: 양수(+) -> 타겟 차선으로 핸들 꺾기
    #    - 0.5~1.0: 음수(-) -> 차체를 일자로 펴기 위한 '카운터 스티어링' 강제 발생
    #이 세 항을 곱하면 핸들이 '꺾였다가 -> 반대로 꺾였다가 -> 복귀하는' 물결파가 생성됨
    steer = direction * power * (progress**2 * (1 - progress)**2 * (0.5 - progress) * 100) 
    return steer

def main():
    '''
    #BEV에서 직선을 얻기 위한 TUning 과정
    cv2.namedWindow('BEV_Tuning')
    cv2.createTrackbar('Top Y', 'BEV_Tuning', 320, 600, nothing)
    cv2.createTrackbar('Top Width', 'BEV_Tuning', 45, 200, nothing)
    cv2.createTrackbar('Bottom Y', 'BEV_Tuning', 420, 600, nothing)
    cv2.createTrackbar('Bottom Margin', 'BEV_Tuning', 151, 400, nothing)
    '''
    actor_list = []
    #논문용 데이터 기록을 위한 리스트 초기화
    time_history = []
    cte_history = []
    steer_history = []
    try:
        #CARLA 클라이언트 접속 및 Town04(고속도로) 로드
        client = carla.Client('localhost', 2000)
        world = client.load_world('Town04')
        #차량 스폰 (Tesla Model 3)
        vehicle_bp = world.get_blueprint_library().filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[119]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        #1인칭 ADAS 카메라 설정
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800'); camera_bp.set_attribute('image_size_y', '600')
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=0.5, z=1.3)), attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(lambda image: process_camera_data(image))
        #3인칭 탑뷰 카메라(물리적 위치 확인용)
        top_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        top_cam_bp.set_attribute('image_size_x', '800'); top_cam_bp.set_attribute('image_size_y', '600')
        top_transform = carla.Transform(carla.Location(x=-8.0, z=15.0), carla.Rotation(pitch=-60.0))
        top_camera = world.spawn_actor(top_cam_bp, top_transform, attach_to=vehicle)
        actor_list.append(top_camera)
        top_camera.listen(lambda image: process_top_camera_data(image))
        #제어기
        pid = PIDController(kp=0.00005, ki=0.0, kd=0.00035) 
        #차선 인식 객체 및 변수
        tracker = LaneTracker()
        frame_count = 0
        current_throttle = 0.0
        lane_width = 350.0 
        is_lane_width_calibrated = False 
        width_measurements = []
        #차선 변경 변수
        is_changing_lane = False
        start_location = None        
        lane_change_distance = 25.0 #차선 변경 수행 거리(m)/튜닝한 값
        settling_mode_frames = 0 #차선 변경 후 안착 제어 타이머
        target_offset_max = 0.0
        current_offset = 0.0
        print("자율주행 시뮬레이션 시작...")
        while True:
            #카메라 뷰
            try: frame = img_queue.get(timeout=0.1)
            except queue.Empty: continue
            #탑 뷰
            try: top_frame = img_queue_top.get(timeout=0.1)
            except queue.Empty: top_frame = None
            cv2.imshow("Original Frame",frame)
            frame_count += 1
            current_location = vehicle.get_location() #현재 위치 받아오기-->나중에 gps를 활용한 거리 받아오기 해볼 것
            v = vehicle.get_velocity() #속도 정보 받아오기
            speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2) #km/h로 변환
            current_throttle = min(0.25, current_throttle + 0.005) #부드러운 가속
            #차선 인식 결과 가져오기
            bev_raw, minv, src_points = get_bird_eye_view(frame)       
            binary_mask = detect_edges(bev_raw)              
            l_fit, r_fit, l_type, r_type = fit_inner_lanes(binary_mask) 
            smooth_l, smooth_r = tracker.update(l_fit, r_fit) 
            #실시간 차선 폭 캘리브레이션, 총 20개를 평균내어서 사용
            if not is_lane_width_calibrated:
                if smooth_l is not None and smooth_r is not None:
                    y_eval = binary_mask.shape[0] 
                    left_x = smooth_l[0] * y_eval + smooth_l[1]
                    right_x = smooth_r[0] * y_eval + smooth_r[1]
                    measured_width = abs(right_x - left_x)
                    if 100 < measured_width < 800:
                        width_measurements.append(measured_width)
                if len(width_measurements) >= 20:
                    is_lane_width_calibrated = True
                    lane_width = sum(width_measurements) / len(width_measurements)
                    print(f"🎯 [캘리브레이션 완료] 차선 폭: {lane_width:.1f}px")
            #키보드 입력 기반 차선 변경
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a') and l_type == "DASHED" and not is_changing_lane:
                is_changing_lane = True
                start_location = current_location
                target_offset_max = -lane_width
                tracker.left_fit_history.clear() 
            elif key == ord('d') and r_type == "DASHED" and not is_changing_lane:
                is_changing_lane = True
                start_location = current_location
                target_offset_max = lane_width
                tracker.right_fit_history.clear()  
            #차선 변경 로직
            ff_steer = 0.0 #feedforward 정보를 이용해 미리 예측하여 과거에 영향을 받지 않도록
            if is_changing_lane and start_location is not None:
                dist_traveled = current_location.distance(start_location) #차선 이동 시 거리
                progress = min(1.0, dist_traveled / lane_change_distance) #차선 변경 진행도
                #5차 다항식 기반 차량 이동 궤적
                # 차선 변경 시 물리적 충격(가속도 변화)을 0으로 만들기 위한 6가지 경계 조건:
                # 1. Start (t=0): 위치=0, 속도=0, 가속도=0  --> c_0, c_1, c_2 는 모두 0이 됨
                # 2. End   (t=1): 위치=1, 속도=0, 가속도=0  --> 연립방정식 결과 c_3=10, c_4=-15, c_5=6
                # 결론 수식: f(t) = 6*t^5 - 15*t^4 + 10*t^3
                curve_factor = (6 * progress**5 - 15 * progress**4 + 10 * progress**3)
                current_offset = target_offset_max * curve_factor
                direction = 1.0 if target_offset_max > 0 else -1.0 #방향 설정
                #80% 지점까지 차선 인식 없이 steering
                steer_progress = min(1.0, progress / 0.80)
                if steer_progress < 1.0:
                    ff_steer = get_quintic_steering(steer_progress, direction, power=0.20)
                #차선 변경 완료
                if progress >= 1.0:
                    is_changing_lane = False
                    start_location = None
                    current_offset = 0.0
                    target_offset_max = 0.0
                    #에러 누적 확실하게 초기화
                    pid.integral = 0.0
                    pid.prev_error = 0.0
                    settling_mode_frames = 30 #안착 제어 모드 활성화 (약 1초)
            #상황별 제어 계수 동적 변경 (Gain Scheduling)
            #차선변경
            if is_changing_lane:
                steer_limit = 0.35 
            #안착 제어
            elif settling_mode_frames > 0:
                pid.kp = 0.00045  
                pid.kd = 0.00350  
                steer_limit = 0.35  
                settling_mode_frames -= 1
            #일반 제어
            else:
                pid.kp = 0.00005
                pid.kd = 0.00050 
                steer_limit = 0.15   
            #가상 궤적 계산 및 시각화
            v_l_fit = smooth_l.copy() if smooth_l is not None else None
            v_r_fit = smooth_r.copy() if smooth_r is not None else None
            if v_l_fit is not None: v_l_fit[1] += current_offset
            if v_r_fit is not None: v_r_fit[1] += current_offset
            final_res, cte = draw_lane_lines(frame, v_l_fit, v_r_fit, minv, l_type, r_type, src_points, lane_width)
            #2.0 픽셀 데드존 적용
            if abs(cte) < 2.0:
                cte = 0.0   
            if is_changing_lane:
                raw_steer = ff_steer # 차선 변경 시에는 Feedforward 궤적 추종
            else:
                raw_steer = pid.get_control(cte) #평소에는 PID 피드백 제어
            final_steer = max(-steer_limit, min(steer_limit, raw_steer))

            #캘리브레이션이 끝나고 제어가 정상적으로 시작된 시점부터 기록
            if is_lane_width_calibrated:
                time_history.append(frame_count)
                cte_history.append(cte)
                steer_history.append(final_steer)

            #UI 정보 표시
            status = f"CHANGING" if is_changing_lane else ("SETTLING" if settling_mode_frames > 0 else "KEEPING")
            cv2.putText(final_res, f"STATE: {status}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(final_res, f"Speed: {speed_kmh:.1f} km/h", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #시각화
            if top_frame is not None: cv2.imshow("0. Top-Down View", top_frame)
            cv2.imshow("1. ADAS View", final_res)
            cv2.imshow("BEV View", bev_raw) # [디버깅용] 탑뷰 원본
            
            cv2.imshow("2. Binary Mask", binary_mask) # [디버깅용] 마스킹 결과
            #맨 처음 차선 폭 캘리브레이션
            if frame_count > 30 and is_lane_width_calibrated:
                throttle = 0.0 if speed_kmh > 45 else current_throttle
                vehicle.apply_control(carla.VehicleControl(steer=final_steer, throttle=throttle))
            if key == ord('q'): break
    finally:
        for actor in actor_list: actor.destroy()
        cv2.destroyAllWindows()

    #주행 종료 후 matplotlib을 이용한 결과 시각화 및 논문용 이미지 저장
        if len(time_history) > 0:
            print("데이터 시각화 중... 그래프 창이 열립니다.")
            
            # 폰트 및 스타일 기본 설정
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            fig.canvas.manager.set_window_title('Control Data Analysis')

            # 1. CTE (조향 오차) 그래프
            ax1.plot(time_history, cte_history, color='blue', linewidth=1.5, label='Cross Track Error (CTE)')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1) # 중심선(0) 표시
            ax1.set_title('Cross Track Error over Time', fontsize=14, fontweight='bold')
            ax1.set_ylabel('CTE (Pixels)', fontsize=12)
            ax1.grid(True, linestyle=':', alpha=0.7)
            ax1.legend(loc='upper right')

            # 2. Steering Angle (조향각) 그래프
            ax2.plot(time_history, steer_history, color='green', linewidth=1.5, label='Steering Angle')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax2.set_title('Steering Angle Control over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Frame Count', fontsize=12)
            ax2.set_ylabel('Steering Command', fontsize=12)
            ax2.grid(True, linestyle=':', alpha=0.7)
            ax2.legend(loc='upper right')

            plt.tight_layout()
            
            # 고화질(300 DPI) 이미지 자동 저장
            plt.savefig('autonomous_control_result.png', dpi=300)
            print("✅ 'autonomous_control_result.png' 파일이 성공적으로 저장되었습니다!")
            
            # 그래프 화면에 띄우기
            plt.show()
        else:
            print("기록된 데이터가 없어 그래프를 그릴 수 없습니다.")
if __name__ == '__main__': main()