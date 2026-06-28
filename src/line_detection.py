import cv2
import numpy as np
from collections import deque
import warnings

warnings.simplefilter('ignore', np.RankWarning) #경고는 무시

class LaneTracker:
    def __init__(self,max_len=30):
        #30개의 frame을 이용해서 차선 인식(좌/우)
        self.left_fit_history=deque(maxlen=max_len)
        self.right_fit_history=deque(maxlen=max_len)
        
    def update(self, left_fit, right_fit):
        #차선이 있으면 계속해서 업데이트
        if left_fit is not None: self.left_fit_history.append(left_fit)
        if right_fit is not None: self.right_fit_history.append(right_fit)
        #a, b를 평균내서 좌/우 각각 parameter 사용(1차 fitting:y=ax+b), 차선이 없으면 None return
        l = np.mean(self.left_fit_history, axis=0) if self.left_fit_history else None
        r = np.mean(self.right_fit_history, axis=0) if self.right_fit_history else None
        return l, r

#차선 찾기 알고리즘: BEV로 변환 --> noise 제거를 위해 blur --> hsv를 활용해 노란색 흰색 찾기(동적 적응형 임계값 적용) --> 모폴로지 연산(노이즈 제거 및 끊어진 점선 잇기)
#1. BEV로 변환
def get_bird_eye_view(frame):
    height, width = frame.shape[:2] #현재 사용 IMG는 800*600
    #BEV 변환 영역 설정(사다리꼴 모양, 밑의 4가지 값은 BEV에서 직선이 나오도록 튜닝한 값)
    top_y, top_width = 320, 45
    bottom_y, bottom_margin = 420, 151
    #src:원본 사다리꼴 영역/dst:변환 후 영역
    src_points = np.float32([[width//2 - top_width, top_y], [width//2 + top_width, top_y],[width - bottom_margin, bottom_y], [bottom_margin, bottom_y]])
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points) #변환행렬 구하기 
    minv = cv2.getPerspectiveTransform(dst_points, src_points) #역행렬 구하기
    return cv2.warpPerspective(frame, matrix, (width, height)), minv, src_points

def detect_edges(bev_frame):
    height, width = bev_frame.shape[:2]
    blurred_frame = cv2.GaussianBlur(bev_frame, (5, 5), 0) #2. bev frame에서 noise 제거를 위해 blur
    #3. hsv를 활용해 노란색 흰색 찾기(동적 적응형 임계값 적용)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2]) #현재 화면(BEV)의 평균 밝기를 계산
    #흰색 차선 임계값 설정 및 적용
    dynamic_v_low = int(np.clip(avg_brightness + 50, 130, 210)) #np.clip을 사용하여 아무리 어두워져도 130 이하로는 안 내려가고, 아무리 밝아도 210은 넘지 않게 함
    mask_w = cv2.inRange(hsv, np.array([0, 0, dynamic_v_low]), np.array([180, 40, 255]))
    #노란색 차선 임계값 설정 및 적용
    dynamic_y_low = int(np.clip(avg_brightness - 20, 40, 100))
    mask_y = cv2.inRange(hsv, np.array([15, 40, dynamic_y_low]), np.array([45, 255, 255]))
    binary_mask = cv2.bitwise_or(mask_y, mask_w) #노란색 혹은 흰색인 부분 mask
    binary_mask[:int(height * 0.45), :] = 0 #본넷 부분 등 불필요한 상단 영역 잘라내기
    #4. 모폴로지 연산(노이즈 제거 및 끊어진 점선 잇기)
    kernel_small = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small) #3*3 kernel로 noise 제거
    kernel_large = np.ones((35, 5), np.uint8)
    pretty_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_large) #차선 원래 두께로 복원
    return pretty_mask

#fitting(직진 차선-->1차 함수 fitting)
def fit_inner_lanes(binary_mask):
    height, width = binary_mask.shape
    midpoint = width // 2
    #histogram을 이용해서 차선 여부 확인
    histogram = np.sum(binary_mask[height//2:, :], axis=0)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    if histogram[leftx_base] < 500: leftx_base = None
    if histogram[rightx_base] < 500: rightx_base = None
    left_x, left_y, right_x, right_y = [], [], [], [] #결과 정보 담을 장소
    curr_l, curr_r = leftx_base, rightx_base #차선 감지 시작점 설정
    margin = 120 #차선 감지 마진
    #아래에서 위로 차선 찾기
    search_range_y = range(height-1, int(height * 0.45), -1)
    for y in search_range_y:
        row = binary_mask[y, :]
        #왼쪽 찾기
        if curr_l is not None:
            l_min, l_max = max(0, curr_l-margin), min(midpoint, curr_l+margin)
            whites = np.where(row[l_min:l_max] > 100)[0]
            if len(whites) > 2: curr_l = int(np.mean(whites)) + l_min; left_x.append(curr_l); left_y.append(y)
        #오른쪽 찾기
        if curr_r is not None:
            r_min, r_max = max(midpoint, curr_r-margin), min(width, curr_r+margin)
            whites = np.where(row[r_min:r_max] > 100)[0]
            if len(whites) > 2: curr_r = int(np.mean(whites)) + r_min; right_x.append(curr_r); right_y.append(y)
    #밀도를 이용해서 점선 실선 구별
    l_type = "SOLID" if (len(left_x) / len(search_range_y)) > 0.6 else "DASHED"
    r_type = "SOLID" if (len(right_x) / len(search_range_y)) > 0.6 else "DASHED"
    #fitting 계수 도출
    l_fit = np.polyfit(left_y, left_x, 1) if len(left_x) > 15 else None
    r_fit = np.polyfit(right_y, right_x, 1) if len(right_x) > 15 else None
    return l_fit, r_fit, l_type, r_type

#차선 그리기
def draw_lane_lines(original_frame, left_fit, right_fit, minv, l_type, r_type, src_points, lane_width_px=350.0):
    height, width = original_frame.shape[:2]
    color_warp = np.zeros_like(original_frame).astype(np.uint8)
    ploty = np.linspace(int(height*0.4), height - 1, 20)
    l_x_at_y, r_x_at_y = None, None
    y_pos = height * 0.7
    #왼쪽 오른쪽 둘다 있을 때
    if left_fit is not None and right_fit is None:
        v_right_fit = left_fit.copy()
        v_right_fit[1] += lane_width_px
        pts = np.array([np.transpose(np.vstack([v_right_fit[0]*ploty + v_right_fit[1], ploty]))])
        cv2.polylines(color_warp, np.int_([pts]), False, (255, 255, 0), 10, lineType=cv2.LINE_AA)
        l_x_at_y = left_fit[0]*y_pos + left_fit[1]
        r_x_at_y = v_right_fit[0]*y_pos + v_right_fit[1]
    #오른쪽 차선만 있을 때, 가상의 선을 만들어준다
    elif right_fit is not None and left_fit is None:
        v_left_fit = right_fit.copy()
        v_left_fit[1] -= lane_width_px
        pts = np.array([np.transpose(np.vstack([v_left_fit[0]*ploty + v_left_fit[1], ploty]))])
        cv2.polylines(color_warp, np.int_([pts]), False, (255, 255, 0), 10, lineType=cv2.LINE_AA)
        r_x_at_y = right_fit[0]*y_pos + right_fit[1]
        l_x_at_y = v_left_fit[0]*y_pos + v_left_fit[1]
    #왼쪽 차선만 있을 때, 가상의 선을 만들어준다
    elif left_fit is not None and right_fit is not None:
        pts_l = np.array([np.transpose(np.vstack([left_fit[0]*ploty + left_fit[1], ploty]))])
        pts_r = np.array([np.transpose(np.vstack([right_fit[0]*ploty + right_fit[1], ploty]))])
        cv2.polylines(color_warp, np.int_([pts_l]), False, (0, 255, 0), 10, lineType=cv2.LINE_AA)
        cv2.polylines(color_warp, np.int_([pts_r]), False, (0, 255, 0), 10, lineType=cv2.LINE_AA)
        l_x_at_y = left_fit[0]*y_pos + left_fit[1]
        r_x_at_y = right_fit[0]*y_pos + right_fit[1]
    newwarp = cv2.warpPerspective(color_warp, minv, (width, height)) #카메라 view를 위해 역행렬을 활용한 변환
    result = cv2.addWeighted(original_frame, 1, newwarp, 0.8, 0) #원래 frame에 결과 합치기
    cv2.polylines(result, [np.int32(src_points)], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    #현재 카메라 frame에서의 중심과 차선 인식 결과로부터의 중심의 차를 계산(cross track error)
    car_center = width / 2
    if l_x_at_y and r_x_at_y: lane_center = (l_x_at_y + r_x_at_y) / 2
    else: lane_center = car_center
    cte = lane_center - car_center
    #logging을 위한 것들
    cv2.putText(result, f"L: {l_type} | R: {r_type}", (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(result, f"CTE Error: {cte:.1f} px", (50, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, lineType=cv2.LINE_AA)
    bar_x, bar_y, bar_w, bar_h = 250, 540, 300, 20
    cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
    cv2.line(result, (bar_x + bar_w//2, bar_y - 5), (bar_x + bar_w//2, bar_y + bar_h + 5), (255, 255, 255), 2, lineType=cv2.LINE_AA)
    error_pos = int(bar_x + bar_w//2 + (cte / 150) * (bar_w//2))
    error_pos = max(bar_x, min(bar_x + bar_w, error_pos))
    cv2.circle(result, (error_pos, bar_y + bar_h//2), 10, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        
    return result, cte