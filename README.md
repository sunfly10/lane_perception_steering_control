# 🚗 CARLA 자율주행: 비전 기반 차선 인지 및 조향 제어

본 프로젝트는 인하대학교 전기전자공학부 종합설계 프로젝트의 일환으로 카메라 및 라이다 센서 융합 기반 자율주행 차량의 통합 인지 및 제어 시스템 중, 비전 기반 차선 인지 및 횡방향 조향 제어 모듈을 독립적으로 구현한 레포지토리입니다.

CARLA 시뮬레이터의 내장 경로 계획 API를 배제하고, 비전 데이터 추출부터 차량 동역학적 조향각 인가까지 구축했습니다. 역투영 변환(IPM)과 동적 임계값을 이용한 강건한 인지 로직, 그리고 5차 다항식(Quintic Polynomial) 플래닝을 통해 차선 변경 시의 물리적 충격과 불안정성을 극복한 실증적 연구 결과를 포함하고 있습니다.

---

## 1. 설치 및 실행
[https://carla.org/], [https://carla.readthedocs.io/en/latest/]  <-- 홈페이지, 공식 문서 참고하여 설치 및 실행
아래의 두 명령어를 통해 시뮬레이션 환경 진행함.
    • conda activate carla_env: 가상환경 활성화
    • ./CarlaUE4.sh -quality-level=Low: 시뮬레이터 사용

---

## 2. 제어 성능 검증
### 2.1 인지 파이프라인 및 시점 변환
| 원본 전방 카메라 (Raw View) | 조감도 및 색상 마스킹 (BEV Masking) | 3인칭 관찰자 시점 (Top-Down View) |
| :---: | :---: | :---: |
| <img width="1043" height="769" alt="Screenshot from 2026-06-28 20-59-54" src="https://github.com/user-attachments/assets/ef4629b5-61f0-4ee6-9038-0b7a2a9a55cf" /> | <img width="1043" height="769" alt="Screenshot from 2026-06-28 21-00-12" src="https://github.com/user-attachments/assets/5c90dff6-4b06-4083-b863-428f30d08405" />| <img width="1043" height="769" alt="Screenshot from 2026-06-28 21-00-01" src="https://github.com/user-attachments/assets/85fdc9c8-b81e-4b77-8bd2-807d5bfcdcb5" />|
| *사다리꼴 형태의 원근 왜곡 발생* | *IPM 변환 및 HSV 동적 마스킹* | *차량의 물리적 위치 검증용* |

### 2.2 동적 차선 변경 및 제어 (Dynamic Lane Changing)
| 1. 직진 차선 유지 (KEEPING) | 2. 차선 변경 수행 (CHANGING) | 3. 차선 변경 완료 및 안착 |
| :---: | :---: | :---: |
| <img src="Screenshot from 2026-06-28 21-00-20.jpg" width="300"/> | <img src="Screenshot from 2026-06-28 21-00-34.jpg" width="300"/> | <img src="Screenshot from 2026-06-28 21-00-44.jpg" width="300"/> |
| *오차 -1.8px (PID 데드존 개입)* | *피드포워드 S-Curve 궤적 추종* | *안착 제어 모드 발동 후 중앙 정렬* |

### 2.3 동역학 조향 제어 안정성 그래프
| Cross Track Error (횡방향 오차) | Steering Angle (실시간 조향각) |
| :---: | :---: |
| <img src="Screenshot from 2026-06-28 21-00-54.png" width="400"/> | <img src="Screenshot from 2026-06-28 21-00-59.png" width="400"/> |
| *차선 변경 시 발생하는 오차를 오버슈트 없이 즉각 수렴시킴* | *물리적 한계를 고려한 부드러운 카운터 스티어링 (S자 파형) 구현* |

---
