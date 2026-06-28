#PID control를 활용한 steering control
#Kp*error+Kd*d(error)/dt+Ki*integral(error)dt
class PIDController:
    def __init__(self,kp,ki,kd):
        #PID 계수
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.prev_error=0 #error 변화율 계산을 위한 값
        self.integral=0 #누적 error

    def get_control(self,error,dt=0.1):
        p_term=self.kp*error #P:현재 오차만큼
        #I:누적된 오차만큼
        self.integral+=error*dt 
        i_term=self.ki*self.integral
        #D:error 변화량만큼
        derivative=(error-self.prev_error)/dt 
        d_term=self.kd*derivative
        self.prev_error=error
        return p_term+i_term+d_term