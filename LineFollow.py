import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
import threading
import matplotlib.pyplot as plt

ORANGE_HSV = (19, 255, 255)  # 橙色的HSV颜色范围
COLOR_THRESHOLD = 10  # 色彩识别的接受百分比阈值
CENTER_X = 320  # 中心点的x坐标

MOTOR_EA, MOTOR_I2, MOTOR_I1, MOTOR_EB, MOTOR_I4, MOTOR_I3 = (13, 19, 26, 16, 20, 21)  # GPIO引脚的编号
PWM_FREQUENCY = 120  # PWM频率

GPIO.setmode(GPIO.BCM)  # 设置GPIO引脚编号模式为BCM

GPIO.setup([MOTOR_EA, MOTOR_I2, MOTOR_I1, MOTOR_EB, MOTOR_I4, MOTOR_I3], GPIO.OUT)  # 设置GPIO引脚的工作模式为输出
GPIO.output([MOTOR_EA, MOTOR_I2, MOTOR_EB, MOTOR_I3], GPIO.LOW)  # 初始化GPIO引脚的输出状态为低电平
GPIO.output([MOTOR_I1, MOTOR_I4], GPIO.HIGH)  # 初始化GPIO引脚的输出状态为高电平

pwm_a = GPIO.PWM(MOTOR_EA, PWM_FREQUENCY)  # 创建PWM对象，控制电机A的速度
pwm_b = GPIO.PWM(MOTOR_EB, PWM_FREQUENCY)  # 创建PWM对象，控制电机B的速度
pwm_a.start(0)  # 启动PWM输出，初始占空比为0
pwm_b.start(0)  # 启动PWM输出，初始占空比为0

cap = cv2.VideoCapture(0)  # 打开摄像头，创建视频捕捉对象


class PIDController:
    def __init__(self, kp, ki, kd, center, duty=10):
        """
        初始化PID控制器对象。

        参数:
            kp (float): 比例系数
            ki (float): 积分系数
            kd (float): 微分系数
            center (float): 理想中心值
            duty (float): 初始占空比 (默认为10)
        """
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.last_err = 0  # 上一次误差
        self.curr_err = 0  # 当前误差
        self.u = 0  # 控制量
        self.integral = 0  # 积分项
        self.ideal_center = center  # 理想中心值
        self.last_duty = duty  # 上一次占空比
        self.curr_duty = duty  # 当前占空比
        self.last_update = time.monotonic()  # 上一次更新时间

    def update(self, feedback):
        """
        更新PID控制器。
        参数: feedback (float): 反馈值
        返回: float: 更新后的占空比
        """
        self.curr_err = self.ideal_center - feedback  # 计算当前误差
        self.integral += self.curr_err  # 更新积分项
        delta_time = time.monotonic() - self.last_update
        if delta_time == 0:
            delta_time = 0.001
        self.u = (
            self.kp * self.curr_err
            + self.ki * self.integral
            + self.kd * (self.curr_err - self.last_err) / delta_time
        )  # 计算控制量
        self.last_err = self.curr_err  # 更新上一次误差
        self.curr_duty = self.last_duty + self.u  # 计算当前占空比
        if self.curr_duty > 100:
            self.curr_duty = 100  # 限制占空比上限为100
        if self.curr_duty < 0:
            self.curr_duty = 0  # 限制占空比下限为0
        self.last_duty = self.curr_duty  # 更新上一次占空比
        self.last_update = time.monotonic()  # 更新上一次更新时间

        return self.curr_duty  # 返回当前占空比


def filter_color(frame):  # 根据颜色过滤图像
    """
    参数: frame(numpy.ndarray): 输入图像帧
    返回: numpy.ndarray: 过滤后的二值图像
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV颜色空间
    lower_color = np.array([0, 100, 100])  # 颜色下限
    upper_color = np.array([ORANGE_HSV[0] + COLOR_THRESHOLD, 255, 255])  # 颜色上限
    return cv2.inRange(hsv, lower_color, upper_color)  # 返回颜色过滤后的二值图像


def get_center_point(mask):  # 获取二值图像中的中心点坐标
    """
    参数: mask(numpy.ndarray): 输入二值图像
    返回: tuple: 中心点的坐标(x, y)
    """
    coords = cv2.findNonZero(mask)  # 找到非零像素点的坐标
    if coords is None or len(coords) == 0:
        return None
    return np.mean(coords, axis=0)[0]  # 计算坐标的平均值作为中心点坐标


def on_trackbar(val):  # 滑动条回调函数，用于调整颜色接受百分比。
    global COLOR_THRESHOLD
    COLOR_THRESHOLD = val  # 滑动条的当前值


cv2.namedWindow("Controls")  # 创建窗口
cv2.createTrackbar("Accept %", "Controls", COLOR_THRESHOLD, 100, on_trackbar)  # 创建滑动条并绑定回调函数


def calibrate():
    # 校准函数，用于设置中心点
    global CENTER_X
    print("正在校准...")
    while True:
        ret, frame = cap.read()  # 读取摄像头帧
        if not ret:
            continue
        frame = filter_color(frame)  # 过滤颜色
        current_center = get_center_point(frame)  # 获取目标中心点
        if current_center is None:
            continue
        cv2.circle(frame, (int(current_center[0]), int(current_center[1])), 5, (0, 0, 255), -1)  # 在帧上绘制当前中心点
        cv2.circle(frame, (int(CENTER_X), 220), 5, (150, 0, 0), -1)  # 在帧上绘制目标中心点
        cv2.imshow("Frame", frame)  # 显示帧
        key = cv2.waitKey(1) & 0xFF  # 等待按键输入
        if key == ord("q"):
            CENTER_X = current_center[0]  # 设置中心点为当前中心点
            print("设置中心点为 x = %s！" % CENTER_X)
        if key == ord("x"):
            cv2.destroyAllWindows()
            break


def main():
    calibrate()
    now = time.monotonic()  # 记录当前时间

    # 三次跑均使用此代码，只调整左右轮PID参数

    # 第一次试跑
    l_pid = PID(17, 0.12, 180, 0.5, 60)  # 左PID控制器
    r_pid = PID(12, 0.18, 180, 0.5, 60)  # 右PID控制器

    # 第二次试跑（最快）
    # l_pid = PID(12, 0.09, 150, 0.5, 100)  # 左PID控制器
    # r_pid = PID(8.5, 0.14, 150, 0.5, 100)  # 右PID控制器

    # 第三次试跑
    # l_pid = PID(10, 0.6, 120, 0.5, 200)  # 左PID控制器
    # r_pid = PID(7, 0.9, 120, 0.5, 200)  # 右PID控制器

    input("按回车键开始！")
    while True:
        ret, frame = cap.read()  # 读取摄像头帧
        if not ret:
            continue
        frame = filter_color(frame)  # 过滤颜色
        if frame is None:
            continue

        current_center = get_center_point(frame)  # 获取目标中心点
        if current_center is None:
            continue
        current_center = current_center[0]
        percent = 1 - (current_center / (CENTER_X * 2))  # 计算百分比偏移量

        if percent < 0:
            percent = 0

        pwma.ChangeDutyCycle(r_pid.update(1 - percent))  # 更新左电机的占空比
        pwmb.ChangeDutyCycle(l_pid.update(percent))  # 更新右电机的占空比

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(time.monotonic() - now)  # 输出程序运行时间
            break

    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口


if __name__ == "__main__":
    main()
