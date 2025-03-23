import sys
import serial
import serial.tools.list_ports
import numpy as np
from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,
                           QHBoxLayout,QLabel,QSlider,QPushButton,QComboBox,
                           QTabWidget,QGroupBox,QLineEdit,QTableWidget,
                           QTableWidgetItem,QMessageBox,QGridLayout,
                           QTextEdit)
from PyQt5.QtCore import Qt,QTimer,pyqtSignal
from PyQt5.QtGui import QTextCharFormat,QColor,QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
##########################线性插值非直线轨迹###############################
# def generate_joint_trajectory(point_list):
#     """生成关节轨迹的离散采样点"""
#     #点列表示例
#     #point_list=[(1,2,3,4,5,6),(1,2,3,4,5,6)] #还可以更多点，分别对应6个舵机的位置
#     point_list=np.array(point_list)  # 转换成np数组

#     #默认按照50ms一插值，然后每个中间点用500ms到达
#     trajectory=[]  # 存储采样点

#     #拟合与采样
#     for joint in range(point_list.shape[1]):  # 对每个关节进行插值处理
#         joint_values=point_list[:,joint]  # 获取每个关节的角度值
#         time_values=np.arange(0,len(point_list)*500,500)  # 每个点之间时间间隔为500ms

#         #创建线性插值函数
#         interp_func=interp1d(time_values,joint_values,kind='linear')

#         #生成采样时间点，每50ms采样一次
#         sampled_times=np.arange(0,time_values[-1]+50,50)
#         sampled_joint_values=interp_func(sampled_times)

#         #将插值后的关节值添加到轨迹列表中
#         if len(trajectory)==0: #对于第一个关节，直接初始化轨迹列表
            
#             trajectory=[[val] for val in sampled_joint_values]
#         else:
#             for i,joint_value in enumerate(sampled_joint_values):
#                 trajectory[i].append(joint_value)

#         trajectory=[[int(joint_value) for joint_value in joint] for joint in trajectory] #转换为整数

#         #可视化插值函数和采样点 测试用
#         plt.figure(figsize=(8,6))
#         plt.plot(time_values,joint_values,'o',label='Original Points',markersize=10)  # 原始点
#         plt.plot(sampled_times,sampled_joint_values,'-',label='Interpolated Curve')  # 插值曲线
#         plt.scatter(sampled_times,sampled_joint_values,color='red',label='Sampled Points',zorder=5)  # 采样点
#         plt.xlabel('Time (ms)')
#         plt.ylabel(f'Joint {joint + 1} Angle')
#         plt.title(f'Interpolation for Joint {joint + 1}')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     return trajectory

# point_list=[
#     [1500,1500,1500,1500,1500,1500],
#     [500,1700,1300,1100,2000,2300],
#     [1500,1500,1500,1500,1500,1500],
#     [1500,1500,1500,1500,1500,1500],
#     [1500,1500,1500,1500,1500,1500],
    
# ]

# filtered_trajectory=[point_list[0]] #第一个点
# for point in point_list:
#         if point!=filtered_trajectory[-1]:
#             filtered_trajectory.append(point) #如果和过滤后的最后一个不一样，就添加进去

# print(len(filtered_trajectory))


#trajectory=generate_joint_trajectory(point_list)
#print(trajectory)

#################正逆运动学###############################
# class RoboticArmKinematics:
#     '''机械臂运动学'''
#     def __init__(self):
#         # 机械臂参数
#         self.d1=80      # 底座高度
#         self.a1=-10     # 大臂偏移
#         self.a2=85   # 大臂长度
#         self.a3=80   # 小臂长度
#         self.d5=100    # 手腕长度
#         self.d6=100    # 爪子长度
        
#         # DH参数表 [alpha(i-1), a(i-1), d(i), theta(i)]
#         self.dh_params=np.array([
#             [0,         0,          self.d1,    0],    # 1: 底座旋转
#             [np.pi/2,   self.a1,    0,          np.pi/2],    # 2: 大臂
#             [0,         self.a2,    0,          0],    # 3: 小臂
#             [0,         self.a3,    0,          np.pi/2],    # 4: 手腕
#             [np.pi/2,   0,          self.d5,    np.pi/2],    # 5: 爪子旋转
#             [0,         0,          self.d6,    0] #爪子末端
#         ]) 

#         #这舵机500和2500的时候不是对应的准的，而且还是270度的，所以搞个插值！
#         self.joint_angle={
#             0:{"servo_values":[2295,1500,814],"angle":np.radians([90,0,-90])},
#             1:{"servo_values":[2266,1900,1500,1203,1000,882],"angle":np.radians([90,45,0,-45,-60,-90])},
#             2:{"servo_values":[2500,2212,1500,799,500],"angle":np.radians([-135,-90,0,90,130])},
#             3:{"servo_values":[2500,2246,1500,884,500],"angle":np.radians([-130,-90,0,90,135])},
#             4:{"servo_values":[2500,2245,1500,822,500],"angle":np.radians([135,90,0,-90,-135])}
#         }

#     def transform_matrix(self, alpha, a, d, theta):
#         """计算DH变换矩阵
#         alpha(i-1), a(i-1), d(i), theta(i)
#         """
#         # 转换角度到弧度
        
#         ct=np.cos(theta)
#         st=np.sin(theta)
#         ca=np.cos(alpha)
#         sa=np.sin(alpha)
        
#         # 根据图中给出的矩阵形式
#         return np.array([
#             [ct,        -st,        0,          a],
#             [st*ca,     ct*ca,     -sa,        -sa*d],
#             [st*sa,     ct*sa,     ca,         ca*d],
#             [0,         0,          0,          1]
#         ])

#     def forward_kinematics(self, joint_angles):
#         """正运动学计算"""
#         T=np.eye(4)
#         positions=[]
        
#         #基座位置
#         positions.append(np.array([0, 0, 0]))
        
#         #计算每个关节的位置
#         for i in range(len(joint_angles)): #传入前5个，最后控制夹爪的不在考虑范围内
#             alpha=self.dh_params[i][0]
#             a=self.dh_params[i][1]
#             d=self.dh_params[i][2]
#             theta=joint_angles[i]+self.dh_params[i][3]#由pwm转换成正方向角度传进来的
            
#             Ti=self.transform_matrix(alpha, a, d, theta)
#             T=T @ Ti
            
#             position=T[:3, 3]
#             positions.append(position)

#         alpha=self.dh_params[5][0]
#         a=self.dh_params[5][1]
#         d=self.dh_params[5][2]
#         theta=self.dh_params[5][3]
#         T6=self.transform_matrix(alpha, a, d, theta)
#         T=T@T6 #夹爪的
#         position=T[:3, 3]
#         positions.append(position) #末端夹爪位置放进去   #??要放吗
        
#         #计算末端姿态角,就取一个yaw和一个roll,我的joint_angle是纯正方向角度
#         yaw=-(joint_angles[1]+joint_angles[2]+joint_angles[3]) #从z0方向往x0方向转的 正180是往外转到垂直向下，负180是往内转到垂直向下，但是一般到不了负180
#         roll=joint_angles[4] #为0的时候就是初始的
         
#         return positions, yaw, roll
    
#     def inverse_kinematics(self, target_pos, yaw,roll=0): #就不考虑roll了，默认为0把，如果没有yaw限制就搞成false，有就搞成具体数值
#         """逆运动学计算"""
#         result=[] #存储两组解

#         x, y, z=target_pos
#         print(f"目标位置: x={x:.1f}, y={y:.1f}, z={z:.1f}")

#         #x=x-self.a1 #去掉第二个关节的偏移量,不能在这边减去
#         z=z-self.d1 #去掉底座高度

#         theta=np.arctan2(y,x) #底座角度，第一个舵机

#         #这里有两种可能，一种是直接正平面到达，另一种是负平面到达，然后转180度
#         for i in range(2):
#             if i==0: #正平面到达的情况
#                 theta0=theta
#                 c=(x**2+y**2)**0.5-self.a1 #c是正的

#                 ##测量并添加一个工作范围
#                 # if theta0>? or theta0<?
#                 #     continue

#                 m=c-(self.d5+self.d6)*np.sin(yaw)
#                 n=z-(self.d5+self.d6)*np.cos(yaw)

#                 #先算theta1的情况
#                 k1=(self.a3**2-self.a2**2-m**2-n**2)/(-2*self.a2)
#                 a1=m**2+n**2
#                 b1=-2*k1*m
#                 c1=k1**2-n**2
#                 delta=b1**2-4*a1*c1
#                 if delta<0: #一元二次方程无解
#                     continue
#                 theta1_big=-np.arcsin((-b1+delta**0.5)/(2*a1)) #较大的解
#                 theta1_small=-np.arcsin((-b1-delta**0.5)/(2*a1)) #较小的解

#                 #再算theta2的情况
#                 k12=(self.a2**2-self.a3**2-m**2-n**2)/(-2*self.a3)
#                 a12=m**2+n**2
#                 b12=-2*k12*m
#                 c12=k12**2-n**2
#                 delta=b12**2-4*a12*c12
#                 if delta<0: #一元二次方程无解
#                     continue
#                 theta12_big=-np.arcsin((-b12+delta**0.5)/(2*a12)) #较大的解
#                 theta12_small=-np.arcsin((-b12-delta**0.5)/(2*a12)) #较小的解

#                 #计算3个角度，theta12大的对应theta1小的
#                 theta1_1=theta1_big
#                 theta2_1=theta12_small-theta1_big
#                 theta3_1=-yaw-theta1_1-theta2_1
#                 result1=[theta0,theta1_1,theta2_1,theta3_1,roll] #theta4默认为0
            
#                 fk_positions1, fk_yaw1, fk_roll1=self.forward_kinematics(result1) #计算正运动学结果
#                 #print('正运动学结果1',fk_positions1,'位姿',fk_yaw1,',',fk_roll1) #测试用
#                 ##一些舍去解的判断
#                 #if not ???:
#                 ##continue
#                 result.append(result1)

#                 theta1_2=theta1_small
#                 theta2_2=theta12_big-theta1_small
#                 theta3_2=-yaw-theta1_2-theta2_2
#                 result2=[theta0,theta1_2,theta2_2,theta3_2,roll]

#                 fk_positions2, fk_yaw2, fk_roll2=self.forward_kinematics(result2) #计算正运动学结果
#                 #print('正运动学结果2',fk_positions2,'位姿',fk_yaw2,',',fk_roll2) #测试用
#                 if not (theta1_2-theta1_1<0.09 and theta2_2-theta2_1<0.09 and theta3_2-theta3_1<0.09):
#                     ##一些舍去解的判断
#                     ##if not???:
#                     result.append(result2)


#             else: #负平面到达的情况
#                 theta0=theta-np.sign(theta)*np.pi #另一种角度
#                 c=-(x**2+y**2)**0.5-self.a1 #c是负的

#                 ##测量并添加一个工作范围
#                 # if theta0>? or theta0<?
#                 #     continue

#                 m=-c-(self.d5+self.d6)*np.sin(-yaw)
#                 n=z-(self.d5+self.d6)*np.cos(-yaw)

#                 #先算theta1的情况
#                 k1=(self.a3**2-self.a2**2-m**2-n**2)/(-2*self.a2)
#                 a1=m**2+n**2
#                 b1=-2*k1*m
#                 c1=k1**2-n**2
#                 delta=b1**2-4*a1*c1
#                 if delta<0: #一元二次方程无解
#                     continue
#                 theta1_big=np.arcsin((-b1+delta**0.5)/(2*a1)) #较大的解
#                 theta1_small=np.arcsin((-b1-delta**0.5)/(2*a1)) #较小的解

#                 #再算theta2的情况
#                 k12=(self.a2**2-self.a3**2-m**2-n**2)/(-2*self.a3)
#                 a12=m**2+n**2
#                 b12=-2*k12*m
#                 c12=k12**2-n**2
#                 delta=b12**2-4*a12*c12
#                 if delta<0: #一元二次方程无解
#                     continue
#                 theta12_big=np.arcsin((-b12+delta**0.5)/(2*a12)) #较大的解
#                 theta12_small=np.arcsin((-b12-delta**0.5)/(2*a12)) #较小的解

#                 #计算3个角度，theta12大的对应theta1小的
#                 theta1_1=theta1_big #为什么这里会有负号啊
#                 theta2_1=theta12_small-theta1_big #为什么这里会有负号啊
#                 theta3_1=-yaw-theta1_1-theta2_1
#                 result1=[theta0,theta1_1,theta2_1,theta3_1,roll] #theta4默认为0
            
#                 fk_positions1, fk_yaw1, fk_roll1=self.forward_kinematics(result1) #计算正运动学结果
#                 #print('正运动学结果3',fk_positions1,'位姿',fk_yaw1,',',fk_roll1) #测试用
#                 ##一些舍去解的判断
#                 #if not ???:
#                 ##continue
#                 result.append(result1)

#                 theta1_2=theta1_small
#                 theta2_2=theta12_big-theta1_small
#                 theta3_2=-yaw-theta1_2-theta2_2
#                 result2=[theta0,theta1_2,theta2_2,theta3_2,roll]

#                 fk_positions2, fk_yaw2, fk_roll2=self.forward_kinematics(result2) #计算正运动学结果
#                 #print('正运动学结果4',fk_positions2,'位姿',fk_yaw2,',',fk_roll2) #测试用
#                 if not (theta1_2-theta1_1<0.09 and theta2_2-theta2_1<0.09 and theta3_2-theta3_1<0.09):
#                     ##一些舍去解的判断
#                     ##if not???:
#                     result.append(result2)

#         print('逆运动学解个数:',len(result))

#         return result,True
    
#     #转换，纯转换，给进去的就一定是有效的，无效的判断放到前面
#     def pwm_to_angle(self, pwm_values):
#         """PWM值转角度"""
#         angles = []
#         for i, pwm in enumerate(pwm_values):
#             servo_values = self.joint_angle[i]["servo_values"]
#             angle_values = self.joint_angle[i]["angle"]
            
#             # 使用 scipy 的插值方法
#             interp_func = interp1d(servo_values, angle_values, kind='linear', fill_value='extrapolate')
#             angle = interp_func(pwm)
#             angles.append(angle)
#         return angles

#     def angle_to_pwm(self, angles):
#         """角度转PWM值"""
#         pwm_values = []
#         for i, angle in enumerate(angles):
#             servo_values = self.joint_angle[i]["servo_values"]
#             angle_values = self.joint_angle[i]["angle"]
            
#             # 使用 scipy 的插值方法
#             interp_func = interp1d(angle_values, servo_values, kind='linear', fill_value='extrapolate')
#             pwm = interp_func(angle)
#             pwm_values.append(int(max(500, min(2500, pwm))))  # 限制 PWM 范围为 500~2500
#         return pwm_values

#     def plot_interpolation(self):
#         """绘制每个关节的PWM与角度的插值关系"""
#         pwm_values = np.linspace(500, 2500, 100)  # 生成从500到2500的100个PWM值
#         plt.figure(figsize=(12, 8))

#         for i in range(5):  # 处理5个关节
#             servo_values = self.joint_angle[i]["servo_values"]
#             angle_values = self.joint_angle[i]["angle"]
            
#             # 使用 scipy 插值
#             interp_func = interp1d(servo_values, angle_values, kind='linear', fill_value='extrapolate')
#             interpolated_angles = interp_func(pwm_values)

#             # 绘图
#             plt.subplot(3, 2, i+1)
#             plt.plot(pwm_values, interpolated_angles, label="Interpolated Curve")
#             plt.scatter(servo_values, angle_values, color='red', label="Data Points")
#             plt.title(f"Joint {i} PWM to Angle")
#             plt.xlabel("PWM Values")
#             plt.ylabel("Angle (radians)")
#             plt.legend()

#         plt.tight_layout()
#         plt.show()
    
# k=RoboticArmKinematics()
# #k.plot_interpolation()
# joint_pwm=[500,500,500,500,500]
# joint_angle=k.pwm_to_angle(joint_pwm)
# print('pwm2angle舵机转角度',joint_angle)
# joint_pwm1=k.angle_to_pwm(joint_angle)
# print('angle2pwm角度转舵机',joint_pwm1)

# popsition,yaw,roll=k.forward_kinematics(joint_angle)
# print('正运动学位置',popsition[-1])
# print('正运动学yaw',yaw)

# position=popsition[-1]
# joint_angle1,label=k.inverse_kinematics(position,yaw,roll)
# for i,value in enumerate(joint_angle1):
#     print('逆运动学解',i,'角度结果',value)
#     joint_pwm1=k.angle_to_pwm(value)
#     print('逆运动学解',i,'舵机位置',joint_pwm1)
#     position,yaw,roll=k.forward_kinematics(value)
#     print('正运动学',i,'位置',position,'角度yaw',yaw)


# def calculate_joint_positions(joint_values): #更新3d视图用到,给出的是从0org到5org，和夹爪的左右两点，传入的是6舵机位置
#     """计算机械臂各关节位置"""
#     joint_6=joint_values[-1]
#     joint_values=joint_values[:5] #前5舵机位置，用于计算运动学
#     angles=k.pwm_to_angle(joint_values)
#     positions,yaw,roll=k.forward_kinematics(angles)
    
#     jaw_root=np.array(positions[-2]) #夹爪根部
#     jaw_end=np.array(positions[-1]) #夹爪末端

#     #计算夹爪末端两个点
#     #夹爪中心向量
#     center_vector=jaw_end-jaw_root #夹爪根部和末端的连线
#     center_vector/=np.linalg.norm(center_vector)

#     #夹爪roll为0时和z轴垂直
#     z_axis=np.array([0,0,1])
#     end_0roll_vector = np.cross(center_vector, z_axis) #末端两点在roll为0的直线
#     end_0roll_vector /= np.linalg.norm(end_0roll_vector)

#     end_1_end_vector=end_0roll_vector #一侧点和中点的连线
#     end_2_end_vector=-end_0roll_vector #另一侧点和中点的连线

#     #计算旋转矩阵，旋转轴是Center_cector
#     K = np.array([
#         [0, -center_vector[2], center_vector[1]],
#         [center_vector[2], 0, -center_vector[0]],
#         [-center_vector[1], center_vector[0], 0]
#     ])
#     rotation_matrix = (
#         np.eye(3) + 
#         np.sin(roll) * K + 
#         (1 - np.cos(roll)) * np.dot(K, K)
#     )

#     #转了roll角度之后的
#     end_1_end_vector = np.dot(rotation_matrix, end_1_end_vector)
#     end_2_end_vector = np.dot(rotation_matrix, end_2_end_vector)

#     #爪子开合宽度
#     gripper_width = 40
#     if joint_6 != 1500:
#         gripper_width = 20 - abs((joint_6 - 1500) / 1000 * 40)

#     # 计算爪子开合宽度
#     end_1=(jaw_end+gripper_width/2*end_1_end_vector).tolist()
#     end_2=(jaw_end+gripper_width/2*end_2_end_vector).tolist()
    
#     return positions[:5] + [end_1, end_2]

# positions=calculate_joint_positions(joint_pwm)
# print(positions)

###########################定速直线运动############################
            

def fun(given_t, point_start, point_end, total_time, a, a_t, v):
    """计算单个时间点的轨迹值"""
    if 0 <= given_t <= a_t:  # 加速段
        y = 0.5 * a * given_t**2 + point_start
    elif a_t < given_t <= total_time - a_t:  # 匀速段
        y = fun(a_t,point_start, point_end, total_time, a, a_t, v) + v * (given_t - a_t)
    elif total_time - a_t < given_t <= total_time:  # 减速段
        y = fun(total_time-a_t,point_start, point_end, total_time, a, a_t, v)+v*(given_t-total_time+a_t)-0.5*a*(given_t-total_time+a_t)**2
    return y

def generate_trajectory(point_list,time_list=None): #输入的是6个笛卡尔位置，包括夹爪开合，时间列表应该是6个笛卡尔位置减1而且是50的正数
    if time_list is None:
        time_list=[500]*(len(point_list)-1) #时间列表
    #初始化存储各分量的列表
    trajectory=[] #最后应该是一个长度为5的列表，然后要形状变一下

    for i in range(6):
        per_trajectory=[point_list[0][i]] #先把第一个点添加进去
        for j in range(len(point_list)-1):
            point_start=point_list[j][i] #起点
            point_end=point_list[j+1][i] #终点
            total_time=time_list[j] #取时间

            if i==5: #夹爪开合
                #时间取样
                time_vals=np.arange(50,total_time,50) #这里因为最后会单独添加，所以采样时间可以不包括total_time
                
                trajectory_vals=[per_trajectory[-1]]*len(time_vals)+[point_end]
                
                per_trajectory+=trajectory_vals #添加进当前分量

            else: #其他笛卡尔位姿
                a=6*(point_end-point_start)/(total_time**2) #加速度
                if a==0: #两点一样，就没有加速度
                    a_t=0
                else:
                    a_t=total_time/2-((a**2*total_time**2-4*a*(point_end-point_start))**0.5)/(2*abs(a)) #加速度时间 #必须有abs
                v=a*a_t

                #时间取样
                time_vals=np.arange(50,total_time+1,50)

                #计算相应采样点
                trajectory_vals = [fun(t, point_start, point_end, total_time, a, a_t, v) for t in time_vals]

                per_trajectory+=trajectory_vals #添加进当前分量
        
        trajectory.append(per_trajectory)
    trajectory=[list(item) for item in zip(*trajectory)] #转置，变成采样数长度，元素是5笛卡尔位姿+夹爪状态
    return trajectory

# 示例输入
point_list = [
    [0, 0, 0, 0, 0,1500],
    [50, 30, -20, 10, -10,2000],
    [300,200,100,130,140,1500]
]  # 5 个分量的点
time_list = [500,800]  # 时间列表

# 生成轨迹
trajectory = generate_trajectory(point_list, time_list)

# 绘图
plt.figure(figsize=(10, 6))
time_total = sum(time_list)
time_vals_full = np.arange(0, time_total + 1, 50)

for dim in range(6):
    plt.plot(time_vals_full, trajectory[dim], label=f"Trajectory {dim + 1}")

plt.title("5-Dimensional Trajectories")
plt.xlabel("Time (ms)")
plt.ylabel("Position")
plt.legend()
plt.grid(True)
plt.show()
