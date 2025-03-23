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
from PyQt5.QtGui import QTextCharFormat, QColor,QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D


class RoboticArmKinematics:
    '''机械臂运动学'''
    def __init__(self):
        # 机械臂参数
        self.l1 = 50    # 底座高度
        self.l2 = 120   # 大臂长度
        self.l3 = 120   # 小臂长度
        self.l4 = 60    # 手腕长度
        self.l5 = 40    # 爪子长度
        
        # DH参数表 [alpha(i-1), a(i-1), d(i), theta(i)]
        self.dh_params = np.array([
            [0,      0,      self.l1,    None],    # 1: 底座旋转
            [np.pi/2,     0,      0,          None],    # 2: 大臂
            [0,      self.l2, 0,         None],    # 3: 小臂
            [0,      self.l3, 0,         None],    # 4: 手腕
            [0,      self.l4, 0,         None],    # 5: 爪子旋转
        ])

    def transform_matrix(self, alpha, a, d, theta):
        """计算DH变换矩阵
        alpha(i-1), a(i-1), d(i), theta(i)
        """
        # 转换角度到弧度
        
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        # 根据图中给出的矩阵形式
        return np.array([
            [ct,        -st,        0,          a],
            [st*ca,     ct*ca,     -sa,        -sa*d],
            [st*sa,     ct*sa,     ca,         ca*d],
            [0,         0,          0,          1]
        ])

    def forward_kinematics(self, joint_angles):
        """正运动学计算"""
        T = np.eye(4)
        positions = []
        
        # 基座位置
        positions.append(np.array([0, 0, 0]))
        
        # 计算每个关节的位置
        for i in range(len(joint_angles)):
            alpha = self.dh_params[i][0]
            a = self.dh_params[i][1]
            d = self.dh_params[i][2]
            theta = joint_angles[i]#由pwm传进来的
            
            Ti = self.transform_matrix(alpha, a, d, theta)
            T = T @ Ti
            
            position = T[:3, 3]
            positions.append(position)
        
        # 计算末端姿态角,与手腕旋转无关，所以不用orientation
        orientation = T[:3, :3]
        pitch=joint_angles[1]+joint_angles[2]+joint_angles[3]

        if pitch<180:
            pitch=-pitch+np.pi
        else:
            pitch=pitch-np.pi
        
        roll=joint_angles[4]
        #pitch = np.arctan2(orientation[2, 0], np.sqrt(orientation[0, 0]**2 + orientation[1, 0]**2))
        #roll = np.arctan2(orientation[1, 0]/np.cos(pitch), orientation[0, 0]/np.cos(pitch))
        
        return positions, orientation, pitch, roll
    
    def inverse_kinematics(self, target_pos, target_orientation=None):
        """逆运动学计算"""
        try:
            x, y, z = target_pos
            print(f"目标位置: x={x:.1f}, y={y:.1f}, z={z:.1f}")
            
            # 1. 计算底座角度 theta1
            theta1 = np.arctan2(y, x)
            print(f"底座角度 theta1={np.degrees(theta1):.1f}°")
            
            # 2. 计算在底座平面的投影距离
            r = np.sqrt(x*x + y*y)
            z = z - self.l1  # 减去底座高度
            print(f"投影距离 r={r:.1f}, 相对高度 z={z:.1f}")
            
            # 3. 考虑手腕的影响
            if target_orientation is not None:
                pitch = target_orientation[1]
                r = r - self.l4 * np.cos(pitch)
                z = z - self.l4 * np.sin(pitch)
                print(f"考虑手腕后: r={r:.1f}, z={z:.1f}")
            
            # 4. 计算大臂和小臂的角度
            L = np.sqrt(r*r + z*z)  # 从肩关节到手腕的距离
            if L > (self.l2 + self.l3):
                raise ValueError(f"目标点超出工作空间: {L:.1f} > {self.l2 + self.l3}")
            
            # 使用余弦定理计算大臂角度
            cos_theta2 = (L*L + self.l2*self.l2 - self.l3*self.l3)/(2*self.l2*L)
            if abs(cos_theta2) > 1:
                raise ValueError(f"无法到达目标点: cos_theta2={cos_theta2:.4f}")
            
            alpha = np.arccos(cos_theta2)
            beta = np.arctan2(z, r)
            
            # 选择肘关节向上的解
            theta2 = beta + alpha
            
            # 5. 计算小臂角度
            cos_theta3 = (self.l2*self.l2 + self.l3*self.l3 - L*L)/(2*self.l2*self.l3)
            if abs(cos_theta3) > 1:
                raise ValueError(f"无法到达目标点: cos_theta3={cos_theta3:.4f}")
            
            theta3 = np.arccos(cos_theta3) - np.pi  # 保持与大臂的相对角度
            
            # 6. 计算手腕角度
            if target_orientation is not None:
                theta4 = pitch - theta2 - theta3
            else:
                theta4 = -theta2 - theta3  # 保持末端垂直
            
            # 7. 手腕旋转角度
            theta5 = target_orientation[2] if target_orientation is not None else 0
            
            print(f"关节角度: theta1={np.degrees(theta1):.1f}°, "
                f"theta2={np.degrees(theta2):.1f}°, "
                f"theta3={np.degrees(theta3):.1f}°, "
                f"theta4={np.degrees(theta4):.1f}°, "
                f"theta5={np.degrees(theta5):.1f}°")
            
            return np.array([theta1, theta2, theta3, theta4, theta5])
            
        except Exception as e:
            print(f"逆运动学计算错误: {str(e)}")
            raise ValueError(f"逆运动学计算失败: {str(e)}")
        
    def inverse_kinematics_gripper(self, target_pos, target_orientation=None, gripper_width=20):
        """#基于抓夹左右两端中心点的逆运动学计算
        Args:
            target_pos: [x, y, z] 抓夹左右两端中心点的目标位置(mm)
            target_orientation: [roll, pitch, yaw] 目标姿态(弧度)，None则保持垂直
            gripper_width: 抓夹张开宽度(mm)，默认20mm
        Returns:
            angles: [theta1~theta5] 关节角度(弧度)
            gripper_pwm: 抓夹PWM值
        """
        try:
            x, y, z = target_pos
            print(f"抓夹中心目标位置: x={x:.1f}, y={y:.1f}, z={z:.1f}")
            
            # 1. 计算底座旋转角度
            theta1 = np.arctan2(y, x)
            
            # 2. 计算手腕位置
            if target_orientation is not None:
                pitch = target_orientation[1]  # 俯仰角
                roll = target_orientation[2]   # 旋转角
                
                # 先计算手腕末端位置（比目标点往回偏移l5的距离）
                end_effector_x = x - self.l5 * np.cos(pitch) * np.cos(theta1)
                end_effector_y = y - self.l5 * np.cos(pitch) * np.sin(theta1)
                end_effector_z = z - self.l5 * np.sin(pitch)
                
                
            else:
                # 如果没有指定姿态，假设保持垂直
                # 先往回偏移l5
                end_effector_x = x - self.l5 * np.cos(theta1)
                end_effector_y = y - self.l5 * np.sin(theta1)
                end_effector_z = z
                
            # 3. 使用原有的逆运动学计算手腕位置对应的关节角度
            angles = self.inverse_kinematics([end_effector_x, end_effector_y, end_effector_z], target_orientation)
            
            # 4. 计算抓夹PWM值
            gripper_pwm = int(1500 + (20 - gripper_width) * 1000 / 40)
            gripper_pwm = max(500, min(2500, gripper_pwm))
            
            print(f"抓夹张开宽度: {gripper_width:.1f}mm, PWM值: {gripper_pwm}")
            
            return angles, gripper_pwm
            
        except Exception as e:
            print(f"抓夹逆运动学计算错误: {str(e)}")
            raise ValueError(f"抓夹逆运动学计算失败: {str(e)}")
            '''
            #1. 计算底座角度 theta1
            if x == 0:
                theta1 = 0
            else:
                theta1 = np.arctan2(y, x)
            print(f"底座角度 theta1={np.degrees(theta1):.1f}°")
            
            cost = np.cos(pitch_rad)
            sint = np.sin(pitch_rad)
            #2. 计算在底座平面的投影距离
            pitch_rad = target_pitch * np.pi / 180.0  # 转换为弧度
            r=np.sqrt(x*x+y*y)- self.l5 * np.cost
            z=z-self.l1-self.l5 * np.sint  #减去底座高度
            print(f"投影距离 r={r:.1f},相对高度 z={z:.1f}")
            
            #3.检查是否超过范围
            L=np.sqrt(r*r+z*z)
            if L>(self.l2+self.l3+self.l4):
                raise ValueError(f"目标超过工作范围：{L:.1f}>{self.l2+self.l3+self.l4}")

            #4.计算大臂的角度theta2
            m=self.l4 *cost-r
            n=self.l4*sint-z
            a=m*m+n*n
            k=0.5*(self.l2*self.l2-self.l3*self.l3-m*m-n*n)/self.l3
            b=-2*n*k
            c=k*k-m*m
            
            theta2=np.arcsin(0.5*(-b-np.sqrt(b*b-4*a*c))/a)#弧度制
            if theta2 > 180.0 or theta2 < 0.0:
                raise ValueError("大臂角度超出范围")
                
            #5.计算小臂的角度theta3(theta2+theta3)
            k=
            '''
            
        
    def pwm_to_angle(self,pwm_values):
        """PWM值转角度"""
        angles=[]
        for i,pwm in enumerate(pwm_values[:5]):
            if i==0:    #底座
                angle=(pwm-1500)/1000*np.pi
            elif i==1:  #大臂
                angle=(pwm-1500)/1000*np.pi+np.pi/2   #减去90度偏移
            elif i==2:  #小臂
                angle=(pwm-1500)/1000*np.pi
            elif i==3:  #手腕俯仰
                angle=(pwm-1500)/1000*np.pi   
            elif i==4:  #手腕旋转
                angle=(pwm-1500)/1000*np.pi#减去90度偏移
            angles.append(angle)
        return np.array(angles)

    def angle_to_pwm(self,angles):
        """角度转PWM值"""
        pwm_values=[]
        for i,angle in enumerate(angles):
            if i==0:    #底座
                pwm=int(angle*1000/np.pi+1500)
            elif i==1:  #大臂
                pwm=int((angle-np.pi/2)*1000/np.pi+1500)  #加上90度偏移
            elif i==2:  #小臂
                pwm=int((angle)*1000/np.pi+1500)
            elif i==3:  #手腕俯仰
                pwm=int((angle)*1000/np.pi+1500)  #加上90度偏移
            elif i==4:  #手腕旋转
                pwm=int((angle)*1000/np.pi+1500)
            pwm_values.append(max(500,min(2500,pwm)))
        return pwm_values

#控制器窗口
class RoboticArmGUI(QMainWindow):

    controller_signal=pyqtSignal(str) #控制器发送给主窗口的信号

    def __init__(self):
        super().__init__()
        self.serial_port=None
        self.kinematics=RoboticArmKinematics()  # 添加运动学实例
        self.trajectory_points=[]
        self.current_position={'x': 0, 'y': 0, 'z': 0, 'pitch': 0, 'roll': 0}
        self.is_connected=False #指示是否已连接
        self.is_realtimecontrol=False #指示是否实时响应，默认不实时

        #信息显示的文本格式
        self.blue_format=QTextCharFormat() 
        self.blue_format.setForeground(QColor("blue")) #蓝色
        self.green_format=QTextCharFormat() 
        self.green_format.setForeground(QColor("green")) #绿色
        self.black_format=QTextCharFormat() 
        self.black_format.setForeground(QColor("black")) #黑色

        self.initUI()

    def initUI(self):
        self.setWindowTitle('5DOF机械臂控制系统') #设置标题
        self.setGeometry(300, 300, 1200, 1000) #设置窗口位置和大小

        #主面板
        central_widget=QWidget() #主面板
        self.setCentralWidget(central_widget) #设置中心部件
        layout=QHBoxLayout(central_widget) #水平布局

        #左侧面板
        left_panel=QWidget() #左侧面板
        left_layout=QVBoxLayout(left_panel) #设置左侧面板的布局，垂直布局

        ##串口设置组
        serial_group=QGroupBox('串口设置') #串口设置的功能组
        serial_layout=QHBoxLayout() #水平布局

        ###可选串口
        self.port_combo=QComboBox() #串口列表下拉框
        self.refresh_ports() #刷新一下

        ###波特率
        self.baud_combo=QComboBox() #波特率下拉框
        self.baud_combo.addItems(['9600', '115200']) #添加选项
        self.baud_combo.setCurrentText('115200') #设置默认波特率
        
        ###连接按钮
        self.connect_btn=QPushButton('连接')
        self.connect_btn.clicked.connect(self.toggle_connection) #绑定回调函数
        
        ###刷新按钮
        refresh_btn=QPushButton('刷新')
        refresh_btn.clicked.connect(self.refresh_ports) #绑定回调函数

        serial_layout.addWidget(QLabel('串口:')) #放置串口设置组部件到布局
        serial_layout.addWidget(self.port_combo)
        serial_layout.addWidget(QLabel('波特率:'))
        serial_layout.addWidget(self.baud_combo)
        serial_layout.addWidget(self.connect_btn)
        serial_layout.addWidget(refresh_btn)
        serial_group.setLayout(serial_layout) #为串口设置组设置垂直布局

        ##关节控制组
        joint_group=QGroupBox('关节控制') #关节控制的功能组
        joint_layout=QVBoxLayout() #垂直布局

        joint_names=['底座旋转', '大臂', '小臂', '手腕', '爪子旋转', '爪子开合'] #关节控制名称
        self.joint_inputs=[] #所有输入框
        self.joint_sliders=[] #所有滑块
        self.joint_values=[] #所有显示值
        
        ###关节控制
        jointcontrol_layout=QGridLayout() #表格布局
        for i,name in enumerate(joint_names):
            label=QLabel(name+':') #对应的控制名称
            
            ####输入框
            value_input=QLineEdit('1500') #单行输入框，默认为1500
            value_input.setFixedWidth(60) #输入框宽度
            self.joint_inputs.append(value_input)
            
            ####滑块
            slider=QSlider(Qt.Horizontal) #水平滑块
            slider.setRange(500, 2500) #设置滑块最大最小值
            slider.setValue(1500) #设置默认值1500
            self.joint_sliders.append(slider)
            
            ####当前数值显示
            value_label=QLabel('1500') #标签,默认为1500
            value_label.setFixedWidth(60) #设置固定长度,防止表格布局变来变去
            self.joint_values.append(value_label)
            
            slider.valueChanged.connect(lambda v, i=i: self.on_slider_changed(i, v)) #绑定回调函数,滑块移动触发
            value_input.returnPressed.connect(lambda i=i: self.on_input_changed(i)) #绑定回调函数,输入框输入完触发
            
            jointcontrol_layout.addWidget(label,i,0) #放置每个关节控制组件到水平布局
            jointcontrol_layout.addWidget(value_input,i,1)
            jointcontrol_layout.addWidget(slider,i,2)
            jointcontrol_layout.addWidget(value_label,i,3)
        
        joint_layout.addLayout(jointcontrol_layout) #将该行水平布局添加到关节控制的垂直布局中

        ###控制按钮
        control_layout=QHBoxLayout() #两个按钮对应的水平布局

        ####实时控制按钮
        self.realtime_control_btn=QPushButton('触发控制') #默认触发控制
        self.realtime_control_btn.clicked.connect(self.toggle_controlmode) #绑定回调函数

        ####发送所有按钮
        self.send_all_btn=QPushButton('发送所有')
        self.send_all_btn.clicked.connect(self.send_command) #绑定回调函数
        
        ####复位按钮
        reset_btn=QPushButton('复位')
        reset_btn.clicked.connect(self.reset_joints) #绑定回调函数
        
        control_layout.addWidget(QLabel('当前的控制模式:'))#放置按钮到水平布局
        control_layout.addWidget(self.realtime_control_btn)
        control_layout.addWidget(self.send_all_btn) 
        control_layout.addWidget(reset_btn)
        joint_layout.addLayout(control_layout) #将该水平布局添加到关节控制的垂直布局




        ###关节轨迹组(不涉及直线) 25.1.3
        joint_trajectory_group=QGroupBox('需要到达的中间点') #关节轨迹的功能组
        joint_trajectory_layout=QVBoxLayout() #垂直布局



        ####关节空间的中间点表格
        self.joint_trajectory_table=QTableWidget(0, 6) #表格，存放中间点
        self.joint_trajectory_table.setMinimumHeight(150) #设置表高度,多显示几行
        self.joint_trajectory_table.setHorizontalHeaderLabels(['底座', '大臂', '小臂', '手腕', '爪子旋转', '爪子开合']) #水平表头

        ####关节轨迹按钮
        joint_trajectory_buttons_layout=QHBoxLayout() #水平布局
        
        #####添加路点按钮
        add_joint_point_btn=QPushButton('添加中间点')
        add_joint_point_btn.clicked.connect(self.add_joint_trajectory_point) #绑定回调函数
        
        #####清除路点按钮
        clear_joint_points_btn=QPushButton('清除路点')
        clear_joint_points_btn.clicked.connect(self.clear_joint_trajectory_points) #绑定回调函数
        
        #####执行轨迹按钮
        execute_joint_trajectory_btn=QPushButton('执行轨迹')
        execute_joint_trajectory_btn.clicked.connect(self.execute_joint_trajectory) #绑定回调函数

        joint_trajectory_buttons_layout.addWidget(add_joint_point_btn) #将3个按钮添加到按钮的水平布局中
        joint_trajectory_buttons_layout.addWidget(clear_joint_points_btn)
        joint_trajectory_buttons_layout.addWidget(execute_joint_trajectory_btn)

        joint_trajectory_layout.addWidget(self.joint_trajectory_table) #添加到关节轨迹组布局
        joint_trajectory_layout.addLayout(joint_trajectory_buttons_layout)
        
        joint_trajectory_group.setLayout(joint_trajectory_layout) #为关节轨迹组设置垂直布局
        joint_layout.addWidget(joint_trajectory_group) #将关节轨迹组布局添加到关节控制组的垂直布局中
        
        joint_group.setLayout(joint_layout) #为关节控制组设置布局为该垂直布局






        ##笛卡尔空间控制组
        cartesian_group=QGroupBox('笛卡尔空间控制') #笛卡尔空间控制的功能组
        cartesian_layout=QGridLayout() #网格布局

        self.cartesian_inputs={}    #笛卡尔输入
        self.cartesian_labels={}    #笛卡尔空间显示

        ###输入框和当前位置显示
        for i, coord in enumerate(['x', 'y', 'z', 'pitch', 'roll']):
            ###标签
            label=QLabel(f'{coord.upper()}:')
            
            ###输入框
            input_field=QLineEdit('0.0')
            self.cartesian_inputs[coord]=input_field
            
            ###当前位置显示
            current_label=QLabel('0.0')
            current_label.setFixedWidth(60) #设置固定长度,防止表格布局变来变去
            self.cartesian_labels[coord]=current_label
            
            cartesian_layout.addWidget(label, i, 0) #放置每个关节控制组件到网格布局
            cartesian_layout.addWidget(input_field, i, 1)
            cartesian_layout.addWidget(QLabel('当前:'), i, 2)
            cartesian_layout.addWidget(current_label, i, 3)
       
        ###执行按钮
        move_btn=QPushButton('执行运动')
        move_btn.clicked.connect(self.move_to_cartesian)
        cartesian_layout.addWidget(move_btn, 5, 0, 1, 4) #从5,0位置开始,占1行4列

        cartesian_group.setLayout(cartesian_layout) #为笛卡尔空间控制组设置布局为网格布局

        ##轨迹规划组
        trajectory_group=QGroupBox('轨迹规划')
        trajectory_layout=QVBoxLayout() #垂直布局

        ###轨迹点表格
        self.trajectory_table=QTableWidget(0, 6)
        self.trajectory_table.setMinimumHeight(150) #设置表高度,多显示几行
        self.trajectory_table.setHorizontalHeaderLabels(['X', 'Y', 'Z', 'Pitch', 'Roll', '时间']) #水平表头

        ###轨迹规划按钮
        trajectory_buttons_layout=QHBoxLayout() #水平布局
        
        ####添加路点按钮
        add_point_btn=QPushButton('添加路点')
        add_point_btn.clicked.connect(self.add_trajectory_point) #绑定回调函数
        
        ####清除路点按钮
        clear_points_btn=QPushButton('清除路点')
        clear_points_btn.clicked.connect(self.clear_trajectory_points) #绑定回调函数
        
        ####执行轨迹按钮
        execute_trajectory_btn=QPushButton('执行轨迹')
        execute_trajectory_btn.clicked.connect(self.execute_trajectory) #绑定回调函数
        
        trajectory_buttons_layout.addWidget(add_point_btn) #将3个按钮添加到按钮的水平布局中
        trajectory_buttons_layout.addWidget(clear_points_btn)
        trajectory_buttons_layout.addWidget(execute_trajectory_btn)

        trajectory_layout.addWidget(self.trajectory_table) #把轨迹规划的两个组件添加到垂直布局
        trajectory_layout.addLayout(trajectory_buttons_layout)

        trajectory_group.setLayout(trajectory_layout) #将这个垂直布局设置为轨迹规划组的布局

        ##信息传递组
        signal_group=QGroupBox('信息传递') #和视觉检测窗口进行信息传递的功能组
        signal_layout=QGridLayout() #表格布局

        ###信号显示文本框
        self.signal_log=QTextEdit() #多行文本框，展示接受和发送信息
        self.signal_log.setReadOnly(True) #设置只读
        self.signal_log.setMinimumHeight(150) #设置表高度,多显示几行
        self.signal_log.append('已初始化')

        ###指令下拉框
        self.signal_combo_label=QLabel('可用指令:')
        self.signal_combo_label.setFixedWidth(80) #最大宽度
        self.signal_combo=QComboBox()
        self.signal_combo.addItems(['测试信息1','测试信息2']) #添加选项
        self.signal_combo.setCurrentText('测试信息1') #设置默认指令

        ###窗口连接按钮
        self.signal_send_btn=QPushButton('发送')
        self.signal_send_btn.clicked.connect(self.send_choose_signal) #绑定回调函数

        signal_layout.addWidget(self.signal_log,0,0,1,4) #把信息传递组的几个组件添加到该垂直布局
        signal_layout.addWidget(self.signal_combo_label,1,0)
        signal_layout.addWidget(self.signal_combo,1,1,1,2)
        signal_layout.addWidget(self.signal_send_btn,1,3)

        signal_group.setLayout(signal_layout) #将这个表格布局设置为信息传递组的布局

        #添加所有组件到左侧面板
        left_layout.addWidget(serial_group)
        left_layout.addWidget(joint_group)
        left_layout.addWidget(cartesian_group)
        left_layout.addWidget(trajectory_group)
        left_layout.addWidget(signal_group)

        #右侧面板
        #创建右侧3D视图
        self.figure=plt.figure()
        self.canvas=FigureCanvas(self.figure)
        self.ax=self.figure.add_subplot(111, projection='3d')
        self.init_3d_view() #初始化3D视图

        #添加左右面板到主布局
        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(self.canvas, stretch=2)

        #状态栏
        self.statusBar().showMessage('未连接')

        #为保证正常显示3d视图和笛卡尔坐标,先更新一次
        self.update_3d_view() #更新3d视图
        self.update_cartesian_display() #更新笛卡尔坐标

    def closeEvent(self, event):
        """关闭窗口时的处理函数"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("串口已关闭")
        self.destroyed.emit() #发出销毁信号.没有销毁
        self.destroy()

    ############################### 串口设置 ###############################
    def refresh_ports(self):
        """刷新可用串口列表"""
        self.port_combo.clear() #清空可选串口
        ports=serial.tools.list_ports.comports() #获取可用串口
        for port in ports:
            self.port_combo.addItem(port.device) #添加至下拉框

    def toggle_connection(self):
        """切换串口连接状态"""
        if not self.is_connected: #未连接串口
            try:
                port=self.port_combo.currentText() #从可用串口下拉框获取串口
                baud=int(self.baud_combo.currentText()) #从波特率下拉框获取波特率
                self.serial_port=serial.Serial(port, baud, timeout=1) #打开串口
                self.is_connected=True
                self.connect_btn.setText('断开') #更新连接按钮文本
                self.statusBar().showMessage(f'已连接到 {port}') #更新最下方状态栏文本
                QMessageBox.information(self, '连接状态', '串口连接成功！') #弹框
                #为了安全性,复位
                self.reset_joints() #重置
            except Exception as e: #串口打开失败
                QMessageBox.warning(self, '连接错误', f'无法连接串口：{str(e)}') #弹框
        else: #已连接
            if self.serial_port: #若当前已连接串口，先关闭
                self.serial_port.close()
            self.is_connected=False
            self.connect_btn.setText('连接') #更新连接按钮文本
            self.statusBar().showMessage('未连接') #更新状态栏
    ############################### 串口设置 ###############################

    ############################### 关节控制 ###############################
    def on_slider_changed(self, joint_index, value):
        """滑块值改变时的处理函数"""
        self.joint_values[joint_index].setText(str(value)) #同步改变标签显示值
        self.joint_inputs[joint_index].setText(str(value)) #同步改变输入框输入值
        if self.is_realtimecontrol: #如果是实时控制状态,才发送指令
            self.send_single_command(joint_index, value) #控制机械臂对应关节
            self.update_3d_view() #更新3d视图
            self.update_cartesian_display() #更新笛卡尔坐标
        
    def on_input_changed(self, joint_index):
        """输入框值改变时的处理函数"""
        try:
            value=int(self.joint_inputs[joint_index].text()) #获取输入框的值
            value=max(500, min(2500, value)) #用上下界截断，防止超出范围
            self.joint_sliders[joint_index].setValue(value) #同步改变滑块的值,会自动调用滑块改变函数
        except ValueError:
            QMessageBox.warning(self, '输入错误', '请输入有效的数值(500-2500)')

    def toggle_controlmode(self):
        """切换控制模式:实时/触发"""
        if self.is_realtimecontrol: #实时控制中
            self.is_realtimecontrol=False
            self.realtime_control_btn.setText('触发控制') #更新按钮文本
            self.send_all_btn.setEnabled(True) #触发控制下发送命令按钮才有用
        else: #触发控制中
            self.is_realtimecontrol=True
            self.realtime_control_btn.setText('实时控制') #更新连接按钮文本
            self.send_all_btn.setEnabled(False) #实时控制下发送命令按钮没有用途，无法按下
            #本来想直接同步当前状态，但为了安全性着想，进行重置
            self.reset_joints()

    def send_single_command(self, joint_index, value):
        """发送单个关节控制命令"""
        if self.is_connected and self.serial_port and self.serial_port.is_open:
            try:
                command=f"#{joint_index:03d}P{value:04d}T0500!"  #修改时间为500
                self.serial_port.write(command.encode()) #发送指令
                print(f"发送命令: {command}")
            except Exception as e:
                print(f"发送命令失败: {str(e)}")

    def send_all_commands(self):
        """发送所有关节控制命令"""
        #if not self.is_connected:
        #    QMessageBox.warning(self, '错误', '请先连接串口！')
        #    return
        if self.is_connected and self.serial_port and self.serial_port.is_open:
            try:
                command="{"  #开始组合命令
                for i, slider in enumerate(self.joint_sliders):
                    value=slider.value()
                    command+=f"#{i:03d}P{value:04d}T0500!"  #修改时间为500
                command+="}"  #结束组合命令
                self.serial_port.write(command.encode())
                print(f"发送命令: {command}")
            except Exception as e:
                QMessageBox.warning(self, '发送错误', f'发送命令失败：{str(e)}')

    def send_command(self):
        """发送指令按钮"""
        self.send_all_commands() #控制机械臂
        self.update_3d_view() #更新3d视图
        self.update_cartesian_display() #更新笛卡尔坐标

    def reset_joints(self):
        """复位所有关节到竖直状态"""
        #设置竖直状态的PWM值
        reset_values=[1500]*6
        
        for i, value in enumerate(reset_values):
            self.joint_sliders[i].setValue(value) #重置滑块,会自动调用滑块改变函数
        if not self.is_realtimecontrol: #如果是触发控制,滑块改变函数不会重置其他东西,手动重置
            self.send_command() #控制机械臂、更新3d视图、更新笛卡尔坐标
    ############################### 关节控制 ###############################

    ############################## 笛卡尔控制 ##############################
    def move_to_cartesian(self):
        """移动到笛卡尔坐标位置"""
        try:
            x=float(self.cartesian_inputs['x'].text()) #获取输入框的值
            y=float(self.cartesian_inputs['y'].text())
            z=float(self.cartesian_inputs['z'].text())
            pitch=np.radians(float(self.cartesian_inputs['pitch'].text()))
            roll=np.radians(float(self.cartesian_inputs['roll'].text()))
            
            #打印调试信息
            print(f"目标位置: x={x:.1f}, y={y:.1f}, z={z:.1f}, pitch={np.degrees(pitch):.1f}°, roll={np.degrees(roll):.1f}°")
                
            target_pos=np.array([x, y, z])
            target_orientation=np.array([0, pitch, roll])
            
            #计算逆运动学
            joint_angles,gripper_pwm=self.kinematics.inverse_kinematics_gripper(target_pos, target_orientation,gripper_width=20)
            
            #转换为PWM值
            pwm_values=self.kinematics.angle_to_pwm(joint_angles)
            
            #设置关节值
            for i, pwm in enumerate(pwm_values):
                self.joint_sliders[i].setValue(pwm)
            self.joint_sliders[5].setValue(gripper_pwm)
            #发送命令
            self.send_command()
            
        except ValueError as e:
            QMessageBox.warning(self, '错误', str(e))

    def calculate_joint_positions(self, joint_values):
        """计算机械臂各关节位置"""
        angles = self.kinematics.pwm_to_angle(joint_values)
        positions, orientation, pitch, roll = self.kinematics.forward_kinematics(angles)
        
        p4 = positions[-1]
        
        # 计算手腕朝向
        # 考虑大臂、小臂和手腕的角度累积
        total_angle = angles[1] + angles[2] + angles[3] + np.pi/2   # 竖直平面内的角度
        base_angle = angles[0]  # 水平面内的旋转角度
        
        # 计算手腕朝向向量（在世界坐标系中）
        wrist_dir = np.array([
            np.cos(base_angle) * np.sin(total_angle),  # x分量
            np.sin(base_angle) * np.sin(total_angle),  # y分量
            -np.cos(total_angle)                        # z分量
        ])
        
        # 计算水平基准向量（始终垂直于手腕朝向）
        base_right = np.array([-np.sin(base_angle), np.cos(base_angle), 0])
        
        # 确保向量正交
        up_vector = np.cross(wrist_dir, base_right)
        up_vector = up_vector / np.linalg.norm(up_vector)  # 归一化
        base_right = np.cross(up_vector, wrist_dir)  # 重新计算以确保正交
        base_right = base_right / np.linalg.norm(base_right)  # 归一化
        
        # 根据第5个关节角度计算抓夹旋转
        angle5 = angles[4]
        gripper_right = base_right * np.cos(angle5) + up_vector * np.sin(angle5)
        
        # 计算爪子开合宽度
        gripper_width = 20
        if joint_values[5] != 1500:
            gripper_width = 20 - abs((joint_values[5] - 1500) / 1000 * 40)
        
        # 计算爪子两个端点的位置
        p5_left = (p4 + 
                gripper_right * (gripper_width/2) + 
                wrist_dir * self.kinematics.l5)
        p5_right = (p4 - 
                    gripper_right * (gripper_width/2) + 
                    wrist_dir * self.kinematics.l5)
        
        # 调试输出
        #print("p4 wrist_dir:", wrist_dir)
        #print("p4 base_right:", base_right)
        #print("p4 up_vector:", up_vector)
        
        return positions + [p5_left, p5_right]

    def update_cartesian_display(self):
        """更新笛卡尔坐标显示"""
        try:
            # 获取当前关节角度
            joint_values = [slider.value() for slider in self.joint_sliders]
            angles = self.kinematics.pwm_to_angle(joint_values)
            
            # 使用正运动学计算各个关节位置
            positions = self.calculate_joint_positions(joint_values)
            
            # 获取手腕末端位置和方向
            wrist_pos = positions[5]  # 手腕末端位置350
            #print(f"手腕末端位置：",wrist_pos)
            # 计算pitch角度（与水平面的夹角）
            pitch = angles[1] + angles[2] + angles[3]  # 大臂 + 小臂 + 手腕俯仰
            roll = angles[4]  # 手腕旋转角度
            
            # 计算抓取方向的单位向量
            direction = np.array([
                np.cos(pitch) * np.cos(angles[0]),  # x分量
                np.cos(pitch) * np.sin(angles[0]),  # y分量
                np.sin(pitch)                       # z分量
            ])
            
            # 计算抓夹中心点位置（从手腕末端沿抓取方向前进l5距离）
            gripper_center = wrist_pos + direction * self.kinematics.l5
            
            # 更新显示
            self.cartesian_labels['x'].setText(f"{gripper_center[0]:.1f}")
            self.cartesian_labels['y'].setText(f"{gripper_center[1]:.1f}")
            self.cartesian_labels['z'].setText(f"{gripper_center[2]:.1f}")
            self.cartesian_labels['pitch'].setText(f"{np.degrees(pitch):.1f}")
            self.cartesian_labels['roll'].setText(f"{np.degrees(roll):.1f}")
            
            # 更新当前位置字典（如果其他地方需要用到）
            self.current_position = {
                'x': gripper_center[0],
                'y': gripper_center[1],
                'z': gripper_center[2],
                'pitch': pitch,
                'roll': roll
            }
            
        except Exception as e:
            print(f"更新笛卡尔坐标显示错误: {str(e)}")
    ############################## 笛卡尔控制 ##############################

    ############################### 轨迹控制 ###############################
    def add_joint_trajectory_point(self):
        """添加关节空间轨迹点"""
        row=self.trajectory_table.rowCount() #获取现在的行数
        self.trajectory_table.insertRow(row) #在最后插入新的一行
        for i in range(6):
            value=self.cartesian_inputs[i].text() #获取笛卡尔输入框里的内容
            self.trajectory_table.setItem(row, i, QTableWidgetItem(value)) #设置新的一行内容

    def clear_joint_trajectory_points(self):
        """清除所有关节空间轨迹点"""
        self.trajectory_table.setRowCount(0)

    def execute_joint_trajectory(self):
        """执行关节空间轨迹规划"""
        if not self.is_connected:
            QMessageBox.warning(self, '错误', '请先连接串口！')
            return
            
        points=[]
        for row in range(self.trajectory_table.rowCount()):
            point=[]
            for col in range(6):
                item=self.trajectory_table.item(row, col)
                point.append(float(item.text() if item else 0))
            points.append(point)







    def add_trajectory_point(self):
        """添加轨迹点"""
        row=self.trajectory_table.rowCount() #获取现在的行数
        self.trajectory_table.insertRow(row) #在最后插入新的一行
        for i, coord in enumerate(['x', 'y', 'z', 'pitch', 'roll']):
            value=self.cartesian_inputs[coord].text() #获取笛卡尔输入框里的内容
            self.trajectory_table.setItem(row, i, QTableWidgetItem(value)) #设置新的一行内容
        self.trajectory_table.setItem(row, 5, QTableWidgetItem('1.0')) #时间默认为1

    def clear_trajectory_points(self):
        """清除所有轨迹点"""
        self.trajectory_table.setRowCount(0)

    def execute_trajectory(self):
        """执行轨迹规划"""
        if not self.is_connected:
            QMessageBox.warning(self, '错误', '请先连接串口！')
            return
            
        points=[]
        for row in range(self.trajectory_table.rowCount()):
            point=[]
            for col in range(6):
                item=self.trajectory_table.item(row, col)
                point.append(float(item.text() if item else 0))
            points.append(point)
            
        # TODO: 实现轨迹规划和执行逻辑
    ############################### 轨迹控制 ###############################
    
    ################################ 3D仿真 ################################
    def init_3d_view(self):
        """初始化3D视图"""
        self.ax.clear()
        self.ax.set_xlim([-300, 300])
        self.ax.set_ylim([-300, 300])
        self.ax.set_zlim([0, 400])
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.view_init(elev=20, azim=45)
        self.canvas.draw()

    def update_3d_view(self):
        """更新3D视图"""
        self.ax.clear()
        
        #获取当前关节角度
        joint_values=[slider.value() for slider in self.joint_sliders]
        
        #计算机械臂各关节位置
        points=self.calculate_joint_positions(joint_values)
        
        #绘制机械臂主体
        x_coords=[p[0] for p in points[:-2]]
        y_coords=[p[1] for p in points[:-2]]
        z_coords=[p[2] for p in points[:-2]]
        
        #绘制连杆
        self.ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2)
        #绘制关节点
        self.ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')
        
        #绘制爪子
        p4=points[-3]
        p5_left=points[-2]
        p5_right=points[-1]
        
        #中心线
        gripper_center = (p5_left + p5_right) / 2
        self.ax.plot([p4[0], gripper_center[0]], 
                    [p4[1], gripper_center[1]], 
                    [p4[2], gripper_center[2]], 'g--', linewidth=1)
        #绘制爪子线段
        self.ax.plot([p4[0], p5_left[0]], [p4[1], p5_left[1]], [p4[2], p5_left[2]], 'g-', linewidth=2)
        self.ax.plot([p4[0], p5_right[0]], [p4[1], p5_right[1]], [p4[2], p5_right[2]], 'g-', linewidth=2)
        
        # 添加点标注
        #self.ax.text(p4[0], p4[1], p4[2], 'p4')
        #self.ax.text(p5_left[0], p5_left[1], p5_left[2], 'p5_left')
        #self.ax.text(p5_right[0], p5_right[1], p5_right[2], 'p5_right')
        
        #设置视图范围
        self.ax.set_xlim([-300, 300])
        self.ax.set_ylim([-300, 300])
        self.ax.set_zlim([0, 400])
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        
        #设置视角
        self.ax.view_init(elev=30, azim=45)
        
        self.canvas.draw()
    ################################ 3D仿真 ################################

    ############################### 通信相关 ###############################
    def send_signal(self,signal):
        """发送信号给主窗口"""
        self.controller_signal.emit(signal) #发送信息

        self.show_signal(signal=signal,mode=1) #展示信息

    def receive_signal(self,signal):
        """接受主窗口转发的视觉检测信号"""
        self.show_signal(signal=signal,mode=2) #展示信息

    def show_signal(self,signal,mode=0):
        """将信息展示到文本框中"""
        if mode==0: #系统提示文本
            cursor=self.signal_log.textCursor() #获取当前文本光标
            cursor.insertBlock() #换行
            cursor.insertText(signal,self.black_format) #具体信息设为黑色加粗
            self.signal_log.setTextCursor(cursor)
        elif mode==1: #控制器信息
            cursor=self.signal_log.textCursor() #获取当前文本光标
            cursor.insertBlock() #换行
            cursor.insertText('[controller]:',self.blue_format) #控制器设为蓝色
            cursor.insertText(signal,self.black_format) #具体信息设为黑色
            self.signal_log.setTextCursor(cursor)
        elif mode==2: #视觉检测信息
            cursor=self.signal_log.textCursor() #获取当前文本光标
            cursor.insertBlock() #换行
            cursor.insertText('[visual]:',self.green_format) #视觉检测设为绿色
            cursor.insertText(signal,self.black_format) #具体信息设为黑色
            self.signal_log.setTextCursor(cursor)

    def send_choose_signal(self):
        """发送选择指令"""
        signal=self.signal_combo.currentText() #获取当前所选指令
        self.send_signal(signal)
    ############################### 通信相关 ###############################


#主函数
def main():
    app=QApplication(sys.argv) #负责应用事件循环
    window=RoboticArmGUI() #自定义的窗口
    window.show() #展示窗口
    sys.exit(app.exec_()) #启动事件循环，满足退出条件的时候就结束事件循环

if __name__ == '__main__':
    main()