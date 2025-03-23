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
from scipy.interpolate import interp1d

from src.kinematics import RoboticArmKinematics

#控制器窗口
class RoboticArmGUI(QMainWindow):

    controller_signal=pyqtSignal(str) #控制器发送给主窗口的信号

    def __init__(self):
        super().__init__()
        #串口相关
        self.serial_port=None
        self.is_connected=False #指示是否已连接

        #机械臂运动学与位置相关
        self.kinematics=RoboticArmKinematics()  #添加运动学实例
        self.current_joint_engine=[1500,1500,1500,1500,1500,1500] #关节位置 舵机
        self.current_joint_angle={'0-5':[0,0,0,0,0],'6':'张开'} #关节位置 角度
        self.current_position={'x,y,z':[0,0,0], 'yaw': 0, 'roll': 0} #笛卡尔空间位置
        
        self.is_trajectory=False #机械臂轨迹执行中
        self.is_pause=False #机械臂是否暂停
        self.last_command='' #存储上一条指令

        #关节空间控制
        self.is_realtimecontrol=False #指示是否实时响应，默认不实时

        #笛卡尔空间控制
        self.is_linetrajectory=False #指示是否使用笛卡尔直线规划，默认为关节曲线规划

        #轨迹相关
        self.trajectory=[] #存储待执行关节轨迹
        self.current_trajectory_index=0 #存储当前执行的轨迹点索引

        #视觉标定
        self.calibration_count=0 #初始化标定点计数器
        
        #信息显示的文本格式
        self.blue_format=QTextCharFormat() 
        self.blue_format.setForeground(QColor("blue")) #蓝色
        self.green_format=QTextCharFormat() 
        self.green_format.setForeground(QColor("green")) #绿色
        self.black_format=QTextCharFormat() 
        self.black_format.setForeground(QColor("black")) #黑色

        self.initUI()

    def initUI(self):
        '''初始化界面'''
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
        joint_group=QGroupBox('关节空间控制') #关节控制的功能组
        joint_layout=QVBoxLayout() #垂直布局

        joint_names=['底座旋转', '大臂', '小臂', '手腕', '爪子旋转', '爪子开合'] #关节控制名称
        self.joint_inputs=[] #所有输入框
        self.joint_sliders=[] #所有滑块
        self.joint_values_engine=[] #所有舵机位置显示值
        self.joint_values_angle=[] #所有角度显示值
        
        ###关节控制
        jointcontrol_layout=QGridLayout() #表格布局
        jointcontrol_layout.addWidget(QLabel('舵机位置'),0,3) #表头
        jointcontrol_layout.addWidget(QLabel('当前角度'),0,4) #表头
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
            
            ####当前舵机位置数值显示 跟着滑块动
            value_label_engine=QLabel('1500') #标签,默认为1500
            value_label_engine.setFixedWidth(60) #设置固定长度,防止表格布局变来变去
            self.joint_values_engine.append(value_label_engine)

            ####当前角度数值显示 不跟着滑块动，跟着笛卡尔更新来
            value_label_angle=QLabel('0') #标签,默认为1500
            value_label_angle.setFixedWidth(60) #设置固定长度,防止表格布局变来变去
            self.joint_values_angle.append(value_label_angle)
            
            slider.valueChanged.connect(lambda v, i=i: self.on_slider_changed(i, v)) #绑定回调函数,滑块移动触发
            value_input.returnPressed.connect(lambda i=i: self.on_input_changed(i)) #绑定回调函数,输入框输入完触发
            
            jointcontrol_layout.addWidget(label,i+1,0) #放置每个关节控制组件到水平布局
            jointcontrol_layout.addWidget(value_input,i+1,1)
            jointcontrol_layout.addWidget(slider,i+1,2)
            jointcontrol_layout.addWidget(value_label_engine,i+1,3)
            jointcontrol_layout.addWidget(value_label_angle,i+1,4)
        
        joint_layout.addLayout(jointcontrol_layout) #将该行表格布局添加到关节控制的垂直布局中

        ###控制按钮
        joint_control_layout=QHBoxLayout() #两个按钮对应的水平布局

        ####实时控制按钮
        self.realtime_control_btn=QPushButton('触发控制') #默认触发控制
        self.realtime_control_btn.clicked.connect(self.toggle_controlmode) #绑定回调函数

        ####发送所有按钮
        self.send_all_btn=QPushButton('发送所有')
        self.send_all_btn.clicked.connect(self.send_command) #绑定回调函数
        
        ####复位按钮
        reset_btn=QPushButton('复位')
        reset_btn.clicked.connect(self.reset_joints) #绑定回调函数
        
        joint_control_layout.addWidget(QLabel('当前的控制模式:'))#放置按钮到水平布局
        joint_control_layout.addWidget(self.realtime_control_btn)
        joint_control_layout.addWidget(self.send_all_btn) 
        joint_control_layout.addWidget(reset_btn)
        joint_layout.addLayout(joint_control_layout) #将该水平布局添加到关节控制的垂直布局

        ###关节轨迹组(不涉及直线) 25.1.3
        joint_trajectory_group=QGroupBox('需要到达的中间点') #关节轨迹的功能组
        joint_trajectory_layout=QVBoxLayout() #垂直布局

        ####关节空间的中间点表格
        self.joint_trajectory_table=QTableWidget(0, 6) #表格，存放中间点
        self.joint_trajectory_table.setMinimumHeight(120) #设置表高度,多显示几行
        self.joint_trajectory_table.setHorizontalHeaderLabels(['底座', '大臂', '小臂', '手腕', '爪子旋转', '爪子开合']) #水平表头

        ####关节轨迹按钮
        joint_trajectory_buttons_layout=QHBoxLayout() #水平布局
        
        #####添加中间点按钮
        add_joint_point_btn=QPushButton('添加中间点')
        add_joint_point_btn.clicked.connect(self.add_joint_trajectory_point) #绑定回调函数
        
        #####清除中间点按钮
        clear_joint_points_btn=QPushButton('清除中间点')
        clear_joint_points_btn.clicked.connect(self.clear_joint_trajectory_points) #绑定回调函数
        
        #####执行轨迹按钮
        self.execute_joint_trajectory_btn=QPushButton('执行轨迹')
        self.execute_joint_trajectory_btn.clicked.connect(self.execute_joint_trajectory) #绑定回调函数

        joint_trajectory_buttons_layout.addWidget(add_joint_point_btn) #将3个按钮添加到按钮的水平布局中
        joint_trajectory_buttons_layout.addWidget(clear_joint_points_btn)
        joint_trajectory_buttons_layout.addWidget(self.execute_joint_trajectory_btn)

        joint_trajectory_layout.addWidget(self.joint_trajectory_table) #添加到关节轨迹组布局
        joint_trajectory_layout.addLayout(joint_trajectory_buttons_layout)
        
        joint_trajectory_group.setLayout(joint_trajectory_layout) #为关节轨迹组设置垂直布局
        joint_layout.addWidget(joint_trajectory_group) #将关节轨迹组布局添加到关节控制组的垂直布局中
        
        joint_group.setLayout(joint_layout) #为关节控制组设置布局为该垂直布局

        ##笛卡尔空间控制组
        cartesian_group=QGroupBox('笛卡尔空间控制') #笛卡尔空间控制的功能组
        cartesian_layout=QVBoxLayout() #垂直布局

        self.cartesian_inputs={}    #笛卡尔输入
        self.cartesian_labels={}    #笛卡尔空间显示

        ###输入框和当前位置显示
        cartesiancontrol_layout=QGridLayout() #表格布局
        for i, coord in enumerate(['x', 'y', 'z', 'yaw', 'roll']):
            ###标签
            label=QLabel(f'{coord.upper()}:')
            
            ###输入框
            input_field=QLineEdit('0.0')
            self.cartesian_inputs[coord]=input_field

            add_btn=QPushButton('加1')
            add_btn.clicked.connect(lambda _, idx=i: self.add_cartesian_one(idx))

            minus_btn=QPushButton('减1')
            minus_btn.clicked.connect(lambda _, idx=i: self.minus_cartesian_one(idx))
            
            ###当前位置显示
            current_label=QLabel('0.0')
            current_label.setFixedWidth(60) #设置固定长度,防止表格布局变来变去
            self.cartesian_labels[coord]=current_label
            
            cartesiancontrol_layout.addWidget(label, i, 0) #放置每个关节控制组件到网格布局
            cartesiancontrol_layout.addWidget(input_field, i, 1)
            cartesiancontrol_layout.addWidget(add_btn,i,2)
            cartesiancontrol_layout.addWidget(minus_btn,i,3)
            cartesiancontrol_layout.addWidget(QLabel('当前:'), i, 4)
            cartesiancontrol_layout.addWidget(current_label, i, 5)

        cartesian_layout.addLayout(cartesiancontrol_layout) #把笛卡尔控制表格添加到笛卡尔空间组

        ###笛卡尔控制按钮
        cartesian_control_layout=QHBoxLayout() #两个按钮对应的水平布局
    
        ####直线控制按钮
        self.line_control_btn=QPushButton('关节曲线') #默认关节曲线
        self.line_control_btn.clicked.connect(self.toggle_trajecorymode) #绑定回调函数

        ####执行按钮
        self.move_btn=QPushButton('执行运动')
        self.move_btn.clicked.connect(self.move_to_cartesian)

        cartesian_control_layout.addWidget(QLabel('当前的轨迹模式:')) #放置按钮到水平布局
        cartesian_control_layout.addWidget(self.line_control_btn)
        cartesian_control_layout.addWidget(self.move_btn)
        cartesian_layout.addLayout(cartesian_control_layout) #将该水平布局添加到笛卡尔控制的垂直布局

        ###笛卡尔轨迹组 可在笛卡尔直线和关节曲线之间切换
        cartesian_trajectory_group=QGroupBox('需要到达的中间点') #笛卡尔轨迹的功能组
        cartesian_trajectory_layout=QVBoxLayout() #垂直布局

        ####笛卡尔空间的中间点表格
        self.cartesian_trajectory_table=QTableWidget(0, 6) #表格，存放中间点
        self.cartesian_trajectory_table.setMinimumHeight(120) #设置表高度,多显示几行
        self.cartesian_trajectory_table.setHorizontalHeaderLabels(['X','Y','Z','Yaw','Roll','爪子开合']) #水平表头

        ####关节轨迹按钮
        cartesian_trajectory_buttons_layout=QHBoxLayout() #水平布局
        
        #####添加中间点按钮
        add_cartesian_point_btn=QPushButton('添加中间点')
        add_cartesian_point_btn.clicked.connect(self.add_cartesian_trajectory_point) #绑定回调函数
        
        #####清除中间点按钮
        clear_cartesian_points_btn=QPushButton('清除中间点')
        clear_cartesian_points_btn.clicked.connect(self.clear_cartesian_trajectory_points) #绑定回调函数
        
        #####执行轨迹按钮
        self.execute_cartesian_trajectory_btn=QPushButton('执行轨迹')
        self.execute_cartesian_trajectory_btn.clicked.connect(self.execute_cartesian_trajectory) #绑定回调函数

        cartesian_trajectory_buttons_layout.addWidget(add_cartesian_point_btn) #将3个按钮添加到按钮的水平布局中
        cartesian_trajectory_buttons_layout.addWidget(clear_cartesian_points_btn)
        cartesian_trajectory_buttons_layout.addWidget(self.execute_cartesian_trajectory_btn)

        cartesian_trajectory_layout.addWidget(self.cartesian_trajectory_table) #添加到笛卡尔轨迹组布局
        cartesian_trajectory_layout.addLayout(cartesian_trajectory_buttons_layout)
        
        cartesian_trajectory_group.setLayout(cartesian_trajectory_layout) #为笛卡尔轨迹组设置垂直布局
        cartesian_layout.addWidget(cartesian_trajectory_group) #将笛卡尔轨迹组布局添加到笛卡尔控制组的垂直布局中

        cartesian_group.setLayout(cartesian_layout) #为笛卡尔空间控制组设置布局为网格布局

        ##信息传递组
        signal_group=QGroupBox('信息传递') #和视觉检测窗口进行信息传递的功能组
        signal_layout=QGridLayout() #表格布局

        ###信号显示文本框
        self.signal_log=QTextEdit() #多行文本框，展示接受和发送信息
        self.signal_log.setReadOnly(True) #设置只读
        self.signal_log.setMinimumHeight(120) #设置表高度,多显示几行
        self.signal_log.append('已初始化')

        ###指令下拉框
        self.signal_combo_label=QLabel('可用指令:')
        self.signal_combo_label.setFixedWidth(80) #最大宽度
        self.signal_combo=QComboBox()
        self.signal_combo.addItems(['测试信息']) #添加选项
        self.signal_combo.setCurrentText('测试信息') #设置默认指令

        ###窗口连接按钮
        self.signal_send_btn=QPushButton('发送')
        self.signal_send_btn.clicked.connect(self.send_choose_signal) #绑定回调函数

        signal_layout.addWidget(self.signal_log,0,0,1,4) #把信息传递组的几个组件添加到该垂直布局
        signal_layout.addWidget(self.signal_combo_label,1,0)
        signal_layout.addWidget(self.signal_combo,1,1,1,2)
        signal_layout.addWidget(self.signal_send_btn,1,3)

        signal_group.setLayout(signal_layout) #将这个表格布局设置为信息传递组的布局

        ##急停恢复组
        pause_resume_group=QGroupBox('急停')
        pause_resume_layout=QHBoxLayout() #水平布局

        ###急停按钮
        pause_btn=QPushButton('急停')
        pause_btn.clicked.connect(self.pause) 

        ###恢复按钮
        resume_btn=QPushButton('恢复')
        resume_btn.clicked.connect(self.resume)

        pause_resume_layout.addWidget(pause_btn)
        pause_resume_layout.addWidget(resume_btn)

        pause_resume_group.setLayout(pause_resume_layout)   

        #添加所有组件到左侧面板
        left_layout.addWidget(serial_group)
        left_layout.addWidget(joint_group)
        left_layout.addWidget(cartesian_group)
        left_layout.addWidget(signal_group)
        left_layout.addWidget(pause_resume_group)

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
        self.update_all() #更新所有

    def closeEvent(self, event):
        """关闭窗口时的处理函数"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("串口已关闭")
        self.destroyed.emit() #发出销毁信号.没有销毁
        self.destroy()

    ############################## 笛卡尔步进 ##############################
    def add_cartesian_one(self,index,per=1):
        """添加笛卡尔分量1"""
        print(index) #测试用
        x=self.current_position['x,y,z'][0]
        y=self.current_position['x,y,z'][1]
        z=self.current_position['x,y,z'][2]
        yaw=self.current_position['yaw']
        roll=self.current_position['roll']

        if index==0:
            x+=per
        elif index==1:
            y+=per
        elif index==2:
            z+=per
        elif index==3:
            yaw+=per
        elif index==4:
            roll+=per
        joint_5_angle,is_solvable=self.kinematics.inverse_kinematics([x,y,z],yaw,roll)
        
        if not is_solvable:
            error_message=f"轨迹规划失败：笛卡尔位置无法到达"
            QMessageBox.warning(self, '轨迹规划失败', error_message)
            return
        
        last_joint_point_angle=self.current_joint_angle['0-5']+[self.current_joint_engine[5]]

        #选择动的最小的点，按照1.4,1.3,1.2,1.1,1加权
        weight=[1.4,1.3,1.2,1.1,1]
        best_joint_5_angle=None
        min_movement=float('inf')
        for per_solution in joint_5_angle:
            total_movement=sum(abs(per_solution[i]-last_joint_point_angle[i])*weight[i] for i in range(5)) #按权重

            if total_movement<min_movement:
                min_movement=total_movement
                best_joint_5_angle=per_solution              
        
        last_joint_point_angle=best_joint_5_angle #为下个点迭代做准备

        #发送指令
        joint_5_angle_engine=self.kinematics.angle_to_pwm(last_joint_point_angle)
        joint_engine=joint_5_angle_engine+[self.current_joint_engine[5]]

        self.send_all_commands(joint_engine) #发送指令
        self.update_all()
            
    def minus_cartesian_one(self,index,per=1):
        """较少笛卡尔分量1"""
        print(index) #测试用
        x=self.current_position['x,y,z'][0]
        y=self.current_position['x,y,z'][1]
        z=self.current_position['x,y,z'][2]
        yaw=self.current_position['yaw']
        roll=self.current_position['roll']

        if index==0:
            x-=per
        elif index==1:
            y-=per
        elif index==2:
            z-=per
        elif index==3:
            yaw-=per
        elif index==4:
            roll-=per
        joint_5_angle,is_solvable=self.kinematics.inverse_kinematics([x,y,z],yaw,roll)
        
        if not is_solvable:
            error_message=f"轨迹规划失败：笛卡尔位置无法到达"
            QMessageBox.warning(self, '轨迹规划失败', error_message)
            return
        
        last_joint_point_angle=self.current_joint_angle['0-5']+[self.current_joint_engine[5]]

        #选择动的最小的点，按照1.4,1.3,1.2,1.1,1加权
        weight=[1.4,1.3,1.2,1.1,1]
        best_joint_5_angle=None
        min_movement=float('inf')
        for per_solution in joint_5_angle:
            total_movement=sum(abs(per_solution[i]-last_joint_point_angle[i])*weight[i] for i in range(5)) #按权重

            if total_movement<min_movement:
                min_movement=total_movement
                best_joint_5_angle=per_solution              
        
        last_joint_point_angle=best_joint_5_angle #为下个点迭代做准备

        #发送指令
        joint_5_angle_engine=self.kinematics.angle_to_pwm(last_joint_point_angle)
        joint_engine=joint_5_angle_engine+[self.current_joint_engine[5]]

        self.send_all_commands(joint_engine) #发送指令
        self.update_all()


    ############################## 笛卡尔步进 ##############################
 
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
            self.serial_port=None
            self.is_connected=False

            self.is_trajectory=False #运动位恢复默认
            self.is_pause=False #停止位也恢复默认
            self.last_command='' #然后存储的上次指令也清零
            
            self.current_joint_engine=[1500,1500,1500,1500,1500,1500] #关节位置 舵机

            self.trajectory=[] #存储待执行关节轨迹
            self.current_trajectory_index=0 #存储当前执行的轨迹点索引

            self.calibration_count=0 #初始化标定点计数器
            
            self.connect_btn.setText('连接') #更新连接按钮文本
            self.statusBar().showMessage('未连接') #更新状态栏
    ############################### 串口设置 ###############################

    ############################### 关节控制 ###############################
    def update_all(self):
        """整合的更新显示"""
        self.update_joint_display() #更新角度显示
        self.update_cartesian_display() #更新笛卡尔空间显示
        self.update_3d_view() #更新3d视图显示

    def update_joint_display(self):
        """更新关节位置显示"""
        #先计算前5个关节转换的角度
        self.current_joint_angle['0-5']=self.kinematics.pwm_to_angle(self.current_joint_engine[:5])

        #计算夹爪状态
        if self.current_joint_engine[5]>2250:
            self.current_joint_angle['6']='闭合'
        else:
            self.current_joint_angle['6']='张开'
        
        #更新显示
        for i,value_rad in enumerate(self.current_joint_angle['0-5']): #更新前5个，还得用角度值显示
            value_deg=value_rad*(180/np.pi) #转换成角度制
            self.joint_values_angle[i].setText(f"{value_deg:.2f}") #保留2位小数
        self.joint_values_angle[5].setText(self.current_joint_angle['6']) #更新夹爪状态

    def on_slider_changed(self, joint_index, value):
        """滑块值改变时的处理函数"""
        #限制滑块1的值在舵机允许位置之间
        if joint_index==1:
            if value<882:
                self.joint_sliders[joint_index].setValue(882)
            elif value>2266:
                self.joint_sliders[joint_index].setValue(2266)

        self.joint_values_engine[joint_index].setText(str(value)) #同步改变标签显示值
        self.joint_inputs[joint_index].setText(str(value)) #同步改变输入框输入值
        if self.is_realtimecontrol: #如果是实时控制状态,会实时发送指令并更新
            self.send_single_command(joint_index, value) #控制机械臂对应关节，会更新当前舵机位置
            self.update_all() #更新所有
        
    def on_input_changed(self, joint_index):
        """输入框值改变时的处理函数"""
        try:
            value=int(self.joint_inputs[joint_index].text()) #获取输入框的值
            value=max(500, min(2500, value)) #用上下界截断，防止超出范围

            if self.is_realtimecontrol:
                self.is_realtimecontrol=False #改成触发控制，不然会一下子过去
                self.joint_sliders[joint_index].setValue(value) #同步改变滑块的值,会自动调用滑块改变函数
                self.is_realtimecontrol=True
                self.send_command() #控制机械臂、更新3d视图、更新笛卡尔坐标

        except ValueError:
            QMessageBox.warning(self, '输入错误', '请输入有效的数值(500-2500)')

    def toggle_controlmode(self):
        """切换控制模式:实时/触发"""
        if self.is_realtimecontrol: #实时控制中
            self.is_realtimecontrol=False
            self.realtime_control_btn.setText('触发控制') #更新按钮文本
            self.send_all_btn.setEnabled(True) #触发控制下发送命令按钮才有用
            self.execute_joint_trajectory_btn.setEnabled(True) #触发控制下才能进行关节轨迹规划
        else: #触发控制中
            self.is_realtimecontrol=True
            self.realtime_control_btn.setText('实时控制') #更新连接按钮文本
            self.send_all_btn.setEnabled(False) #实时控制下发送命令按钮没有用途，无法按下
            self.execute_joint_trajectory_btn.setEnabled(False) #实时控制下也不能进行关节轨迹规划
            #本来想直接同步当前状态，但为了安全性着想，进行重置
            self.reset_joints()

    def send_single_command(self, joint_index, value,time=500): #只管发送和更新关节位置
        """发送单个关节控制命令"""
        is_done=True #指示是否执行成功
        if self.is_connected and self.serial_port and self.serial_port.is_open:
            try:
                command=f"#{joint_index:03d}P{value:04d}T{time:04d}!"  #修改时间为500

                if not self.is_pause: #如果是暂停，不发送指令但是还想更新指令，这样实时控制可以暂停，然后会跟着滑块动
                    self.serial_port.write(command.encode()) #发送指令
                    print(f"发送命令: {command}")
                
                self.last_command=command.encode() #存储上次发送指令
            except Exception as e:
                print(f"发送命令失败: {str(e)}")
                is_done=False
        if is_done:
            self.current_joint_engine[joint_index]=value #更新关节位置

            ##测试用
            # result,is_solvable=self.kinematics.inverse_kinematics(self.current_position['x,y,z'],self.current_position['yaw'],self.current_position['roll'])
            # if len(result)==4:
            #     print(result)
 
    def send_all_commands(self,joint_value,time=500): #只管发送和更新关节位置，joint_value是一个列表，6个舵机值
        """发送所有关节控制命令"""
        #if not self.is_connected:
        #    QMessageBox.warning(self, '错误', '请先连接串口！')
        #    return
        #print(time) #测试
        is_done=True #指示是否执行成功
        if self.is_connected and self.serial_port and self.serial_port.is_open:
            try:
                command="{"  #开始组合命令
                for i in range(6):
                    value=joint_value[i] #取目标关节角度
                    command+=f"#{i:03d}P{value:04d}T{time:04d}!"
                command+="}"  #结束组合命令
                self.serial_port.write(command.encode())
                self.last_command=command #存储上次发送指令
                #print(f"发送命令: {command}")
            except Exception as e:
                QMessageBox.warning(self, '发送错误', f'发送命令失败：{str(e)}')
                is_done=False
        if is_done: #不然离线仿真没法更新
            self.current_joint_engine=joint_value #更新关节位置

            ##测试用
            # result,is_solvable=self.kinematics.inverse_kinematics(self.current_position['x,y,z'],self.current_position['yaw'],self.current_position['roll'])
            # if len(result)==4:
            #     print(result)

    def send_command(self): #只进行没有中间点的插值，因为是插值，会开启线程，会自动更新
        """发送指令按钮"""
        points=[]
        points.append(self.current_joint_engine) #当前点是起始点
        point=[]
        for i in range(6):
            value=self.joint_inputs[i].text() #获取关节输入框里的内容
            point.append(int(value))
        points.append(point)
        
        #调用do_joint_trajectory
        #print(points) #测试用
        self.do_joint_trajectory(points) #执行轨迹规划

    def reset_joints(self):
        """复位所有关节到竖直状态"""
        #设置竖直状态的PWM值
        reset_values=[1500]*6
        
        is_realtimecontrol=self.is_realtimecontrol #记录是否是实时控制
        self.is_realtimecontrol=False #改成触发控制

        for i, value in enumerate(reset_values):
            self.joint_sliders[i].setValue(value) #重置滑块,会自动调用滑块改变函数
        self.is_realtimecontrol=is_realtimecontrol #这样两种控制都调用插值复位

        self.send_command() #控制机械臂、更新3d视图、更新笛卡尔坐标
    ############################### 关节控制 ###############################

    ############################# 关节轨迹规划 #############################
    def generate_joint_trajectory(self,point_list,interpolation_time=1000,sample_time=50): #负责根据输入点生成连续插值函数，然后再采样，返回采样点列表 interpolation_time是插值的相邻两点的时间sampletime是采样时间
        """生成关节轨迹的离散采样点"""
        #点列表示例
        #point_list=[[1,2,3,4,5,6],[1,2,3,4,5,6]] #还可以更多点，分别对应6个舵机的位置
        point_list=np.array(point_list)  # 转换成np数组

        #默认按照50ms一插值，然后每个中间点用1000ms到达
        trajectory=[]  # 存储采样点

        #拟合与采样
        for joint in range(point_list.shape[1]):  # 对每个关节进行插值处理
            joint_values=point_list[:,joint]  # 获取每个关节的角度值
            time_values=np.arange(0,len(point_list)*interpolation_time,interpolation_time)  # 每个点之间时间间隔为1000ms

            #创建线性插值函数
            interp_func=interp1d(time_values,joint_values,kind='linear')

            #生成采样时间点，每50ms采样一次
            sampled_times=np.arange(0,time_values[-1]+sample_time,sample_time)
            sampled_joint_values=interp_func(sampled_times)

            #将插值后的关节值添加到轨迹列表中
            if len(trajectory)==0: #对于第一个关节，直接初始化轨迹列表
                
                trajectory=[[val] for val in sampled_joint_values]
            else:
                for i,joint_value in enumerate(sampled_joint_values):
                    trajectory[i].append(joint_value)

            trajectory=[[int(joint_value) for joint_value in joint] for joint in trajectory] #转换为整数

            #可视化插值函数和采样点 测试用
            # plt.figure(figsize=(8,6))
            # plt.plot(time_values,joint_values,'o',label='Original Points',markersize=10)  #原始点
            # plt.plot(sampled_times,sampled_joint_values,'-',label='Interpolated Curve')  #插值曲线
            # plt.scatter(sampled_times,sampled_joint_values,color='red',label='Sampled Points',zorder=5)  #采样点
            # plt.xlabel('Time (ms)')
            # plt.ylabel(f'Joint {joint + 1} Angle')
            # plt.title(f'Interpolation for Joint {joint + 1}')
            # plt.legend()
            # plt.grid(True)
            # plt.show()

        return trajectory #返回的是列表的列表
    
    def do_joint_trajectory(self,trajectory,interpolation_time=1000,sample_time=50): #解耦，给的是要经过点
        """执行一个关节轨迹,输入是要经过点"""
        #对输入点进行处理，去除重复相邻点
        filtered_trajectory=[trajectory[0]] #第一个点
        for point in trajectory:
            if point!=filtered_trajectory[-1]:
                filtered_trajectory.append(point) #如果和过滤后的最后一个不一样，就添加进去

        if len(filtered_trajectory)==1: #如果只有一个点也就是不用动的情况
            return

        filtered_trajectory=self.generate_joint_trajectory(filtered_trajectory,interpolation_time=interpolation_time,sample_time=sample_time) #生成采样点
        
        self.do_trajectory(filtered_trajectory,sample_time) #调用执行轨迹

    def do_trajectory(self,trajectory,sample_time=50,subesube_para=1):
        """执行一个关节轨迹,输入的是待执行关节空间离散采样点"""
        #调整trajectory，让每个点都是原来的值加上相邻关节变量差*1，最后一个点不变,这样轨迹应该能滑溜溜
        trajectory=np.array(trajectory,dtype=float)
        subesube_trajectory=trajectory.copy()
        for i in range(1,len(trajectory)-1):
            subesube_trajectory[i]+=subesube_para*(trajectory[i]-trajectory[i-1])

        subesube_trajectory = np.clip(subesube_trajectory, 500, 2500) #限制在500到2500，防止错误

        # print('原来的',trajectory) #测试用
        # print('滑溜溜的',subesube_trajectory) #测试用

        self.trajectory=subesube_trajectory.astype(int).tolist() #设置为类成员,方便传递给do_per_trajectory执行

        #安全起见,冻结发送
        self.send_all_btn.setEnabled(False) #关节空间发送
        self.execute_joint_trajectory_btn.setEnabled(False) #关节空间执行轨迹
        self.move_btn.setEnabled(False) #笛卡尔的发送
        self.execute_cartesian_trajectory_btn.setEnabled(False) #笛卡尔空间的执行轨迹

        #创建计时器
        self.current_trajectory_index=0 #指示发送了几条指令
        self.joint_trajectory_timer=QTimer()
        self.joint_trajectory_timer.timeout.connect(lambda:self.do_per_trajectory(sample_time,subesube_para)) #绑定回调函数
        self.is_trajectory=True
        self.joint_trajectory_timer.start(sample_time) #开始执行

    def do_per_trajectory(self,sample_time=50,subesube_para=0.2):
        '''关节轨迹单次发送指令'''
        if self.is_pause: #暂停的情况
            self.joint_trajectory_timer.stop() #停止
        else:
            if self.current_trajectory_index<len(self.trajectory):
                joint_value=self.trajectory[self.current_trajectory_index]
                #print(joint_value) #测试用
                self.send_all_commands(joint_value,time=int((1+subesube_para)*sample_time)) #发送指令，更改当前位置,按照1.2倍时间发送，这样轨迹会更丝滑
                self.update_all() #更新所有
                self.current_trajectory_index+=1
            else: #执行完的情况
                self.joint_trajectory_timer.stop()
                self.joint_trajectory_timer.deleteLater()  #清理计时器资源
                self.joint_trajectory_timer=None
                self.is_trajectory=False
                #允许使用按钮
                if not self.is_realtimecontrol: #如果是触发控制，才解冻按钮
                    self.send_all_btn.setEnabled(True)
                    self.execute_joint_trajectory_btn.setEnabled(True)
                self.move_btn.setEnabled(True) #笛卡尔的发送
                self.execute_cartesian_trajectory_btn.setEnabled(True) #笛卡尔空间的执行轨迹

    def add_joint_trajectory_point(self):
        """添加关节空间轨迹点"""
        row=self.joint_trajectory_table.rowCount() #获取现在的行数
        self.joint_trajectory_table.insertRow(row) #在最后插入新的一行
        for i in range(6):
            value=self.joint_inputs[i].text() #获取关节输入框里的内容
            self.joint_trajectory_table.setItem(row, i, QTableWidgetItem(value)) #设置新的一行内容

    def clear_joint_trajectory_points(self):
        """清除所有关节空间轨迹点"""
        self.joint_trajectory_table.setRowCount(0)

    def execute_joint_trajectory(self):
        """执行关节空间轨迹规划"""
        # if not self.is_connected:
        #     QMessageBox.warning(self, '错误', '请先连接串口！')
        #     return
        #为了离线仿真
            
        points=[]
        points.append(self.current_joint_engine) #当前点是起始点

        for row in range(self.joint_trajectory_table.rowCount()): #添加中间点
            point=[]
            for col in range(6):
                item=self.joint_trajectory_table.item(row, col)
                point.append(int(item.text() if item else 0))
            points.append(point)

        point=[]
        for i in range(6):
            value=self.joint_inputs[i].text() #获取笛卡尔输入框里的内容，添加终点
            point.append(int(value))
        points.append(point)
        
        #调用do_joint_trajectory
        #print(points)
        self.do_joint_trajectory(points) #执行轨迹规划
    ############################# 关节轨迹规划 #############################

    ############################## 笛卡尔控制 ##############################
    def update_cartesian_display(self): #根据当前角度改，纯更新笛卡尔坐标
        """更新笛卡尔坐标显示"""
        try:
            #获取当前关节角度
            joint_values=self.current_joint_engine
            joint_values=joint_values[:5] #逆运动学不涉及最后一个角度
            angles=self.kinematics.pwm_to_angle(joint_values)
            
            #使用正运动学计算各个关节位置
            positions,yaw,roll=self.kinematics.forward_kinematics(angles) #position包括了7个值呢，要的是最后一行
            position=positions[-1]
            
            #更新显示
            self.cartesian_labels['x'].setText(f"{position[0]:.1f}")
            self.cartesian_labels['y'].setText(f"{position[1]:.1f}")
            self.cartesian_labels['z'].setText(f"{position[2]:.1f}")
            self.cartesian_labels['yaw'].setText(f"{np.degrees(yaw):.1f}")
            self.cartesian_labels['roll'].setText(f"{np.degrees(roll):.1f}")
            
            #更新当前位置字典（如果其他地方需要用到）
            self.current_position = {
                'x,y,z': [position[0],position[1],position[2]],
                'yaw': yaw,
                'roll': roll
            }
            
        except Exception as e:
            print(f"更新笛卡尔坐标显示错误: {str(e)}")

    def toggle_trajecorymode(self):
        """切换轨迹控制模式:关节曲线/笛卡尔直线"""
        if self.is_linetrajectory: #笛卡尔直线中
            self.is_linetrajectory=False
            self.line_control_btn.setText('关节曲线') #更新按钮文本
        else: #关节曲线中
            self.is_linetrajectory=True
            self.line_control_btn.setText('笛卡尔直线') #更新按钮文本

    def move_to_cartesian(self): #按钮调用，作用是只获取当前输入框的内容并到达，然后夹爪状态不动
        """移动到笛卡尔坐标位置"""

        x=float(self.cartesian_inputs['x'].text()) #获取输入框的值
        y=float(self.cartesian_inputs['y'].text())
        z=float(self.cartesian_inputs['z'].text())
        yaw=np.radians(float(self.cartesian_inputs['yaw'].text()))
        roll=np.radians(float(self.cartesian_inputs['roll'].text()))
        
        #打印调试信息
        #print(f"目标位置: x={x:.1f}, y={y:.1f}, z={z:.1f}, yaw={np.degrees(yaw):.1f}°, roll={np.degrees(roll):.1f}°")

        points=[] #xyz和yaw的位置，然后还有最后关节的舵机位置
        points.append(self.current_position['x,y,z']+[self.current_position['yaw']]+[self.current_position['roll']]+[self.current_joint_engine[5]])
        points.append([x,y,z,yaw,roll]+[self.current_joint_engine[5]])

        self.do_cartesian_trajectory(points) #执行轨迹
    ############################## 笛卡尔控制 ##############################

    ############################ 笛卡尔轨迹规划 ############################
    def do_cartesian_trajectory(self,trajectory,if_line=0,interpolation_time=1000,sample_time=50): #解耦，给的是要经过点 if_line是0按类成员，是1按照笛卡尔直线，是2按照关节曲线，trajectory需包括笛卡尔位置和夹爪的舵机位置
        """执行一个关节轨迹,输入是要经过点"""
        #对输入点进行处理，去除重复相邻点
        filtered_trajectory=[trajectory[0]] #第一个点
        for point in trajectory:
            if point!=filtered_trajectory[-1]:
                filtered_trajectory.append(point) #如果和过滤后的最后一个不一样，就添加进去

        if len(filtered_trajectory)==1: #如果只有一个点也就是不用动的情况
            return
        
        if (if_line==0 and self.is_linetrajectory) or if_line==1:
            #使用笛卡尔直线控制
            sample_points=self.generate_cartesian_trajectory_to_cartersian(filtered_trajectory,interpolation_time=interpolation_time,sample_time=sample_time) #转换成笛卡尔采样点

            #转换到关节空间
            points,is_solvable,error_message=self.trans_cartesian_to_joint(sample_points) #已采样舵机6关节，可直接执行
            if not is_solvable:
                QMessageBox.warning(self, '轨迹规划失败', error_message)
                return False
            
            #可视化笛卡尔空间和关节空间插值函数和采样点 测试用
            # sample_points_list=np.array(sample_points)
            # sampled_times = [i * sample_time for i in range(sample_points_list.shape[0])]

            # label_cartesian=['x','y','z','yaw','roll','夹爪闭合']

            # plt.figure(figsize=(10, 6))
            # for i in range(sample_points_list.shape[1]):  # 遍历每个状态变量（列）
            #     plt.plot(sampled_times, sample_points_list[:, i], label=label_cartesian[i], marker='o')

            # plt.xlabel('Time (ms)')
            # plt.ylabel('State Value')
            # plt.title('cartesian')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
            
            # points_list=np.array(points)
            
            # plt.figure(figsize=(10, 6))
            # for i in range(points_list.shape[1]):  # 遍历每个状态变量（列）
            #     plt.plot(sampled_times, points_list[:, i], label=f'joint {i + 1}', marker='o')

            # plt.xlabel('Time (ms)')
            # plt.ylabel('State Value')
            # plt.title('joint')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
            
            self.do_trajectory(points) #执行轨迹即可

        elif (if_line==0 and not self.is_linetrajectory) or if_line==2:
            #使用关节曲线控制,就直接求输入点对应的关节位置，然后进行关节轨迹控制
            #转换笛卡尔空间到关节空间
            points,is_solvable,error_message=self.trans_cartesian_to_joint(filtered_trajectory) #转换成未进行采样的舵机6关节角度,进行判断
            if not is_solvable:
                QMessageBox.warning(self, '轨迹规划失败', error_message)
                return False

            self.do_joint_trajectory(points,interpolation_time=interpolation_time,sample_time=sample_time) #在这里进行采样

    def trans_cartesian_to_joint(self,cartesian_list): #列表包含5个位置值和夹爪闭合舵机位置，返回舵机位置
        """将笛卡尔空间列表转换为关节空间"""
        weight=[1.4,1.3,1.2,1.1,1] #各关节权重
        engine_points_list=[] #初始化舵机位置列表

        for i,value in enumerate(cartesian_list):
            if i==0: #第一个点，起点其实就是当前位置，直接用存好的
                last_joint_point_angle=self.current_joint_angle['0-5'] #用这个来为多解取值
                engine_points_list.append(self.current_joint_engine) #添加到列表，是6个舵机位置
                continue #第一个点也不用计算有没有解了

            joint_5_angle,is_solvable=self.kinematics.inverse_kinematics(value[:3],value[3],value[4]) #用x,y,z和位姿来计算 逆运动学里一定考虑到舵机可表示范围，传回来的一定是能用舵机到达的
            
            if not is_solvable:
                error_message=f"轨迹规划失败：笛卡尔位置 {value[:5]} 无法到达"
                return [],False,error_message
            
            #选择动的最小的点，按照1.4,1.3,1.2,1.1,1加权
            best_joint_5_angle=None
            min_movement=float('inf')
            for per_solution in joint_5_angle:
                total_movement=sum(abs(per_solution[i]-last_joint_point_angle[i])*weight[i] for i in range(5)) #按权重

                if total_movement<min_movement:
                    min_movement=total_movement
                    best_joint_5_angle=per_solution              
            
            last_joint_point_angle=best_joint_5_angle #为下个点迭代做准备

            joint_5_angle_engine=self.kinematics.angle_to_pwm(best_joint_5_angle) #转换成舵机位置
            engine_points_list.append(joint_5_angle_engine+[int(value[5])]) #加上夹爪的状态
        
        return engine_points_list,True,None

    def linear_with_parabolic(self,given_t, point_start, point_end, total_time, a, a_t, v):
        """带抛物线过渡的线性插值"""
        if 0 <= given_t <= a_t:  # 加速段
            y = 0.5 * a * given_t**2 + point_start
        elif a_t < given_t <= total_time - a_t:  # 匀速段
            y = self.linear_with_parabolic(a_t,point_start, point_end, total_time, a, a_t, v) + v * (given_t - a_t)
        elif total_time - a_t < given_t <= total_time:  # 减速段
            y = self.linear_with_parabolic(total_time-a_t,point_start, point_end, total_time, a, a_t, v)+v*(given_t-total_time+a_t)-0.5*a*(given_t-total_time+a_t)**2
        return y

    def generate_cartesian_trajectory_to_cartersian(self,point_list,time_list=None,interpolation_time=1000,sample_time=50): #输入的是6个笛卡尔位置，包括夹爪开合，时间列表应该是6个笛卡尔位置减1而且是50的正数
        """生成笛卡尔轨迹的离散采样点,直线"""
        #负责根据输入点生成连续插值函数，然后再采样，返回采样点列表，全都是笛卡尔位置,直线运动用，不进行可行解的判断
        if time_list is None:
            time_list=[interpolation_time]*(len(point_list)-1) #时间列表
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
                    time_vals=np.arange(sample_time,total_time,sample_time) #这里因为最后会单独添加，所以采样时间可以不包括total_time
                    
                    trajectory_vals=[per_trajectory[-1]]*len(time_vals)+[point_end]
                    
                    per_trajectory+=trajectory_vals #添加进当前分量

                else: #其他笛卡尔位姿
                    a=12*(point_end-point_start)/(total_time**2) #加速度
                    if a==0: #两点一样，就没有加速度
                        a_t=0
                    else:
                        a_t=total_time/2-((a**2*total_time**2-4*a*(point_end-point_start))**0.5)/(2*abs(a)) #加速度时间 #必须有abs
                    v=a*a_t

                    #时间取样
                    time_vals=np.arange(sample_time,total_time+sample_time,sample_time)

                    #计算相应采样点
                    trajectory_vals = [self.linear_with_parabolic(t, point_start, point_end, total_time, a, a_t, v) for t in time_vals]

                    per_trajectory+=trajectory_vals #添加进当前分量
            
            trajectory.append(per_trajectory)
        trajectory=[list(item) for item in zip(*trajectory)] #转置，变成采样数长度，元素是5笛卡尔位姿+夹爪状态
        return trajectory

    def add_cartesian_trajectory_point(self):
        """添加轨迹点"""
        row=self.cartesian_trajectory_table.rowCount() #获取现在的行数
        self.cartesian_trajectory_table.insertRow(row) #在最后插入新的一行
        for i, coord in enumerate(['x', 'y', 'z', 'yaw', 'roll']):
            value=self.cartesian_inputs[coord].text() #获取笛卡尔输入框里的内容
            self.cartesian_trajectory_table.setItem(row, i, QTableWidgetItem(value)) #设置新的一行内容
        joint6_value=self.joint_inputs[5].text() #获取当前夹爪的值
        self.cartesian_trajectory_table.setItem(row, 5, QTableWidgetItem(joint6_value)) #夹爪角度

    def clear_cartesian_trajectory_points(self):
        """清除所有轨迹点"""
        self.cartesian_trajectory_table.setRowCount(0)

    def execute_cartesian_trajectory(self):
        """执行轨迹规划"""
        # if not self.is_connected:
        #     QMessageBox.warning(self, '错误', '请先连接串口！')
        #     return
            
        points=[]
        points.append(self.current_position['x,y,z']+[self.current_position['yaw']]+[self.current_position['roll']]+[self.current_joint_engine[5]]) #添加当前点为起始点

        for row in range(self.cartesian_trajectory_table.rowCount()): #添加中间点
            point=[]
            for col in range(6):
                item=self.cartesian_trajectory_table.item(row, col)
                if col==3 or col==4: #俩角度的，要变成弧度制
                    point.append(np.radians(float(item.text() if item else 0)))
                else:
                    point.append(float(item.text() if item else 0))
            points.append(point)

        x=float(self.cartesian_inputs['x'].text()) #获取输入框的值
        y=float(self.cartesian_inputs['y'].text())
        z=float(self.cartesian_inputs['z'].text())
        yaw=np.radians(float(self.cartesian_inputs['yaw'].text()))
        roll=np.radians(float(self.cartesian_inputs['roll'].text()))
        points.append([x,y,z,yaw,roll]+[self.current_joint_engine[5]]) #添加输入框为终点

        #print(points) #测试用

        self.do_cartesian_trajectory(points) #执行轨迹
    ############################ 笛卡尔轨迹规划 ############################
    
    ################################ 3D仿真 ################################
    def calculate_joint_positions(self, joint_values): #更新3d视图用到,给出的是从0org到5org，和夹爪的左右两点，传入的是6舵机位置
        """计算机械臂各关节位置"""
        joint_6=joint_values[-1]
        joint_values=joint_values[:5] #前5舵机位置，用于计算运动学
        angles=self.kinematics.pwm_to_angle(joint_values)
        positions,yaw,roll=self.kinematics.forward_kinematics(angles)
        
        jaw_root=np.array(positions[-2]) #夹爪根部
        jaw_end=np.array(positions[-1]) #夹爪末端

        #计算夹爪末端两个点
        #夹爪中心向量
        center_vector=jaw_end-jaw_root #夹爪根部和末端的连线
        center_vector/=np.linalg.norm(center_vector)

        #夹爪roll为0时和z轴垂直
        z_axis=np.array([0,0,1])
        end_0roll_vector = np.cross(center_vector, z_axis) #末端两点在roll为0的直线
        end_0roll_vector /= np.linalg.norm(end_0roll_vector)

        end_1_end_vector=end_0roll_vector #一侧点和中点的连线
        end_2_end_vector=-end_0roll_vector #另一侧点和中点的连线

        #计算旋转矩阵，旋转轴是Center_cector
        K = np.array([
            [0, -center_vector[2], center_vector[1]],
            [center_vector[2], 0, -center_vector[0]],
            [-center_vector[1], center_vector[0], 0]
        ])
        rotation_matrix = (
            np.eye(3) + 
            np.sin(roll) * K + 
            (1 - np.cos(roll)) * np.dot(K, K)
        )

        #转了roll角度之后的
        end_1_end_vector = np.dot(rotation_matrix, end_1_end_vector)
        end_2_end_vector = np.dot(rotation_matrix, end_2_end_vector)

        #爪子开合宽度 2000之前全都是50mm.之后2500对应10mm
        gripper_width=50
        if joint_6>2000:
            gripper_width=50-0.08*(joint_6-2000)

        # 计算爪子开合宽度
        end_1=(jaw_end+gripper_width/2*end_1_end_vector).tolist()
        end_2=(jaw_end+gripper_width/2*end_2_end_vector).tolist()
        
        return positions[:6] + [end_1, end_2]

    def init_3d_view(self):
        """初始化3D视图"""
        self.ax.clear()
        self.ax.set_xlim([-500, 500])
        self.ax.set_ylim([-500, 500])
        self.ax.set_zlim([0, 500])
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.view_init(elev=30, azim=-135)
        self.canvas.draw()

    def update_3d_view(self):
        """更新3D视图"""
        self.ax.clear()
        
        #获取当前关节角度
        joint_values=self.current_joint_engine
        
        #计算机械臂各关节位置
        points=self.calculate_joint_positions(joint_values)
        #print(points)
        
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
        p5_left=np.array(points[-2])
        p5_right=np.array(points[-1])
        
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
        self.ax.set_xlim([-500, 500])
        self.ax.set_ylim([-500, 500])
        self.ax.set_zlim([0, 500])
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        
        #设置视角
        #self.ax.view_init(elev=30, azim=-135)
        
        self.canvas.draw()
    ################################ 3D仿真 ################################
    
    ############################### 人脸随动 ###############################
    def do_move_face(self,engine_03):
        """执行一次人脸随动"""
        #只动04，然后时间长一点
        engine=self.current_joint_engine
        
        engine[0]=int(max(500, min(2500, engine[0]-engine_03[0])))
        engine[3]=int(max(500, min(2500, engine[3]-engine_03[1])))

        self.send_all_commands(engine,time=500) #发送所有关节指令
        
    ############################### 人脸随动 ###############################

    ############################### 通信相关 ###############################
    def send_signal(self,signal):
        """发送信号给主窗口"""
        self.controller_signal.emit(signal) #发送信息

        self.show_signal(signal=signal,mode=1) #展示信息

    def receive_signal(self,signal):
        """接受主窗口转发的视觉检测信号"""
        self.show_signal(signal=signal,mode=2) #展示信息

        if signal=="start_calibration": #如果是开始标记
            self.signal_combo.addItems(['发送标定点']) #添加选项
            self.calibration_count=0 #初始化标定点计数器

        elif signal.startswith('move_face engine:'): #人脸随动发送指令
            data_str=signal[len('move_face engine:'):] #去头
            try:
                engine_04=eval(data_str)
                if isinstance(engine_04, list):
                    self.do_move_face(engine_04) #TODO 时间长一点的
                else:
                    print("Error: Data is not a list.")
            except Exception as e:
                print(f"Failed to parse calibration data: {e}")
        
        elif signal=='start_move_with_face': #如果是开始人脸随动
            #设置
            init_value=[1500,1820,2150,1500,1500,1500]
            for i in range(len(init_value)):
                self.joint_inputs[i].setText(str(init_value[i])) #设置关节输入框的值
            self.send_command() #初始化

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
        signal_choose=self.signal_combo.currentText() #获取当前所选指令
        
        if signal_choose=='发送标定点':
            calibration_point=self.current_position['x,y,z'][:2] #取x,y
            # if self.calibration_count==0:
            #     calibration_point=[221.176,130.626] #测试用
            # elif self.calibration_count==1:
            #     calibration_point=[193.288,-106.53800] #测试用
            # elif self.calibration_count==2:
            #     calibration_point=[78.409,-84.7644] #测试用
            # elif self.calibration_count==3:
            #     calibration_point=[80.5877,122.2352] #测试用   
            signal='calibration point:'+str(calibration_point)
            self.calibration_count+=1 #计数器加1
            self.send_signal(signal)

            if self.calibration_count==4:
                signal='calibration done'
                self.send_signal(signal) #发送完成标记
        else:
            self.send_signal(signal_choose)
    ############################### 通信相关 ###############################

    ############################### 急停恢复 ###############################
    def pause(self):
        """暂停,停下正在执行轨迹"""
        if (self.is_trajectory or self.is_realtimecontrol) and not self.is_pause: #如果连接了串口并且正在运动(轨迹或者实时控制)的话才暂停 
            try:
                self.is_pause=True #停止标记变成真 计时器里有判断，如果真就不发送下一条指令
                command=f"$DST!"  #停止指令 然后这里直接停下当前的运动，这样暂停的比较优雅的说
                if self.is_connected and self.serial_port and self.serial_port.is_open:
                    self.serial_port.write(command.encode()) #发送指令
                    print(f"发送命令: {command}")
            except Exception as e:
                print(f"发送命令失败: {str(e)}")

    def resume(self):
        """恢复,恢复正在执行的轨迹"""
        if (self.is_trajectory or self.is_realtimecontrol) and self.is_pause: #如果连接了串口并且正在运动而且是暂停状态
            try:
                self.is_pause=False #先恢复停止标记,不然计时器恢复工作也会停止
                if self.is_connected and self.serial_port and self.serial_port.is_open:
                    self.serial_port.write(self.last_command.encode()) #上一次的指令可能因为暂停没执行完，先执行
                    print(f"发送命令: {self.last_command}")
                if self.is_trajectory: #如果是轨迹运行中
                    self.joint_trajectory_timer.start() #恢复计时器工作,默认使用上一次时间
            except Exception as e:
                print(f"发送命令失败: {str(e)}")
    ############################### 急停恢复 ###############################


#主函数
def main():
    app=QApplication(sys.argv) #负责应用事件循环
    window=RoboticArmGUI() #自定义的窗口
    window.show() #展示窗口
    sys.exit(app.exec_()) #启动事件循环，满足退出条件的时候就结束事件循环

if __name__ == '__main__':
    main()