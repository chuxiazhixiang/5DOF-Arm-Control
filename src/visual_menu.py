import sys
import serial
import serial.tools.list_ports
import numpy as np
from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,
                           QHBoxLayout,QLabel,QSlider,QPushButton,QComboBox,
                           QTabWidget,QGroupBox,QLineEdit,QTableWidget,
                           QTableWidgetItem,QMessageBox,QGridLayout,
                           QTextEdit)
from PyQt5.QtCore import Qt,QTimer,pyqtSignal,QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QTextCharFormat,QColor,QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import cv2
import time

from src.visual_tools import FocusFinder,ColorCylinderDetecter
from src.move_with_face import MovewithFace

#检测线程
class DetectThread(QThread):

    update_ready_signal=pyqtSignal(dict) #可更新信号，字典类型

    def __init__(self): #连接成功才能检测
        super().__init__()
        self.capture=None #连接的摄像头
        self.focusfinder=FocusFinder() #注意力检测(标定)
        self.colorcylinderdetecter=ColorCylinderDetecter() #彩色圆柱检测
        self.running=True #标志运行情况

    def trans_camera(self,capture):
        """传递摄像头"""
        self.capture=capture

    def update_calibration(self,calibration_list):
        self.colorcylinderdetecter.update(calibration_list)

    def run(self):
        while self.running:
            try:
                ret,frame=self.capture.read()  #从摄像头读取一帧
                if not ret:
                    self.update_ready_signal.emit({'status':'error stop'}) #发送错误停止
                    break

                #检测
                focus_img,if_find=self.focusfinder.find_focus(frame) #注意力图片，找工作区
                if if_find: #找到的情况
                    annoted_img,norm_color_result,if_detect=self.colorcylinderdetecter.detect(focus_img) #标注图片
                    if if_detect: #检测可更新
                        self.update_ready_signal.emit({'status':'upgrade','img':annoted_img,'result':norm_color_result}) #发送标注图片和检测结果

            except Exception as e:
                self.update_ready_signal.emit({'status':'error stop'}) #发送错误停止
                print(e)
                break

        if not self.running: #因为安全停止running是false，所以会发送正常停止信号；而错误停止的running是true，所以不会发正常停止信号
            self.update_ready_signal.emit({'status':'natural stop'}) #发送正常停止信号，正常结束

    def stop(self):
        """安全停止线程"""
        self.running=False

    def reset(self):
        """安全停止线程"""
        self.running=True

#人脸随动线程
class MoveFaceThread(QThread):

    update_ready_signal=pyqtSignal(dict) #可更新信号，字典类型

    def __init__(self): #连接成功才能检测
        super().__init__()
        self.capture=None #连接的摄像头
        self.movewithface=MovewithFace() #人脸随动实例化
        self.running=True #标志运行情况

    def trans_camera(self,capture):
        """传递摄像头"""
        self.capture=capture

    def run(self):
        while self.running:
            try:
                ret,frame=self.capture.read()  #从摄像头读取一帧
                if not ret:
                    self.update_ready_signal.emit({'status':'error stop'}) #发送错误停止
                    break

                #计算离中间点的距离
                frame,velocity,isdetected=self.movewithface.upgrade_center(frame)
                engine=self.movewithface.generate_engine(velocity)
            
                self.update_ready_signal.emit({'status':'upgrade','img':frame,'result':engine}) #发送标注图片和检测结果
                
                time.sleep(0.1) #100ms ##TODO 检测时间搞短一点

            except Exception as e:
                self.update_ready_signal.emit({'status':'error stop'}) #发送错误停止
                print(e)
                break

        if not self.running: #因为安全停止running是false，所以会发送正常停止信号；而错误停止的running是true，所以不会发正常停止信号
            self.update_ready_signal.emit({'status':'natural stop'}) #发送正常停止信号，正常结束

    def stop(self):
        """安全停止线程"""
        self.running=False

    def reset(self):
        """安全停止线程"""
        self.running=True


#视觉检测窗口
class VisualGUI(QMainWindow):

    visual_signal=pyqtSignal(str) #视觉检测发给主窗口的信号，字符

    def __init__(self):
        super().__init__()
        self.is_connected=False #指示摄像头连接
        self.capture=None #连接的摄像头
        self.decive_id=None #连接的摄像头id

        #检测
        self.is_detect=False #指示检测结果
        self.detect_thread=DetectThread() #实例化检测线程
        self.detect_thread.update_ready_signal.connect(self.receive_thread_signal_detect) #接受线程信号时触发的函数
        self.detect_result=[] #存储检测结果

        #人脸随动
        self.is_move_face=False #指示是否人脸随动
        self.move_face_thread=MoveFaceThread() #实例化人脸随动线程
        self.move_face_thread.update_ready_signal.connect(self.receive_thread_signal_move_face) #接受线程信号时触发的函数
        self.move_face_result=[] #存储要发送给控制器的引擎变量
        
        self.calibration_points=[]
        
        #信息显示的文本格式
        self.blue_format=QTextCharFormat() 
        self.blue_format.setForeground(QColor("blue")) #蓝色
        self.green_format=QTextCharFormat() 
        self.green_format.setForeground(QColor("green")) #绿色
        self.black_format=QTextCharFormat() 
        self.black_format.setForeground(QColor("black")) #黑色

        self.initUI()

    def initUI(self):
        self.setWindowTitle('视觉检测') #设置标题
        self.setGeometry(300,300,1200,1000) #设置窗口位置和大小

        #主面板
        central_widget=QWidget() #主面板
        self.setCentralWidget(central_widget)
        layout=QVBoxLayout(central_widget) #垂直布局

        ##摄像头设置组
        camera_group=QGroupBox('摄像头设置') #摄像头设置的功能组
        camera_group.setFixedHeight(100)
        camera_layout=QHBoxLayout() #水平布局

        ###可选摄像头
        self.camera_combo_label=QLabel('摄像头:')
        self.camera_combo_label.setFixedWidth(80) #最大宽度
        self.camera_combo=QComboBox() #摄像头列表下拉框
        self.refresh_cameras() #刷新一下

        ###刷新率
        self.refresh_rate_combo_label=QLabel('刷新率:')
        self.refresh_rate_combo_label.setFixedWidth(80) #最大宽度
        self.refresh_rate_combo=QComboBox() #刷新率列表下拉框
        self.refresh_rate_combo.addItems(['15','30','100','200']) #添加选项
        self.refresh_rate_combo.setCurrentText('30') #设置默认刷新率
        
        ###连接按钮
        self.connect_btn=QPushButton('连接')
        self.connect_btn.clicked.connect(self.toggle_connection) #绑定回调函数
        
        ###刷新按钮
        refresh_btn=QPushButton('刷新')
        refresh_btn.clicked.connect(self.refresh_cameras) #绑定回调函数

        camera_layout.addWidget(self.camera_combo_label) #放置摄像头设置组部件到布局
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.refresh_rate_combo_label)
        camera_layout.addWidget(self.refresh_rate_combo)
        camera_layout.addWidget(QLabel('毫秒'))
        camera_layout.addWidget(self.connect_btn)
        camera_layout.addWidget(refresh_btn)
        camera_group.setLayout(camera_layout) #为摄像头设置组设置垂直布局

        ##图片显示组
        img_group=QGroupBox('画面') #显示画面和开始检测的功能组
        img_group.setMinimumHeight(700) #最小高度
        img_layout=QVBoxLayout() #垂直布局

        ###画面
        self.img_label=QLabel(self) 
        #self.img_label.setMinimumHeight(600) #最小高度
        self.img_label.setText("等待连接摄像头...") #默认文字
        self.img_label.setAlignment(Qt.AlignCenter) #居中展示
        self.img_label.setStyleSheet("background-color: lightgray;") #设置背景颜色，浅灰

        self.timer=QTimer() #定时器更新画面
        self.timer.timeout.connect(self.update_frame)

        ###按钮组
        detect_btn_layout=QHBoxLayout() #水平布局

        ####开始检测按钮
        self.detect_btn=QPushButton('开始检测')
        self.detect_btn.clicked.connect(self.toggle_detect) #绑定回调函数

        ####人脸随动按钮
        self.move_face_btn=QPushButton('开始人脸随动')
        self.move_face_btn.clicked.connect(self.toggle_move_face) #绑定回调函数

        detect_btn_layout.addWidget(self.detect_btn)
        detect_btn_layout.addWidget(self.move_face_btn)

        img_layout.addWidget(self.img_label)
        img_layout.addLayout(detect_btn_layout)
        img_group.setLayout(img_layout) #为图片显示组设置垂直布局

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
        self.signal_combo.addItems(['当前检测结果','开始标定']) #添加选项
        self.signal_combo.setCurrentText('当前检测结果') #设置默认指令

        ###窗口连接按钮
        self.signal_send_btn=QPushButton('发送')
        self.signal_send_btn.clicked.connect(self.send_choose_signal) #绑定回调函数

        signal_layout.addWidget(self.signal_log,0,0,1,4) #把信息传递组的几个组件添加到该垂直布局
        signal_layout.addWidget(self.signal_combo_label,1,0)
        signal_layout.addWidget(self.signal_combo,1,1,1,2)
        signal_layout.addWidget(self.signal_send_btn,1,3)

        signal_group.setLayout(signal_layout) #将这个表格布局设置为信息传递组的布局

        layout.addWidget(camera_group) #把图片和摄像头控制组添加到窗口
        layout.addWidget(img_group)
        layout.addWidget(signal_group)

    def closeEvent(self, event):
        """关闭窗口时的处理函数"""
        if self.capture and self.capture.isOpened():
            self.capture.release()
            print("摄像头已关闭")
        self.destroyed.emit() #发出销毁信号.没有销毁
        self.destroy()

    ############################## 摄像头设置 ##############################
    def refresh_cameras(self): #最多显示10个设备
        """刷新可用摄像头列表,应用新刷新率"""
        self.camera_combo.clear() #清空可选摄像头
        #self.camera_combo.addItem(str(3)) #测试用
        for device_id in range(10):
            if device_id==self.decive_id:
                continue #跳过当前正在使用的摄像头,也不添加进下拉框，不然冲突了会一直报错
            cap=cv2.VideoCapture(device_id) #尝试连接
            ret,_=cap.read() #捕获一帧
            cap.release() #释放
            if not ret: #如果没有捕获到就直接停止循环，因为是递增的 终端打印error是正常的
                break
            self.camera_combo.addItem(str(device_id)) #有内容，添加至下拉框

        if self.is_connected and self.capture and self.capture.isOpened() and not self.is_detect: #如果正在展示画面且不是检测中
            self.timer.start(int(self.refresh_rate_combo.currentText())) #应用新刷新率

    def toggle_connection(self):
        """切换摄像头连接状态"""
        if not self.is_connected: #未连接摄像头
            self.connect_camera() #连接摄像头
            self.timer.start(int(self.refresh_rate_combo.currentText())) #开始显示画面，按所设置刷新率
            self.refresh_cameras() #刷新一次可用摄像头
        else: #已连接
            self.timer.stop() #停止显示画面
            self.img_label.setText("等待连接摄像头...") #更新画面提示文字
            self.disconnect_camera() #断开摄像头
            self.refresh_cameras() #刷新一次可用摄像头

    def connect_camera(self):
        """连接摄像头"""
        try:
            device_id=int(self.camera_combo.currentText()) #从可用摄像头下拉框获取摄像头
            self.capture=cv2.VideoCapture(device_id) #尝试打开摄像头

            if self.capture.isOpened(): #成功打开
                self.is_connected=True
                self.decive_id=device_id
                self.connect_btn.setText('断开') #更新连接按钮文本
                self.detect_btn.setEnabled(True) #连接上摄像头才允许连接
                self.statusBar().showMessage(f'已连接到 {device_id}') #更新最下方状态栏文本
                QMessageBox.information(self, '连接状态', '摄像头连接成功！') #弹框
            else:
                raise RuntimeError("摄像头未能成功打开")
        except Exception as e: #摄像头打开失败
            if self.capture:
                self.capture.release() #释放
            QMessageBox.warning(self, '连接错误', f'无法连接摄像头：{str(e)}') #弹框

    def disconnect_camera(self):
        """断开摄像头"""
        if self.capture: #若当前已连接摄像头，先关闭
            self.capture.release()
            self.capture=None
        self.is_connected=False
        self.decive_id=None
        self.connect_btn.setText('连接') #更新连接按钮文本
        self.detect_btn.setEnabled(False) #断开摄像头也不允许检测
        self.statusBar().showMessage('未连接') #更新状态栏
    ############################## 摄像头设置 ##############################

    ############################### 画面处理 ###############################
    def update_frame(self):
        """更新视频帧"""
        if self.is_connected and self.capture and self.capture.isOpened():
            ret,frame=self.capture.read()  #从摄像头读取一帧
            if not ret:
                self.timer.stop() #停止更新画面
                self.img_label.setText("无法读取摄像头内容！请重新连接") #设置画面提示文字
                self.disconnect_camera() #断开摄像头
                self.refresh_cameras() #刷新一次串口
                return
            
            #转换为RGB格式
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            #转换为QImage
            height,width,channel=frame.shape
            bytes_per_line=3*width
            q_image=QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            #显示到QLabel
            pixmap=QPixmap.fromImage(q_image).scaled(self.img_label.width(),
                                                     self.img_label.height(),
                                                     Qt.KeepAspectRatio) #等比缩放
            self.img_label.setPixmap(pixmap)

    def toggle_detect(self):
        """切换检测状态"""
        if not self.is_connected: #未连接摄像头
            QMessageBox.warning(self,'警告','请先连接摄像头') #弹框
        else:
            if not self.is_detect: #未开启检测
                self.timer.stop() #停止画面的显示
                self.img_label.setText("检测中") #设置画面提示文字
                self.detect_thread.trans_camera(self.capture) #把摄像头对象传递给子线程
                self.detect_thread.start() #开始运行线程
                self.detect_btn.setText("停止检测") #切换按钮状态
                self.is_detect=True
            else: #已连接，计时器的重启通过信号控制，画面提示文字也是
                self.detect_thread.stop() #停止运行线程，调用自定义方法
                self.detect_btn.setText("开始检测") #切换按钮状态

    def receive_thread_signal_detect(self,signal):
        """接受来自线程的信号"""
        if signal['status']=='natural stop': #正常停止的情况
            self.img_label.setText("等待摄像头内容……") #设置画面提示文字
            self.timer.start(int(self.refresh_rate_combo.currentText())) #重启计时器
            self.is_detect=False
            self.detect_thread.reset()
        elif signal['status']=='error stop': #错误停止，摄像头断了
            self.img_label.setText("无法读取摄像头内容！请重新连接") #设置画面提示文字
            self.disconnect_camera() #断开摄像头
            self.refresh_cameras() #刷新一次串口
            self.is_detect=False
        elif signal['status']=='upgrade': #更新窗口
            #转换为RGB格式
            frame=cv2.cvtColor(signal['img'],cv2.COLOR_BGR2RGB)

            #转换为QImage
            height,width,channel=frame.shape
            bytes_per_line=3*width
            q_image=QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            #显示到QLabel
            pixmap=QPixmap.fromImage(q_image).scaled(self.img_label.width(),
                                                     int(self.img_label.width()*0.6),
                                                     Qt.KeepAspectRatio) #强制缩放
            self.img_label.setPixmap(pixmap)

            self.detect_result=signal['result'] #存储检测结果

    def toggle_move_face(self):
        """切换人脸随动检测"""
        if not self.is_connected: #未连接摄像头
            QMessageBox.warning(self,'警告','请先连接摄像头') #弹框
        else:
            if not self.is_move_face: #未开启人脸随动
                self.timer.stop() #停止画面的显示
                self.img_label.setText("人脸随动初始化中") #设置画面提示文字
                
                signal='start_move_with_face'
                self.send_signal(signal) #给控制器发送人脸随动开始信号

                self.move_face_thread.trans_camera(self.capture) #把摄像头对象传递给子线程
                self.move_face_thread.start() #开始运行线程
                self.move_face_btn.setText("停止人脸随动") #切换按钮状态
                self.is_move_face=True
            else: #已连接，计时器的重启通过信号控制，画面提示文字也是
                self.move_face_thread.stop() #停止运行线程，调用自定义方法
                self.move_face_btn.setText("开始人脸随动") #切换按钮状态

    def receive_thread_signal_move_face(self,signal):
        """接受人脸随动线程信号"""
        if signal['status']=='natural stop': #正常停止的情况
            self.img_label.setText("等待摄像头内容……") #设置画面提示文字
            self.timer.start(int(self.refresh_rate_combo.currentText())) #重启计时器
            self.is_move_face=False
            self.move_face_thread.reset()
        elif signal['status']=='error stop': #错误停止，摄像头断了
            self.img_label.setText("无法读取摄像头内容！请重新连接") #设置画面提示文字
            self.disconnect_camera() #断开摄像头
            self.refresh_cameras() #刷新一次串口
            self.is_move_face=False
        elif signal['status']=='upgrade': #更新窗口
            #转换为RGB格式
            frame=cv2.cvtColor(signal['img'],cv2.COLOR_BGR2RGB)

            #转换为QImage
            height,width,channel=frame.shape
            bytes_per_line=3*width
            q_image=QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            #显示到QLabel
            pixmap=QPixmap.fromImage(q_image).scaled(self.img_label.width(),
                                                     int(self.img_label.width()*0.6),
                                                     Qt.KeepAspectRatio) #强制缩放
            self.img_label.setPixmap(pixmap)

            self.move_face_result=signal['result'] #存储检测结果

            #编写指令发送给控制器窗口
            send_signal='move_face engine:'+str(self.move_face_result) #编写指令
            self.send_signal(send_signal)
    ############################### 画面处理 ###############################

    ############################### 标定相关 ###############################
    def calibration(self):
        """开始标定"""
        self.is_start_calibration=True #允许修改标定结果
        #print(self.is_start_calibration) #测试用

    def add_calibration(self,calibration_data):
        """处理四个标定数据"""
        self.calibration_points.append(calibration_data) #添加进去

    ############################### 标定相关 ###############################

    ############################### 通信相关 ###############################
    def send_signal(self,signal):
        """发送信号给主窗口"""
        self.visual_signal.emit(signal) #发送信息

        self.show_signal(signal=signal,mode=2) #展示信息

    def receive_signal(self,signal):
        """接受主窗口转发的控制器信号"""
        self.show_signal(signal=signal,mode=1) #展示信息

        if signal.startswith('calibration point:'):
            data_str=signal[len('calibration point:'):] #去头
            try:
                calibration_data=eval(data_str)
                if isinstance(calibration_data, list):
                    self.add_calibration(calibration_data)
                else:
                    print("Error: Data is not a list.")
            except Exception as e:
                print(f"Failed to parse calibration data: {e}")
        if signal=='calibration done':
            self.detect_thread.colorcylinderdetecter.calibration_points=self.calibration_points #如果完成标定，传递标定点
            #self.detect_thread.update_calibration([[221.176,130.626],[193.288,-106.53800],[78.409,-84.7644],[80.5877,122.2352]]) #测试用
            visual_signal='calibration done'
            self.send_signal(visual_signal)
            self.is_start_calibration=False

    def show_signal(self,signal,mode=0):
        """将信息展示到文本框中"""
        if mode==0: #系统提示文本
            cursor=self.signal_log.textCursor() #获取当前文本光标
            cursor.insertBlock() #换行
            cursor.insertText(signal,self.black_format) #具体信息设为黑色
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

        if signal_choose=='当前检测结果':
            signal='detect_result:'+str(self.detect_result) 

            self.send_signal(signal)
        elif signal_choose=='开始标定':
            signal='start_calibration'
            self.calibration() #标定函数
            self.send_signal(signal)
        else:
            self.send_signal(signal_choose)
    ############################### 通信相关 ###############################


#主函数
def main():
    app=QApplication(sys.argv) #负责应用事件循环
    window=VisualGUI() #自定义的窗口
    window.show() #展示窗口
    sys.exit(app.exec_()) #启动事件循环，满足退出条件的时候就结束事件循环

if __name__ == '__main__':
    main()