import sys
import serial
import serial.tools.list_ports
import numpy as np
from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,
                           QHBoxLayout,QLabel,QSlider,QPushButton,QComboBox,
                           QTabWidget,QGroupBox,QLineEdit,QTableWidget,
                           QTableWidgetItem,QMessageBox,QGridLayout)
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

#视觉检测窗口
class VisualGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_connected=False #指示摄像头连接
        self.capture=None #连接的摄像头
        self.decive_id=None #连接的摄像头id

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

        ##图片显示
        self.img_label=QLabel(self) 
        self.img_label.setText("等待连接摄像头...") #默认文字
        self.img_label.setAlignment(Qt.AlignCenter) #居中展示
        self.img_label.setStyleSheet("background-color: lightgray;") #设置背景颜色，浅灰

        layout.addWidget(camera_group) #把图片和摄像头控制组添加到窗口
        layout.addWidget(self.img_label)

        self.timer=QTimer() #定时器更新画面
        self.timer.timeout.connect(self.update_frame)

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

        if self.is_connected and self.capture and self.capture.isOpened(): #如果正在展示画面
            self.timer.start(int(self.refresh_rate_combo.currentText())) #应用新刷新率

    def toggle_connection(self):
        """切换摄像头连接状态"""
        if not self.is_connected: #未连接摄像头
            self.connect_camera() #连接摄像头
            self.timer.start(int(self.refresh_rate_combo.currentText())) #开始显示画面，按所设置刷新率
            self.refresh_cameras() #刷新一次串口
        else: #已连接
            self.timer.stop() #停止显示画面
            self.img_label.setText("等待连接摄像头...") #更新画面提示文字
            self.disconnect_camera() #断开摄像头
            self.refresh_cameras() #刷新一次串口

    def connect_camera(self):
        """连接摄像头"""
        try:
            device_id=int(self.camera_combo.currentText()) #从可用摄像头下拉框获取摄像头
            self.capture=cv2.VideoCapture(device_id) #尝试打开摄像头

            if self.capture.isOpened(): #成功打开
                self.is_connected=True
                self.decive_id=device_id
                self.connect_btn.setText('断开') #更新连接按钮文本
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
    ############################### 画面处理 ###############################


#主函数
def main():
    app=QApplication(sys.argv) #负责应用事件循环
    window=VisualGUI() #自定义的窗口
    window.show() #展示窗口
    sys.exit(app.exec_()) #启动事件循环，满足退出条件的时候就结束事件循环

if __name__ == '__main__':
    main()