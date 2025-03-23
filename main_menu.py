import sys
from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,
                           QHBoxLayout,QLabel,QSlider,QPushButton,QComboBox,
                           QTabWidget,QGroupBox,QLineEdit,QTableWidget,
                           QTableWidgetItem,QMessageBox,QGridLayout,
                           QTextEdit)
from PyQt5.QtCore import Qt,QTimer,pyqtSignal

from src.controller_menu import RoboticArmGUI #控制器窗口
from src.visual_menu import VisualGUI #视觉检测窗口

#主窗口
class MainGUI(QMainWindow):

    signal_to_controller=pyqtSignal(str) #从控制器接受信号
    signal_to_visual=pyqtSignal(str) #从视觉检测接受信号

    def __init__(self):
        super().__init__()
        self.subwindows={'controller':None,'visual':None} #存储打开的子窗口
        self.initUI()

    def initUI(self):
        self.setWindowTitle('5DOF机械臂系统')
        self.setGeometry(300, 300, 400, 300) #设置窗口位置和大小

        #主面板
        central_widget=QWidget() #主面板
        self.setCentralWidget(central_widget) #设置中心部件
        layout=QVBoxLayout(central_widget) #垂直布局

        ##控制器按钮
        controller_btn=QPushButton('控制器')
        controller_btn.setFixedWidth(200) #设置宽度
        controller_btn.setFixedHeight(50) #设置高度
        controller_btn.clicked.connect(self.open_controller) #绑定回调函数

        ##视觉按钮
        visual_btn=QPushButton('视觉检测')
        visual_btn.setFixedWidth(200) #设置宽度
        visual_btn.setFixedHeight(50) #设置高度
        visual_btn.clicked.connect(self.open_visual) #绑定回调函数
        
        layout.setSpacing(10) #控件之间距离
        layout.addWidget(QLabel('5DOF机械臂系统:'),alignment=Qt.AlignCenter)
        layout.addWidget(controller_btn,alignment=Qt.AlignCenter)
        layout.addWidget(visual_btn,alignment=Qt.AlignCenter)

    def open_controller(self):
        """打开控制器窗口"""
        if self.subwindows['controller'] is None: #当子窗口没打开
            try:
                self.subwindows['controller']=RoboticArmGUI() #实例化控制器窗口
                self.subwindows['controller'].destroyed.connect(self.on_controller_closed) #绑定回调函数
                self.subwindows['controller'].controller_signal.connect(self.controller_to_visual) #绑定槽函数，当控制器的控制信号发出时，调用主窗口的转发函数
                self.signal_to_controller.connect(self.subwindows['controller'].receive_signal) #绑定槽函数，当主窗口转发信息给控制器时，调用控制器的接受函数
                self.subwindows['controller'].show() #显示控制器窗口
                self.controller_flag=True
            except Exception as e:
                QMessageBox.warning(self, '控制器窗口打开失败', f'无法打开控制器窗口。\n错误信息：{str(e)}')
        else: #已打开
            self.subwindows['controller'].raise_() #置顶子窗口

    def on_controller_closed(self):
        """关闭控制器窗口"""
        self.subwindows['controller']=None #当窗口关闭，清理子窗口引用，下次还能打开

    def open_visual(self):
        """打开视觉检测窗口"""
        if self.subwindows['visual'] is None: #当子窗口没打开
            try:
                self.subwindows['visual']=VisualGUI() #实例化控制器窗口
                self.subwindows['visual'].destroyed.connect(self.on_visual_closed) #绑定回调函数
                self.subwindows['visual'].visual_signal.connect(self.visual_to_controller) #绑定槽函数，当视觉检测的视觉信号发出时，调用主窗口的转发函数
                self.signal_to_visual.connect(self.subwindows['visual'].receive_signal) #绑定槽函数，当主窗口转发信息给视觉检测时，调用视觉窗口的接受函数
                self.subwindows['visual'].show() #显示控制器窗口
                self.controller_flag=True
            except Exception as e:
                QMessageBox.warning(self, '视觉检测窗口打开失败',f'无法打开视觉检测窗口。\n错误信息：{str(e)}')
        else: #已打开
            self.subwindows['visual'].raise_() #置顶子窗口

    def on_visual_closed(self):
        """关闭视觉检测窗口"""
        self.subwindows['visual']=None #当窗口关闭，清理子窗口引用，下次还能打开

    ############################### 通信相关 ###############################
    def controller_to_visual(self,signal):
        """转发控制信号给视觉检测"""
        self.signal_to_visual.emit(signal) #转发控制器发送信号给视觉检测

    def visual_to_controller(self,signal):
        """转发视觉检测信号给控制器"""
        self.signal_to_controller.emit(signal) #转发视觉检测发送信号给控制器

    ############################### 通信相关 ###############################

if __name__=="__main__":
    app=QApplication(sys.argv) #负责应用事件循环
    window=MainGUI() #自定义的窗口
    window.show() #展示窗口
    sys.exit(app.exec_()) #启动事件循环，满足退出条件的时候就结束事件循环
