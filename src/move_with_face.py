import mediapipe as mp
import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time

class MovewithFace(object):
    def __init__(self):
        self.face_detection=mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.previous_center=None #上一帧中心点
        
        self.generate_fuzzy_control() #生成模糊控制器

    def upgrade_center(self,frame):
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results =self.face_detection .process(rgb_frame)
        dx=0
        dy=0
        if results.detections:
            for detection in results.detections:
                bbox=detection.location_data.relative_bounding_box
                h,w,_=frame.shape  #图像尺寸

                #计算中心点（转换为绝对像素坐标）
                center_x=int((bbox.xmin+bbox.width/2)*w)
                center_y=int((bbox.ymin+bbox.height/2)*h)

                #在图像上绘制中心点
                cv2.circle(frame,(center_x,center_y),5,(0,255,0),-1)

                image_center_x=frame.shape[1]/2
                image_center_y=frame.shape[0]/2

                #计算速度分量（假设帧率恒定）
                if self.previous_center is not None:
                    dx=center_x-image_center_x
                    dy=-(center_y-image_center_y) #归一化距离

                    #在画面上显示速度分量
                    cv2.putText(frame,f"Vx: {dx:.2f} px/frame",(10,30),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                    cv2.putText(frame,f"Vy: {dy:.2f} px/frame",(10,60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

            #更新上一帧中心点
            self.previous_center=(center_x,center_y)

            return frame,[dx,dy],True #返回有标注的frame和当前速度
        else:
            return frame,[dx,dy],False
        
    def generate_fuzzy_control(self):
        vx=ctrl.Antecedent(np.arange(-260,260,1),'vx')  #Vx范围：-260到260
        vy=ctrl.Antecedent(np.arange(-233,233,1),'vy')  #Vy范围：-233到233

        theta_x=ctrl.Consequent(np.arange(-170,170,1),'theta_x')  #舵机角度范围：-170到170
        theta_y=ctrl.Consequent(np.arange(-110,90,1),'theta_y')  #舵机角度范围：-110到90

        vx['N']=fuzz.trapmf(vx.universe,[-260,-260,-130,0])  #Negative
        vx['Z']=fuzz.trimf(vx.universe,[-130,0,130])         #Zero
        vx['P']=fuzz.trapmf(vx.universe,[0,130,260,260])     #Positive

        vy['N']=fuzz.trapmf(vy.universe,[-233,-233,-116,0])  #Negative
        vy['Z']=fuzz.trimf(vy.universe,[-116,0,116])         #Zero
        vy['P']=fuzz.trapmf(vy.universe,[0,116,233,233])     #Positive

        theta_x['L']=fuzz.trapmf(theta_x.universe,[-85,-85,-42,0])  #Left
        theta_x['S']=fuzz.trimf(theta_x.universe,[-21,0,21])         #Stop
        theta_x['R']=fuzz.trapmf(theta_x.universe,[0,42,85,85])     #Right

        theta_y['D']=fuzz.trapmf(theta_y.universe,[-55,-55,-22,0])  #Down
        theta_y['S']=fuzz.trimf(theta_y.universe,[-11,0,11])         #Stop
        theta_y['U']=fuzz.trapmf(theta_y.universe,[0,22,55,55])     #Up

        rule1=ctrl.Rule(vx['P'],theta_x['R'])  #如果vx是Positive，theta_x向右
        rule2=ctrl.Rule(vx['N'],theta_x['L'])  #如果vx是Negative，theta_x向左
        rule3=ctrl.Rule(vx['Z'],theta_x['S'])  #如果vx是Zero，theta_x停止

        rule4=ctrl.Rule(vy['P'],theta_y['U'])  #如果vy是Positive，theta_y向上
        rule5=ctrl.Rule(vy['N'],theta_y['D'])  #如果vy是Negative，theta_y向下
        rule6=ctrl.Rule(vy['Z'],theta_y['S'])  #如果vy是Zero，theta_y停止

        self.theta_ctrl=ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6])
        self.theta_simulation=ctrl.ControlSystemSimulation(self.theta_ctrl)

    def generate_engine(self,velocity):
        input_vx=velocity[0]
        input_vy=velocity[1]
        self.theta_simulation.input['vx']=input_vx
        self.theta_simulation.input['vy']=input_vy

        self.theta_simulation.compute()

        engine_x=int(self.theta_simulation.output['theta_x']) #对应engine0
        engine_y=int(self.theta_simulation.output['theta_y']) #对应engine4 转整数
        
        return [engine_x,engine_y]

if __name__=="__main__":
    #测试
    img_moveface=MovewithFace()
    capture=cv2.VideoCapture(1) #尝试打开摄像头
    if not capture.isOpened(): #如果摄像头无法打开
        print("Error:Could not open camera.")
        exit()

    while True:
        ret,frame=capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow('ori_img',frame) #测试用
        frame,velocity,isdetected=img_moveface.upgrade_center(frame)
        print(velocity)
        cv2.imshow('velocity_img',frame)

        time.sleep(0.1)

        engine=img_moveface.generate_engine(velocity)
        print(engine)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()