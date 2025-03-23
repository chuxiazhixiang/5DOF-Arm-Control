import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class RoboticArmKinematics:
    '''机械臂运动学'''
    def __init__(self):
        #机械臂参数
        self.d1=80      #底座高度
        self.a1=-10     #大臂偏移
        self.a2=85   #大臂长度
        self.a3=80   #小臂长度
        self.d5=100    #手腕长度
        self.d6=100    #爪子长度
        
        #DH参数表 [alpha(i-1),a(i-1),d(i),theta(i)]
        self.dh_params=np.array([
            [0,        0,         self.d1,   0],   #1: 底座旋转
            [np.pi/2,  self.a1,   0,         np.pi/2],   #2: 大臂
            [0,        self.a2,   0,         0],   #3: 小臂
            [0,        self.a3,   0,         np.pi/2],   #4: 手腕
            [np.pi/2,  0,         self.d5,   np.pi/2],   #5: 爪子旋转
            [0,        0,         self.d6,   0] #爪子末端
        ]) 

        #这舵机500和2500的时候不是对应的准的，而且还是270度的，所以搞个插值！
        self.joint_angle={
            0:{"servo_values":[2295,1500,814],"angle":np.radians([90,0,-90])},
            #1:{"servo_values":[2266,1900,1500,1203,1000,882],"angle":np.radians([90,45,0,-45,-60,-90])},
            1:{"servo_values":[2266,1900,1500,1203,882],"angle":np.radians([90,45,0,-45,-90])},
            2:{"servo_values":[2500,2212,1500,799,500],"angle":np.radians([-135,-90,0,90,130])},
            3:{"servo_values":[2500,2246,1500,884,500],"angle":np.radians([-130,-90,0,90,135])},
            4:{"servo_values":[2500,2245,1500,822,500],"angle":np.radians([135,90,0,-90,-135])}
        }

        #舵机角度限制
        self.joint_limits=[
            (-2.290,1.976), #Joint 0
            #(-2.505,2.073), #Joint 1 舵机最大可到达角度，但是会被挡住，所以用
            (-np.pi/2,np.pi/2),#joint 1
            (-2.356,2.269), #Joint 2 
            (-2.269,2.356), #Joint 3
            (-2.356,2.356), #Joint 4
        ]

    def transform_matrix(self,alpha,a,d,theta):
        """计算DH变换矩阵
        alpha(i-1),a(i-1),d(i),theta(i)
        """
        #转换角度到弧度
        
        ct=np.cos(theta)
        st=np.sin(theta)
        ca=np.cos(alpha)
        sa=np.sin(alpha)
        
        #根据图中给出的矩阵形式
        return np.array([
            [ct,       -st,       0,         a],
            [st*ca,    ct*ca,    -sa,       -sa*d],
            [st*sa,    ct*sa,    ca,        ca*d],
            [0,        0,         0,         1]
        ])

    def forward_kinematics(self,joint_angles): #传入的角度应该都在-π到π之间！！！
        """正运动学计算"""
        T=np.eye(4)
        positions=[]
        
        #基座位置
        positions.append(np.array([0,0,0]))
        
        #计算每个关节的位置
        for i in range(len(joint_angles)): #传入前5个，最后控制夹爪的不在考虑范围内
            alpha=self.dh_params[i][0]
            a=self.dh_params[i][1]
            d=self.dh_params[i][2]
            theta=joint_angles[i]+self.dh_params[i][3]#由pwm转换成正方向角度传进来的
            
            Ti=self.transform_matrix(alpha,a,d,theta)
            T=T@Ti
            
            position=T[:3,3]
            positions.append(position)

        alpha=self.dh_params[5][0]
        a=self.dh_params[5][1]
        d=self.dh_params[5][2]
        theta=self.dh_params[5][3]
        T6=self.transform_matrix(alpha,a,d,theta)
        T=T@T6 #夹爪的
        position=T[:3,3]
        positions.append(position) #末端夹爪位置放进去   #??要放吗
        
        #计算末端姿态角,就取一个yaw和一个roll,我的joint_angle是纯正方向角度
        yaw=-(joint_angles[1]+joint_angles[2]+joint_angles[3]) #从z0方向往x0方向转的 正180是往外转到垂直向下，负180是往内转到垂直向下，但是一般到不了负180
        roll=joint_angles[4] #为0的时候就是初始的
         
        return positions,yaw,roll
    
    def inverse_kinematics(self,target_pos,yaw,roll=0,epsilon=1e-5): #就不考虑roll了，默认为0把，如果没有yaw限制就搞成false，有就搞成具体数值
        """逆运动学计算"""
        result=[] #存储两组解

        x,y,z=target_pos
        #print(f"目标位置: x={x:.1f},y={y:.1f},z={z:.1f}")

        #x=x-self.a1 #去掉第二个关节的偏移量,不能在这边减去
        z=z-self.d1 #去掉底座高度

        theta=np.arctan2(y,x) #底座角度，第一个舵机

        #这里有两种可能，一种是直接正平面到达，另一种是负平面到达，然后转180度
        for i in range(2):
            if i==0: #正平面到达的情况
                theta0=theta
                r=(x**2+y**2)**0.5-self.a1 #c是正的

                #给了yaw
                m=r-(self.d5+self.d6)*np.sin(yaw)
                n=z-(self.d5+self.d6)*np.cos(yaw)

                ##我草，arcsin的范围在-pi/2-pi/2，我的角度可能是-pi-pi的！，我擦不能直接用arcsin

                #先算theta1的情况
                k1=(self.a3**2-self.a2**2-m**2-n**2)/(-2*self.a2)
                a1=m**2+n**2
                b1=-2*k1*m
                c1=k1**2-n**2
                delta=b1**2-4*a1*c1
                if abs(delta)<epsilon:
                    delta=0
                if delta<0: #一元二次方程无解
                    continue
                
                sign=np.sign(m*n) #mn的符号会影响结果！！！！！
                    
                sintheta1_big=(-b1+delta**0.5)/(2*a1) #较大的解
                costheta1_big_mn=(k1-sintheta1_big*m)*m
                sintheta1_big_mn=sintheta1_big*m*n
                theta1_big=-np.arctan2(sintheta1_big_mn*sign,costheta1_big_mn*sign)

                sintheta1_small=(-b1-delta**0.5)/(2*a1) #较小的解
                costheta1_small_mn=(k1-sintheta1_small*m)*m
                sintheta1_small_mn=sintheta1_small*m*n
                theta1_small=-np.arctan2(sintheta1_small_mn*sign,costheta1_small_mn*sign)

                #再算theta2的情况
                k12=(self.a2**2-self.a3**2-m**2-n**2)/(-2*self.a3)
                a12=m**2+n**2
                b12=-2*k12*m
                c12=k12**2-n**2
                delta=b12**2-4*a12*c12
                if abs(delta)<epsilon:
                    delta=0
                if delta<0: #一元二次方程无解
                    continue
                sintheta12_big=(-b12+delta**0.5)/(2*a12) #较大的解
                costheta12_big_mn=(k12-sintheta12_big*m)*m
                sintheta12_big_mn=sintheta12_big*m*n
                theta12_big=-np.arctan2(sintheta12_big_mn*sign,costheta12_big_mn*sign)

                sintheta12_small=(-b12-delta**0.5)/(2*a12) #较小的解
                costheta12_small_mn=(k12-sintheta12_small*m)*m
                sintheta12_small_mn=sintheta12_small*m*n
                theta12_small=-np.arctan2(sintheta12_small_mn*sign,costheta12_small_mn*sign)

                #计算3个角度，theta12大的对应theta1小的
                theta1_1=theta1_big
                theta2_1=self.normalize_angle(theta12_small-theta1_big)
                theta3_1=self.normalize_angle(-yaw-theta1_1-theta2_1)
                result1=[theta0,theta1_1,theta2_1,theta3_1,roll] #theta4默认为0
            
                fk_positions1,fk_yaw1,fk_roll1=self.forward_kinematics(result1) #计算正运动学结果
                #print('正运动学结果1',fk_positions1,'位姿',fk_yaw1,',',fk_roll1) #测试用
                ##一些舍去解的判断
                #if not ???:
                ##continue
                result.append(result1)

                theta1_2=theta1_small
                theta2_2=self.normalize_angle(theta12_big-theta1_small)
                theta3_2=self.normalize_angle(-yaw-theta1_2-theta2_2)
                result2=[theta0,theta1_2,theta2_2,theta3_2,roll]

                fk_positions2,fk_yaw2,fk_roll2=self.forward_kinematics(result2) #计算正运动学结果
                #print('正运动学结果2',fk_positions2,'位姿',fk_yaw2,',',fk_roll2) #测试用
                ##一些舍去解的判断
                ##if not???:
                result.append(result2)


            else: #负平面到达的情况
                theta0=theta-np.sign(theta)*np.pi #另一种角度
                r=-(x**2+y**2)**0.5-self.a1 #c是负的

                m=-r-(self.d5+self.d6)*np.sin(-yaw)
                n=z-(self.d5+self.d6)*np.cos(-yaw)

                #先算theta1的情况
                k1=(self.a3**2-self.a2**2-m**2-n**2)/(-2*self.a2)
                a1=m**2+n**2
                b1=-2*k1*m
                c1=k1**2-n**2
                delta=b1**2-4*a1*c1
                delta=m**2+n**2-k1**2
                if abs(delta)<epsilon: #因为计算机的阶段误差，可能会有超小数，在写的过程中有出现delta算出来是-7.27*1e-12的，但他实际上应该是0
                    delta=0
                if delta<0: #一元二次方程无解
                    continue

                sign=np.sign(m*n) #mn的符号会影响结果！！！！！

                sintheta1_big=(-b1+delta**0.5)/(2*a1) #较大的解
                costheta1_big_mn=(k1-sintheta1_big*m)*m
                sintheta1_big_mn=sintheta1_big*m*n #因为cos算的时候要除n，如果n很小就完蛋了，所以都乘mn
                theta1_big=np.arctan2(sintheta1_big_mn*sign,costheta1_big_mn*sign) #我草，mn的符号还会影响结果的

                sintheta1_small=(-b1-delta**0.5)/(2*a1) #较小的解
                costheta1_small_mn=(k1-sintheta1_small*m)*m
                sintheta1_small_mn=sintheta1_small*m*n
                theta1_small=np.arctan2(sintheta1_small_mn*sign,costheta1_small_mn*sign)

                #再算theta2的情况
                k12=(self.a2**2-self.a3**2-m**2-n**2)/(-2*self.a3)
                a12=m**2+n**2
                b12=-2*k12*m
                c12=k12**2-n**2
                delta=b12**2-4*a12*c12
                if abs(delta)<epsilon:
                    delta=0
                if delta<0: #一元二次方程无解
                    continue
                sintheta12_big=(-b12+delta**0.5)/(2*a12) #较大的解
                costheta12_big_mn=(k12-sintheta12_big*m)*m
                sintheta12_big_mn=sintheta12_big*m*n
                theta12_big=np.arctan2(sintheta12_big_mn*sign,costheta12_big_mn*sign)

                sintheta12_small=(-b12-delta**0.5)/(2*a12) #较小的解
                costheta12_small_mn=(k12-sintheta12_small*m)*m
                sintheta12_small_mn=sintheta12_small*m*n
                theta12_small=np.arctan2(sintheta12_small_mn*sign,costheta12_small_mn*sign)

                #计算3个角度，theta12大的对应theta1小的
                theta1_1=theta1_big
                theta2_1=self.normalize_angle(theta12_small-theta1_big)
                theta3_1=self.normalize_angle(-yaw-theta1_1-theta2_1)
                result1=[theta0,theta1_1,theta2_1,theta3_1,roll] #theta4默认为0
            
                fk_positions1,fk_yaw1,fk_roll1=self.forward_kinematics(result1) #计算正运动学结果
                #print('正运动学结果3',fk_positions1,'位姿',fk_yaw1,',',fk_roll1) #测试用
                ##一些舍去解的判断
                #if not ???:
                ##continue
                result.append(result1)

                theta1_2=theta1_small
                theta2_2=self.normalize_angle(theta12_big-theta1_small)
                theta3_2=self.normalize_angle(-yaw-theta1_2-theta2_2)
                result2=[theta0,theta1_2,theta2_2,theta3_2,roll]

                fk_positions2,fk_yaw2,fk_roll2=self.forward_kinematics(result2) #计算正运动学结果
                #print('正运动学结果4',fk_positions2,'位姿',fk_yaw2,',',fk_roll2) #测试用
                ##一些舍去解的判断
                ##if not???:
                result.append(result2)
            
        result=self.remove_duplicates(result) #去重复解
        result=self.remove_beyond_engine(result) #去超出舵机范围和角度限制的

        #print('逆运动学解个数:',len(result)) #测试用
        if len(result)==0:
            print('逆运动学无解')
            return [],False
        
        return result,True
    
    def remove_duplicates(self,solutions,tolerance=0.09):
        """解去重"""
        unique_solutions=[]
        for solution in solutions:
            if not any(np.allclose(solution,unique,atol=tolerance) for unique in unique_solutions):
                unique_solutions.append(solution)
        return unique_solutions
    
    def remove_beyond_engine(self,solutions):
        """去除超出舵机范围和角度限制的"""
        valid_solutions=[
            solution for solution in solutions
            if all((self.joint_limits[i][0]-0.14)<=angle<=(self.joint_limits[i][1]+0.14) for i,angle in enumerate(solution))
        ] #这里范围给放宽了一些
        return valid_solutions

    
    def normalize_angle(self,angle):
        """转换到-π和π之间"""
        return (angle+np.pi)%(2*np.pi)-np.pi
    
    #转换，纯转换，给进去的就一定是有效的，无效的判断放到前面
    def pwm_to_angle(self,pwm_values):
        """PWM值转角度"""
        angles=[]
        for i,pwm in enumerate(pwm_values):
            servo_values=self.joint_angle[i]["servo_values"]
            angle_values=self.joint_angle[i]["angle"]
            
            #使用 scipy 的插值方法
            interp_func=interp1d(servo_values,angle_values,kind='linear',fill_value='extrapolate')
            angle=interp_func(pwm)
            angles.append(angle)
        return angles

    def angle_to_pwm(self,angles):
        """角度转PWM值"""
        pwm_values=[]
        for i,angle in enumerate(angles):
            servo_values=self.joint_angle[i]["servo_values"]
            angle_values=self.joint_angle[i]["angle"]
            
            #使用 scipy 的插值方法
            interp_func=interp1d(angle_values,servo_values,kind='linear',fill_value='extrapolate')
            pwm=interp_func(angle)
            pwm_values.append(int(max(500,min(2500,pwm))))  #限制 PWM 范围为 500~2500
        return pwm_values

    def plot_interpolation(self):
        """绘制每个关节的PWM与角度的插值关系"""
        pwm_values=np.linspace(500,2500,100)  #生成从500到2500的100个PWM值
        plt.figure(figsize=(12,8))

        for i in range(5):  #处理5个关节
            servo_values=self.joint_angle[i]["servo_values"]
            angle_values=self.joint_angle[i]["angle"]
            
            #使用 scipy 插值
            interp_func=interp1d(servo_values,angle_values,kind='linear',fill_value='extrapolate')
            interpolated_angles=interp_func(pwm_values)

            #计算插值上下界
            min_angle=np.min(interpolated_angles)
            max_angle=np.max(interpolated_angles)
            print(f"Joint {i}: Interpolation range -> Min: {min_angle:.3f} rad,Max: {max_angle:.3f} rad") #给上下界

            #绘图
            plt.subplot(3,2,i+1)
            plt.plot(pwm_values,interpolated_angles,label="Interpolated Curve")
            plt.scatter(servo_values,angle_values,color='red',label="Data Points")
            plt.title(f"Joint {i} PWM to Angle")
            plt.xlabel("PWM Values")
            plt.ylabel("Angle (radians)")
            plt.legend()

        plt.tight_layout()
        plt.show()
    
if __name__=='__main__':

    k=RoboticArmKinematics()
    #k.plot_interpolation()
    joint_pwm=[1500,1430,1500,1500,1500]
    joint_angle=k.pwm_to_angle(joint_pwm)
    print('pwm2angle舵机转角度',joint_angle)
    joint_pwm1=k.angle_to_pwm(joint_angle)
    print('angle2pwm角度转舵机',joint_pwm1)

    popsition,yaw,roll=k.forward_kinematics(joint_angle)
    print('正运动学位置',popsition[-1])
    print('正运动学yaw',yaw)

    position=popsition[-1]
    
    #position=[0,0,150]
    #yaw=0
    #roll=0

    joint_angle1,label=k.inverse_kinematics(position,yaw,roll)
    for i,value in enumerate(joint_angle1):
        print('逆运动学解',i,'角度结果',value)
        joint_pwm1=k.angle_to_pwm(value)
        print('逆运动学解',i,'舵机位置',joint_pwm1)
        position,yaw,roll=k.forward_kinematics(value)
        print('正运动学',i,'位置',position[-1],'角度yaw',yaw)