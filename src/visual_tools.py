import cv2
import numpy as np
import time

#找工作区
class FocusFinder(object):
    def __init__(self,scale=1,allowed_moving_girth=300):
        self.scale=scale #缩放尺寸
        self.allowed_moving_girth=allowed_moving_girth #允许移动周长，物体可变范围
        self.allowed_moving_length=10 #允许移动长度，角点可变范围
        self.pre_corner_point=[(0,0),(0,0),(0,0),(0,0)] #上一次提取角点
        self.pre_max_length=0 #上一次最大长度
        self.is_first=True #是否是第一次处理

    #找物体
    def find_focus(self,img,min_threshold=30,max_threshold=250):
        img=cv2.flip(img,1)  #该摄像头是水平翻转的
        #img=cv2.flip(img,2)  #如果是反过来拍的垂直翻转一下
        #cv2.imshow("ori_img",img) #测试用

        source_img=img.copy()
        #img=cv2.GaussianBlur(img,(3,3),0,0) #高斯模糊，去噪声，不过加了这个，细的线就看不见了

        canny=cv2.Canny(img,min_threshold,max_threshold) #canny算子（高斯梯度），边缘检测，双阈值
        #cv2.imshow("canny",canny) #测试用
        k=np.ones((3,3),np.uint8) #结构元，全1
        canny=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,k) #形态学闭操作 
        #cv2.imshow('canny_close',canny) #测试用

        contours,hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #提取轮廓，简化为直线段
        contours=sorted(contours,key=cv2.contourArea,reverse=True)[:1] #选最大轮廓，工作区边框
        if len(contours)==0:#没找到
            return 0,False
        max_length=abs(cv2.arcLength(contours[0],True)) #最大轮廓周长，保存棋盘周长

        if max_length<600:#找错了
            return 0,False
        temp_caver=np.ones(canny.shape,np.uint8)*255 #空白图像，用来画轮廓
        
        contours1=cv2.approxPolyDP(contours[0],10,True) #多边形逼近
        cv2.drawContours(temp_caver,[contours1],-1,(0,255,0),1) #画空白图像上，绿色
        #cv2.imshow('max_contour',temp_caver) #测试用
        # 备选拐点
        corners=cv2.goodFeaturesToTrack(temp_caver,25,0.1,10) #shi-Tomasi角点检测算法
        if corners is None:#没角点
            return 0,False
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001) #角点亚像素级优化迭代终止条件
        cv2.cornerSubPix(temp_caver,corners,(11,11),(-1,-1),criteria) #角点亚像素级优化
        corners=corners.astype(np.int32) #浮点变整数
        point_list=[]
        for i in corners:
            x,y=i.ravel()
            point_list.append((x,y))

        corner_point=self.find_corner(point_list)  #找到4个顶点
        sort_corner_list=self.sort_corner(corner_point) #角点排序，顺时针

        if self.is_first:#第一次检测情况
            self.pre_corner_point=sort_corner_list
            self.pre_max_length=max_length
            self.is_first=False

        if abs(self.pre_max_length-max_length)>self.allowed_moving_girth:#发生了物体入侵的情况
            # LOG.debug("物体入侵")
            self.pre_max_length=max_length
            return 0,False

        if np.max(abs(np.array(sort_corner_list)-np.array(self.pre_corner_point)))>self.allowed_moving_length:#角点移动太多
            # LOG.warning(f"角点错误:{sort_corner_list},{self.pre_corner_point}")
            self.pre_corner_point=sort_corner_list
            return 0,False

        self.pre_corner_point=sort_corner_list
        self.pre_max_length=max_length

        hight,width=self.calSize(sort_corner_list,self.scale) #原来的四个点，也就是棋盘可能是梯形，或者斜的，给他变成正的长方形
        aim_size=np.float32([[0,0],[width,0],[width,hight],[0,hight]])
        raw_size=[]

        for x,y in sort_corner_list:
            raw_size.append([x,y])

        raw_size=np.float32(raw_size)
        translate_map=cv2.getPerspectiveTransform(raw_size,aim_size) #变换矩阵
        translate_img=cv2.warpPerspective(source_img,translate_map,(int(width),int(hight)))
        #cv2.imshow('translate_img',translate_img) #测试用
        return translate_img,True

    #缩放图片
    def calSize(self,sort_corner_list,scale):
        h1=(sort_corner_list[2][1]-sort_corner_list[1][1])
        h2=(sort_corner_list[3][1]-sort_corner_list[0][1])
        hight=max(h1,h2)*scale

        w1=(sort_corner_list[0][0]-sort_corner_list[1][0])
        w2=(sort_corner_list[3][0]-sort_corner_list[2][0])
        width=max(w1,w2)*scale

        return hight,width

    #角点顺时针排序，冒泡
    def sort_corner(self,corner_point):
        for i in range(len(corner_point)):
            for j in range(i+1,len(corner_point)):
                if corner_point[i][1]>corner_point[j][1]:
                    tmp=corner_point[j]
                    corner_point[j]=corner_point[i]
                    corner_point[i]=tmp
        top=corner_point[:2]
        bot=corner_point[2:]

        if top[0][0]>top[1][0]:
            tmp=top[1]
            top[1]=top[0]
            top[0]=tmp

        if bot[0][0]>bot[1][0]:
            tmp=bot[1]
            bot[1]=bot[0]
            bot[0]=tmp

        tl=top[1]
        tr=top[0]
        bl=bot[0]
        br=bot[1]
        corners=[tl,tr,bl,br]
        return corners

    #算三角形面积，用叉积法
    def area(self,a,b,c):
        return (a[0]-c[0])*(b[1]-c[1])-(b[0]-c[0])*(a[1]-c[1])

    #找角点
    def find_corner(self,point_list):
        corner_num=len(point_list) #点数量
        ans=0.0 #最大面积
        ans_point_index_list=[0,0,0,0] #最大面积点对应索引
        m1_point=0
        m2_point=0
        for i in range(corner_num):
            for j in range(corner_num):#每次选两个点出来
                if (i==j):
                    continue
                m1=0.0 #存左侧点
                m2=0.0 #存右侧点

                for k in range(corner_num):
                    if (k==i or k==j):
                        continue
                    a=point_list[i][1]-point_list[j][1]
                    b=point_list[j][0]-point_list[i][0]
                    c=point_list[i][0]*point_list[j][1]-point_list[j][0]*point_list[i][1]
                    temp=a*point_list[k][0]+b*point_list[k][1]+c #有向面积，判断这个点在ij的左边还是右边

                    if (temp>0):#k在ij左边
                        tmp_area=abs(self.area(point_list[i],point_list[j],point_list[k])/2)
                        if tmp_area>m1:
                            m1=tmp_area
                            m1_point=k

                    elif (temp<0):#k在ij右边
                        tmp_area=abs(self.area(point_list[i],point_list[j],point_list[k])/2)
                        if tmp_area>m2:
                            m2=tmp_area
                            m2_point=k

                if (m1==0.0 or m2==0.0):
                    continue
                if (m1+m2>ans):
                    ans_point_index_list[0]=i
                    ans_point_index_list[1]=j
                    ans_point_index_list[2]=m1_point
                    ans_point_index_list[3]=m2_point
                    ans=m1+m2
        ans_point_list=[]
        for i in ans_point_index_list:
            ans_point_list.append(point_list[i])
        return ans_point_list

#给出颜色检测结果
class ColorCylinderDetecter(object): #写于15*25的工作台
    def __init__(self):
        self.average_circles=[] #当前更新圆环
        self.histroy_circles=[] #上一次圆环
        self.frame_count=0 #计数器
        self.update_frame=10 #更新帧数

        self.histroy_color=[] #上一次颜色检测结果
        self.norm_color=[] #整理过的颜色检测结果
        self.world_color=[] #世界坐标系的颜色检测结果
        self.color_detect=False #颜色检测默认不可更新

        self.calibration_points=[] #标定点 [[1,2],[2,3],[3,4],[4,5]]

        #self.last_time=time.time() #测试用

        self.std_colors = {
            'red': [(42,50,176),(2,5,138),(0,5,137),(44,45,174),(39,51,209)],
            'yellow': [(122,193,234),(112,183,220),(118,188,221),(104,177,246),(100,176,200),(73,152,189),(54,132,170),(29,104,140)],
            'light_blue': [(173,152,77),(162,145,70),(190,172,95),(182,165,90)],
            'dark_blue': [(140,87,20),(128,70,7),(141,93,25),(94,65,47),(140,93,27),(162, 113, 53)],
            'green': [(131,152,109),(102,132,75),(99,134,77),(81,124,57),(116,145,94),(106,138,80),(40,92,29),(140,173,118),(136,164,116),(147,178,123),(57, 106, 40)],
            'gray': [(100, 100, 100),(150,150,150),(120,120,120),(110,110,110),(130, 130, 130),(140, 140, 140),(0,0,0)]
        } #不同光照下的颜色,BGR的

        self.display_colors = {
            'red': (0,0,255),
            'yellow': (0,255,255),
            'green': (0,255,0),
            'dark_blue': (255,0,0),
            'light_blue': (255,255,0),
            'gray': (255,255,255) 
        } #颜色标注,BGR的

    def update(self,calibration_list):
        self.calibration_points=calibration_list
    
    #检测圆环
    def detect_circle_perframe(self,img):
        #source_img=img.copy() #测试用

        #对比度增强(不然边缘不明显)
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        l=clahe.apply(l)
        lab=cv2.merge([l,a,b])
        img_enhanced=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
        #cv2.imshow("enhanced",img_enhanced) #测试用
        
        canny=cv2.Canny(img_enhanced,40,150) #Canny边缘检测 40比50好
        #cv2.imshow("canny",canny) #测试用

        k=np.ones((3,3),np.uint8) #结构元，全1
        canny=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,k) #形态学闭操作
        #cv2.imshow("closed_canny",canny) #测试用

        circles=cv2.HoughCircles(
            canny,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=35,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=50
        ) #霍夫变换检测圆形
        if circles is None:
            return [] #返回列表，不然没法更新
        
        circles=np.round(circles[0,:]).astype("int") #提取的圆环
        # for (x,y,r) in circles: #测试用
        #     cv2.circle(source_img,(x,y),r,(0,255,0),2) #测试用
        #     cv2.circle(source_img,(x,y),2,(255,0,0),3) #测试用
        # cv2.imshow('circles',source_img) #测试用

        filtered_circles=self.filter_circles(canny,circles) #过滤圆环
        # for (x,y,r,overlap_ratio) in filtered_circles: #测试用
        #     cv2.circle(source_img,(x,y),r,(0,0,255),2) #测试用
        #     cv2.circle(source_img,(x,y),2,(255,0,0),3) #测试用
        #     cv2.putText(source_img,str(overlap_ratio),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,1) #文字标签
        # cv2.imshow('filtered_circles',source_img) #测试用

        return filtered_circles
    
    #筛选霍夫变换得到的圆
    def filter_circles(self,canny_edges,circles,threshold_ratio=0.75):
        filtered_circles=[]

        for (x,y,r) in circles:
            mask=np.zeros_like(canny_edges) #创建空白图片，用于绘制霍夫变换得到的圆
            cv2.circle(mask,(x,y),r,255,3) #绘制圆环

            total_points=cv2.countNonZero(mask) #计算完整圆环点数

            overlap_mask=cv2.bitwise_and(canny_edges,mask) #与操作得到重叠
            k=np.ones((5,5),np.uint8) #结构元，全1
            overlap_mask=cv2.morphologyEx(overlap_mask,cv2.MORPH_DILATE,k) #形态学膨胀
            overlap2_mask=cv2.bitwise_and(overlap_mask,mask) #二次重叠

            overlap_points=cv2.countNonZero(overlap2_mask) #计算重合边缘点个数

            overlap_ratio=overlap_points/total_points #可信度

            if overlap_ratio>threshold_ratio:
                filtered_circles.append((x,y,r,overlap_ratio))
        return filtered_circles
    
    #更新一次圆环结果
    def update_circle(self,circles,distance_threshold=40):
        #frame_count帧更新一次
        for (x,y,r,ratio) in circles:
            if_match=False #标记是否在已有类别
            #遍历累积结果，近的归为一类
            for i,(total_x,total_y,total_r,total_ratio) in enumerate(self.average_circles): #遍历当前更新圆环序列
                if np.linalg.norm((total_x/total_ratio-x,total_y/total_ratio-y))<distance_threshold:
                    new_total_x=total_x+ratio*x
                    new_total_y=total_y+ratio*y
                    new_total_r=total_r+ratio*r
                    new_total_ratio=total_ratio+ratio
                    self.average_circles[i]=(new_total_x,new_total_y,new_total_r,new_total_ratio)
                    if_match=True
                    break
            if not if_match:
                self.average_circles.append((x*ratio,y*ratio,r*ratio,ratio))
        self.frame_count+=1 #多计一次

        if self.frame_count==self.update_frame: #计数器满了
            if self.average_circles: #如果检测到结果就更新给history
                self.average_circles=np.array(self.average_circles)
                self.average_circles[:,:3]/=self.average_circles[:,3][:,np.newaxis] #结果取平均

                self.histroy_circles=self.average_circles[:,:3].astype(int).tolist() #把本次结果记录下来，history一直只有3列
                self.average_circles=[] #本次更新重置
            else:
                self.histroy_circles=[] #没结果也要把history更新好，不然上次结果会一直留着
                self.average_circles=[] #本次更新重置
            self.frame_count=0 #重新计数
            self.color_detect=True #可进行颜色检测

    #颜色检测,包含标注
    def detect_color(self,img):
        if self.color_detect: #颜色检测可更新
            self.update_color(img) #更新检测结果到类成员
            annoted_img=self.visualize_results(img) #进行可视化标注
            return annoted_img,True
        else:
            return 0,False
    
    #更新颜色检测结果
    def update_color(self,img,min_distance_threshold=5):
        #更新history
        result=[] #存储检测结果
        for x,y,r in self.histroy_circles: #如果有圆环才开始检测,没有直接返回空
            mask=np.zeros(img.shape[:2],dtype=np.uint8) 
            cv2.circle(mask,(x,y),r,255,-1) #圆形掩膜

            mean_color=tuple(map(int,cv2.mean(img,mask=mask)[:3])) #计算平均gbr颜色

            min_distance=float('inf') #初始最大距离
            closest_color=None #最近的颜色

            for color_name,std_color in self.std_colors.items():
                distance=np.linalg.norm(np.array(std_color)-np.array(mean_color),axis=1)
                #if np.any(distance<min_distance_threshold): #如果有一种颜色特别接近，就直接取
                #    closest_color=color_name  #直接取这种颜色
                #    min_distance=min(distance)
                #    continue
                distance=np.mean(distance)
                if distance<min_distance:
                    min_distance=distance #更新最小距离
                    closest_color=color_name #更新最小距离对应颜色
            
            if closest_color=='gray': #是阴影就不添加了
                continue

            #result.append((x,y,r,closest_color,mean_color)) #测试用
            result.append((x,y,r,closest_color))
        self.histroy_color=result

        #更新norm
        self.norm_color=[] 
        height,width,channels=img.shape #获取图片长宽
        for x,y,r,color_name in self.histroy_color:
            x_norm=x/width #点的归一化位置,相对于左下角
            y_norm=(height-y)/height
            r_norm=r/height #半径相对于高度的归一化长度
            self.norm_color.append((x_norm,y_norm,r_norm,color_name,width,height))
        
        #更新世界坐标系
        self.world_color=[]
        if not len(self.calibration_points)==0:
            picture_list=[[0,0],[width-1,0],[width-1,height-1],[0,height-1]] #图片列表 按照从左上角到右上角到右下角到左下角的顺时针
            perspective_matrix = cv2.getPerspectiveTransform(np.array(picture_list, dtype=np.float32), np.array(self.calibration_points, dtype=np.float32)) #计算映射关系
            for x,y,r,color_name in self.histroy_color:
                pic_point=np.array([x,y], dtype=np.float32).reshape((-1, 1, 2))
                world_point=cv2.perspectiveTransform(pic_point, perspective_matrix)
                self.world_color.append((x,y,r,world_point[0][0][0],world_point[0][0][1],color_name))

        self.color_update=False #重置颜色更新

    #检测，返回标注图片
    def detect(self,img):
        source_img=img.copy()
        circles=self.detect_circle_perframe(img) #检测一次圆环
        self.update_circle(circles) #更新一次圆环
        annoted_img,if_detect=self.detect_color(img) #进行颜色检测

        #for x,y,r,color_name,gbr in self.histroy_color: #测试用
        #for x,y,r,color_name in self.histroy_color: #测试用
        #    cv2.circle(source_img,(x,y),r,(0,255,0),2) #测试用
        #    cv2.circle(source_img,(x,y),2,(255,0,0),3) #测试用
        #    #text=f"{color_name} ({x},{y})" #文字标签
        #    text=f"{color_name} ({x},{y},{gbr})" #文字标签
        #    cv2.putText(source_img,text,(x-r,y-r),cv2.FONT_HERSHEY_SIMPLEX,0.5,self.display_colors[color_name],2) #文字标签
        #    #cv2.putText(source_img,text,(x-r,y-r),cv2.FONT_HERSHEY_SIMPLEX,0.5,gbr,2) #测试用,字体颜色是平均颜色
        #cv2.imshow('detect_img',source_img) #测试用
        if len(self.calibration_points)==0:
            return annoted_img,self.norm_color,if_detect
        else:
            return annoted_img,self.world_color,if_detect

    #标注结果
    def visualize_results(self,img):
        if len(self.calibration_points)==0:
            for x,y,r,color_name in self.histroy_color:
                cv2.circle(img,(x,y),r,self.display_colors[color_name],2) #绘制中心点
                cv2.circle(img,(x,y),2,(255,255,255),3) #测试用
                text=f"{color_name} ({x},{y})" #文字标签
                cv2.putText(img,text,(x-r,y-r),cv2.FONT_HERSHEY_SIMPLEX,0.5,self.display_colors[color_name],2) #文字标签
            return img #已标注图片
        else: #用实际位置标注
            for x,y,r,world_x,world_y,color_name in self.world_color:
                cv2.circle(img,(x,y),r,self.display_colors[color_name],2) #绘制中心点
                cv2.circle(img,(x,y),2,(255,255,255),3) #测试用
                text=f"{color_name} ({world_x:.2f},{world_y:.2f})" #文字标签
                cv2.putText(img,text,(x-r,y-r),cv2.FONT_HERSHEY_SIMPLEX,0.5,self.display_colors[color_name],2) #文字标签
            return img #已标注图片
    
if __name__=="__main__":
    #测试
    img_focus=FocusFinder()
    img_detecter=ColorCylinderDetecter()
    capture=cv2.VideoCapture(1) #尝试打开摄像头
    #img_detecter.update([[221.176,130.626],[193.288,-106.53800],[78.409,-84.7644],[80.5877,122.2352]])
    if not capture.isOpened(): # 如果摄像头无法打开
        print("Error:Could not open camera.")
        exit()

    while True:
        ret,frame=capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow('ori_img',frame) #测试用
        focus_img,if_find=img_focus.find_focus(frame) #注意力图片
        if if_find:
            cv2.imshow('focus_img',focus_img) #测试用
            annoted_img,norm_color_result,if_detect=img_detecter.detect(focus_img) #检测结果
            if if_detect:
                cv2.imshow('annoted_img',annoted_img) #测试用

        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()