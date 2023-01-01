import numpy as np
import math
from collections import deque
import tool.fuzzy_function as ff
import tool.others_function as of


#軌道追従制御のクラス(設計書)
class obj_func:
    def __init__(self,time_limit=20,time_interval=0.5,time_detail=0.1,
                 fpos=[0,0,math.pi/2],init_vo=[360,0],footsteps=5,
                 fuzzy_rule1=None,set_types1=None,out_level1=None,fuzzy_rule2=None,set_types2=None,out_level2=None,
                 lms=None,Kis=None,object_items=None):
        self.time_limit=time_limit #行う時間
        self.time_interval=time_interval #制御周期
        self.time_detail = time_detail #1コマの時間
        self.fpos=fpos
        self.pose = fpos
        self.fpos_list = None
        self.init_vo = init_vo
        self.nu=init_vo[0]
        ##############ロボットの大きさ変更########################################
        self.wheel_d = 0.0665
        self.robot_width = 0.117 #左右輪の中心の間の距離(m)
        ######################################################
        self.nu_m = init_vo[0]/360*(self.wheel_d*math.pi)
        self.nur = init_vo[0]  
        self.nul = init_vo[0]
        self.omega=init_vo[1]
        self.lms = None
        self.sum_time = 0.0
        
        #記憶できる足跡数と連番初期化
        self.orbit_que = deque([np.array(fpos)for i in range(footsteps)])
        self.past_no = 0
        self.orbit_no = 0
        self.now_no = 1 #現在の目標点
        self.next_no = 2 #次の目標点
        self.no_len = 3 #目標点の数
        self.point_r = 0.1 #目標点を判定する半径の大きさ
        
        #ファジィルール設定
        self.fuzzy_rule1 = fuzzy_rule1
        self.length_ff1 = len(self.fuzzy_rule1)#theta2
        self.length_ff2 = len(self.fuzzy_rule1[0])# omega
        self.length_ff3 = len(self.fuzzy_rule1[0][0]) # distance
        self.set_nums1 = [self.length_ff1,self.length_ff2,self.length_ff3,out_level1]
        self.set_types1 = set_types1
        
        self.fuzzy_rule2 = fuzzy_rule2
        self.length_ff4 = len(self.fuzzy_rule2)
        self.length_ff5 = len(self.fuzzy_rule2[0])
        self.set_nums2 = [self.length_ff4,self.length_ff5,out_level2]
        self.set_types2 = set_types2
        
        #fuzzy初期化
        self.Ki1 = None
        self.Ki2 = None
        self.fuzzy1 = None
        self.fuzzy2 = None 
        
        #その他+記録用
        self.move_end = 0
        self.sum_no = 0 #目標点を通過した数
        self.li1 = []
        self.li2 = []
        self.li3 = []
        self.now_nos = []
        self.lms3 = []
        self.object_items = object_items
        
        
    def KandLMset(self,Kis,lms):
        self.Ki1 = [Kis[0],Kis[1],Kis[2],Kis[3]]
        self.Ki2 = [Kis[4],Kis[5],Kis[6]]
        self.lms = lms
        return True
    
    def fuzzy_produce(self,Kis,lms):
        self.KandLMset(Kis,lms)
        self.fuzzy1 = ff.fuzzy(self.fuzzy_rule1,self.Ki1,self.set_nums1,self.set_types1)
        self.fuzzy2 = ff.fuzzy(self.fuzzy_rule2,self.Ki2,self.set_nums2,self.set_types2)
        return True
        
    def move(self):
        for i in range(int(self.time_limit/self.time_detail)+1):
            move_end = self.one_step(self.time_interval,self.time_detail)
            if move_end==True:
                return float(i/10)
        return float(self.time_limit)
    
    def one_step(self,time_interval,time_detail): #time_intervalは1ステップ何秒であるかを指定する
        #制御周期か否か判断
        time_judge = ((self.sum_time*10.0)/(time_interval*10.0)).is_integer()
        if time_judge:
            nu,omega=self.decision(self.pose)
            self.nu_m,self.omega = nu,omega #速度・角速度更新
        else:
            nu,omega= self.nu_m,self.omega
        self.sum_time = (self.sum_time*10+(10*time_detail))/10 #合計時間

        #pose記録
        self.li1.append(self.pose[0])
        self.li2.append(self.pose[1])
        self.li3.append(self.pose[2])
        self.now_nos.append(self.now_no)
        #time_detail秒後の自己位置計算
        self.pose=self.state_transition(self.nu_m,omega,self.time_detail,self.pose)
        if self.pose[2] >= 2*math.pi:
            self.pose[2]=self.pose[2] - 2*math.pi
        elif self.pose[2] < 0:
            self.pose[2] = self.pose[2] +2*math.pi
        #目標切り替えのための軌跡登録
        self.orbit_register(self.pose)
        return self.move_end
    
    def decision(self,pose):
        #目標軌道点の入れ替え
        for point in self.orbit_que:
            change_r = math.sqrt((point[0]-self.lms3[self.now_no][0])**2+(point[1]-self.lms3[self.now_no][1])**2)
            if change_r <= self.point_r: #(m)
                self.past_no+=1
                self.now_no+=1
                self.next_no+=1
                self.sum_no+=1
                if self.sum_no ==self.no_len:
                    self.move_end=True
                    #return 0,0
                if self.past_no == self.no_len:
                    self.past_no=0
                if self.now_no == self.no_len:
                    self.now_no = 0
                if self.next_no == self.no_len:
                    self.next_no = 0
                break #始めのif条件満たしたら、このfor文を抜ける
        
            #制御の入力から出力まで
        delta_degrees = int(self.fuzzy2.fuzzy_calc(self.Input2(pose)))
        nu_degrees = int(self.fuzzy1.fuzzy_calc(self.Input1(pose,delta_degrees)))
        nu,omega = self.calu(nu_degrees,delta_degrees)
        return nu,omega
    
    def Input1(self,pose,omega):
        #目標点と車両の距離を求める(mm)
        S = math.sqrt((self.lms3[self.now_no][0]-pose[0])**2+(self.lms3[self.now_no][1]-pose[1])**2)*1000 #(mm)
        #車両から見た現在の目標点と次の目標点の間の角度 
        a = math.sqrt((self.lms3[self.now_no][0]-pose[0])**2+(self.lms3[self.now_no][1]-pose[1])**2) 
        b = math.sqrt((self.lms3[self.next_no][0]-pose[0])**2+(self.lms3[self.next_no][1]-pose[1])**2)
        c = math.sqrt((self.lms3[self.now_no][0]-self.lms3[self.next_no][0])**2+(self.lms3[self.now_no][1]-self.lms3[self.next_no][1])**2)
        theta2 =math.acos((a**2+c**2-b**2)/(2*a*c+10e-10)) #(rad)
        input_x = np.array([theta2,omega,S])
        return input_x
    
    def Input2(self,pose):
        #新thetax and 新Sx
        x1 =  self.lms3[self.past_no][0]
        x2 = self.lms3[self.now_no][0]
        y1 = self.lms3[self.past_no][1]
        y2 = self.lms3[self.now_no][1]
        d_X = x2-x1 #xの相対距離
        d_Y = y2-y1 #yの相対距離
        
        alpha = math.atan2(d_Y,d_X)
        thetax = alpha - pose[2]
        if thetax<-math.pi:
            thetax +=2*math.pi
        if x1 == x2 :
            if y2>y1:
                Sx = (pose[0]-x1)*1000
            elif y1>y2:
                Sx = -(pose[0]-x1)*1000    
        elif y1 == y2:
            if x2>x1:
                Sx = -(pose[1]-y1)*1000
            elif x1>x2:
                Sx = (pose[1]-y1)*1000
        else:
            a = y2-y1
            b = (x1-x2)
            c = x2*y1-x1*y2
            Sx = (a*pose[0]+b*pose[1]+c)/math.sqrt(a**2+b**2)*1000 #mmに直す  
        input_x = np.array([thetax,Sx])
        return input_x 

    def calu(self,nu_degrees,delta_degrees):
        #右車輪、左車輪の速度変換を入力
        delta_vr = delta_degrees
        delta_vl = -delta_degrees
        self.nur = self.nu+delta_vr+nu_degrees  #右車輪の速度
        self.nul = self.nu+delta_vl+nu_degrees  #左車輪の速度
        v = (self.nur+self.nul)/720*self.wheel_d*math.pi
        omega = (self.nur-self.nul)*math.pi/360*self.wheel_d/self.robot_width
        return v,omega
    
    def state_transition(cls,nu,omega,time,pose): #Δt秒後のロボットの姿勢(x,y,θ)
        t0=pose[2]
        if math.fabs(omega)<1e-10:
            return pose+np.array([nu*math.cos(t0),nu*math.sin(t0),omega])*time
        else:
            return  pose+np.array([nu/omega*(math.sin(t0+omega*time)-math.sin(t0)),nu/omega*(-math.cos(t0+omega*time)+math.cos(t0)),
                                  omega*time])
    
    def orbit_register(self,pose): #軌跡のキュー
            self.orbit_que.popleft()
            self.orbit_que.append(pose)
            
    def data_process(self,fit_dataset):
        y_sum = 0
        dth_sum = 0
        fitt=0
        fity=0
        fitz=0
        for value in fit_dataset:
            t = value[0]
            for i in range(len(value[4])):
                x = value[1][i]
                y = value[2][i]
                th = value[3][i]
                nn = value[4][i]
                if nn==1:
                    sp = self.lms3[0][0:2]
                    ep = self.lms3[1][0:2]
                elif nn ==2:
                    sp = self.lms3[1][0:2]
                    ep = self.lms3[2][0:2]
                elif nn ==0:
                    sp = self.lms3[2][0:2]
                    ep = self.lms3[0][0:2]
                new_pos = of.pos_conversion([x,y,th],sp,ep)
                y_sum+=abs(new_pos[1])
            for j in range(len(value[3])-1):
                del_th = abs(value[3][j]-value[3][j+1])
                if del_th >= math.pi :
                    del_th = 2*math.pi-del_th
                dth_sum +=del_th
            
            fitt += t
           # print("FIT_T={}  T={}  lms.shape[0]={} ".format(fitt,t,self.lms.shape[0]))
            fity += y_sum/self.lms.shape[0]
            fitz += dth_sum/self.lms.shape[0]
            y_sum = 0
            dth_sum = 0
        #print("FIT_T={} ".format(fitt/self.lms.shape[0]))
        return [fitt/self.lms.shape[0],fity,fitz]
    
    def reset_init(self):
        self.move_end = 0
        self.sum_no = 0 #目標点を通過した数
        self.li1 = []
        self.li2 = []
        self.li3 = []
        self.now_nos = []
        self.orbit_que = deque([np.array(self.fpos)for i in range(5)])
        self.past_no = 0
        self.orbit_no = 0
        self.now_no = 1 #現在の目標点
        self.next_no = 2 #次の目標点
        self.pose = self.fpos
        self.nu=self.init_vo[0]
        self.nu_m = self.init_vo[0]/360*(0.0665*math.pi)
        self.nur = self.init_vo[0]  
        self.nul = self.init_vo[0]
        self.omega=self.init_vo[1]
        self.sum_time = 0.0
        
    def move_to_obj(self):
        fit_dataset = []
        self.no_len = 3
        for i in range(self.lms.shape[0]):
            self.reset_init()
            #self.lms3 = np.vstack([np.array(self.fpos[0:2]),self.lms[i]])
            self.fpos = self.fpos_list[i]
            self.pose = self.fpos_list[i] 
            self.lms3 =self.lms[i]
            fit_dataset.append([self.move(),self.li1,self.li2,self.li3,self.now_nos]) ##u =  [T,[list1,list2..]]
        Fits = self.data_process(fit_dataset)
        return Fits
    
    def move_trial(self):
        self.reset_init()
        self.no_len = len(self.lms)
        self.lms3 = self.lms
        self.move()
        return True