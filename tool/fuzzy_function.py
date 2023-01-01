import numpy as np
import copy
#1つのファジィ制御のクラス(設計書)
class fuzzy:
    #ファジィルール，K群，集合の数(前件，後件)，メンバーシップ関数のタイプ(前件，後件)
    def __init__(self,rule,Ki,set_nums,set_types):
        self.rule = rule
        self.Ki,self.Kw = self.update_Klist(Ki,set_nums,set_types)
        self.set_nums = set_nums
        self.set_types = set_types
        self.si_set = self.slope_intercept_set(self.Ki,self.Kw,set_types)
        self.appearanceList =[]
        
    def making_widths_step(self,one_Ki,set_type):
        one_Kw = []
        sum_k = 0
        one_Kw.append(sum_k)
        if set_type==0:
            for k in one_Ki:
                sum_k += k
                one_Kw.append(sum_k)
        elif set_type==1 or set_type==-1:
            for k in one_Ki:
                sum_k += k
                one_Kw.append(sum_k)
        return one_Kw
    
    def update_Klist(self,Kis,set_nums,set_types):
        new_Kis = []
        Kis_widths = []
        for i in range(len(Kis)):
            one_Kw = []
            set_num = set_nums[i]
            member_type = set_types[i]
            if member_type == 0:
                side_num = set_num//2
                if side_num != len(Kis[i]):
                    one_Ki = [Kis[i][0]]*side_num
                    one_Kw = self.making_widths_step(one_Ki,0)
                else:
                    one_Ki = Kis[i]
                    one_Kw = self.making_widths_step(one_Ki,0)
            elif member_type == 1:
                if set_num != len(Kis[i])+1:
                    one_Ki = [Kis[i][0]]*(set_num-1)
                    one_Kw = self.making_widths_step(one_Ki,1)
                else:
                    one_Ki = Kis[i]
                    one_Kw = self.making_widths_step(one_Ki,1)
            elif member_type == -1:
                if set_num != len(Kis[i])+1:
                    one_Ki = [Kis[i][0]]*(set_num-1)
                    one_Kw = self.making_widths_step(one_Ki,-1)
                else:
                    one_Ki = Kis[i]
                    one_Kw = self.making_widths_step(one_Ki,-1)
            new_Kis.append(one_Ki)
            Kis_widths.append(one_Kw)
        return new_Kis,Kis_widths
    
    def slope_intercept_set(self,Ki,Kw,set_types):
        sl_list = []
        #Kw = copy.deepcopy(Kw)
        #右側
        for k,w,t in zip(Ki,Kw,set_types):
            num_set = len(k)
            k_arr = np.array(k)
            for i in range(num_set):
                w.insert(0,-w[-(num_set-i)])
            w_arr = np.array(w)
            sr_arr=1/k_arr
            sl_arr=-1*sr_arr
            ir_arr=np.flip(w_arr[1:num_set+1]).copy()*sr_arr
            il_arr=-w_arr[num_set+1:]*sl_arr
            r_arr = np.stack([sr_arr, ir_arr],1)
            l_arr = np.stack([sl_arr, il_arr],1)
            union_arr_right = np.stack([r_arr,l_arr],1)
            if t==1:
                sl_list.append(union_arr_right)
            else:
                union_arr_left = np.flip(union_arr_right,(0,1)).copy()
                union_arr_left[:,:,0] = -1*union_arr_left[:,:,0]
                if t==-1:
                    sl_list.append(union_arr_left)
                else:
                    union_all = np.vstack([union_arr_left,union_arr_right])
                    sl_list.append(union_all)
        return sl_list
    
    def fuzzy_set(self,i,value,si):
        grade = si[0] * value + si[1]
        result = [i,grade]
        return result
    
    def fuzzy_ante(self,x,k,Kw,set_num,set_type,si_set):
        start_no = 0
        end_no = set_num - 1
        if set_type == 1:
            start_no = set_num-1
            end_no = start_no*2
        value_sets = []
        if x <= Kw[start_no]:
            value_sets.append([0,1.0])
        elif x >= Kw[end_no]:
            value_sets.append([end_no-start_no,1.0])
        for i in range(start_no,end_no):
            if Kw[i] < x and x < Kw[i+1]:
                value_sets.append(self.fuzzy_set(i-start_no, x, si_set[i-start_no][1]))
                value_sets.append(self.fuzzy_set(i-start_no + 1, x,si_set[i-start_no][0]))
                break
            elif x == Kw[i]:
                value_sets.append([i-start_no,1.0])
                break
            elif x == Kw[i+1]:
                value_sets.append([i-start_no+1,1.0])
                break
        return  value_sets
    
    def fuzzy_min_case(self,antes ,x_num):
        outs=[]
        if x_num == 2:
            for j in range(len(antes[0])):
                for i in range(len(antes[1])):
                    self.appearanceList.append([antes[0][j][0],antes[1][i][0]])
                    outs.append(self.fuzzy_min2(antes[0][j], antes[1][i]))
        
        elif x_num == 3:
            for k in range(len(antes[0])):
                for j in range(len(antes[1])):
                    for i in range(len(antes[2])):
                        self.appearanceList.append([antes[0][k][0],antes[1][j][0],antes[2][i][0]])
                        outs.append(self.fuzzy_min3(antes[0][k], antes[1][j], antes[2][i]))
        return outs
    
    def fuzzy_min2(self,ant1,ant2):
        out_Set = self.rule[ant2[0]][ant1[0]]
        if ant1[1] >= ant2[1]:
            out_Value = ant2[1]
        elif ant1[1] < ant2[1]:
            out_Value = ant1[1]
        out_set = [out_Set,out_Value]
        return out_set
    
    def fuzzy_min3(self,ant3,ant2,ant1):
        out_Set = self.rule[ant3[0]][ant2[0]][ant1[0]]
        if (ant1[1] <= ant2[1] and ant1[1] <= ant3[1]):
            out_Value = ant1[1]
        elif ant2[1] <= ant1[1] and ant2[1] <= ant3[1]:
            out_Value = ant2[1]
        elif ant3[1] <= ant1[1] and ant3[1] <= ant2[1]:
            out_Value = ant3[1]
        out_set = [out_Set,out_Value]
        return out_set
    
    def fuzzy_cons1(self,outs):
        set_num = self.set_nums[-1]
        set_type = self.set_types[-1]
        start_no = 0
        end_no = set_num
        if set_type == 1:
            start_no = set_num-1
            end_no = start_no*2+1
        out = [0] * set_num
        Kw_last = copy.deepcopy(self.Kw[-1])
        for i in range(len(outs)):
            if out[outs[i][0]] != 0:
                if outs[i][1] > out[outs[i][0]]:
                    out[outs[i][0]] = outs[i][1]
            else:
                out[outs[i][0]] = outs[i][1]            
        deno = 0
        nume = 0
        for i in range(start_no,end_no):
            deno += out[i-start_no]
            nume += out[i-start_no] * Kw_last[i]
        output = nume / deno
        return output
    
    def fuzzy_calc(self,x):
        antes = []
        outs = []
        for i in range(len(x)):
            antes.append(self.fuzzy_ante(x[i], self.Ki[i],self.Kw[i], self.set_nums[i],self.set_types[i],self.si_set[i]))
        outs = self.fuzzy_min_case(antes,len(x))
        output_speed = self.fuzzy_cons1(outs)
        return output_speed