import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import copy
import math


def drive_test(problem,Kis,fpos,lms,tim,goal_no):
    Obj = problem.obj_control
    Obj.time_limit = tim
    Obj.fpos = fpos
    Obj.pose = fpos
    Obj.fuzzy_produce(Kis,lms)
    Obj.move_trial(goal_no)

    lmxy_max = np.max(lms,axis=0)
    lmxy_min = np.min(lms,axis=0)
    if fpos[0] > lmxy_max[0]:
        lmxy_max[0] = fpos[0]
    elif fpos[0] < lmxy_min[0]:
        lmxy_min[0] = fpos[0] 
    if fpos[1] > lmxy_max[1]:
        lmxy_max[1] = fpos[1]
    elif fpos[1] < lmxy_min[1]:
        lmxy_min[1] = fpos[1] 

    fig=plt.figure(figsize=(4,4))
    ax=fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_xlim(lmxy_min[0]-0.2,lmxy_max[0]+0.2)
    ax.set_ylim(lmxy_min[1]-0.2,lmxy_max[1]+0.2)
    ax.set_xlabel("X",fontsize=10)
    ax.set_ylabel("Y",fontsize=10)
    ax.scatter(fpos[0],fpos[1],s=50,marker="o",label="landmarks",color="cyan")

    count  = 0
    for lm in lms: #静止体の描画1
        ax.scatter(lm[0],lm[1],s=50,marker="*",label="landmarks",color="orange")
        ax.text(lm[0],lm[1],"id:"+str(count),fontsize=10)
        count +=1
    
    plt.scatter(Obj.li1,Obj.li2,s=1,marker="o")
    plt.show()
    return [Obj.li1,Obj.li2,Obj.li3]

def print_obj_func(problem,res,order):
    if order==0:
        pop=res.pop
        obj_values = pop.get("F")
    else:
        obj_values = res.F
    #グラフ表示(2次元)
    plt.rcParams["figure.figsize"] = (10,3)
    plt.subplot(131)
    for a,b in zip(obj_values[:,0],obj_values[:,1]):
        plt.scatter(a,b)
    plt.subplot(132)
    for a,b in zip(obj_values[:,0],obj_values[:,2]):
        plt.scatter(a,b)
    plt.subplot(133)
    for a,b in zip(obj_values[:,1],obj_values[:,2]):
        plt.scatter(a,b)
    plt.show()
    
def member_func_print(fuzzy_obj):
    num = len(fuzzy_obj.si_set)
    y = np.array([0,1])
    for i in range(num):
        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(1, 1, 1)
        x = []
        x_1=None
        abset = fuzzy_obj.si_set[i].reshape(-1,2)
        for j in range(len(abset)):
            x_tmp = (y-abset[j][1])/abset[j][0]
            x.append(x_tmp)
        if i!=num-1:
            for one_x in x:
                ax.plot(one_x, y,color ="black")
                one_x[0]=round(one_x[0], 2)
                if one_x[0]!=x_1:
                    ax.annotate("{:.2f}".format(one_x[0]), (one_x[0], 0), size = 10, textcoords='offset points', xytext=(0,-20), ha='center', va='bottom')
                    x_1 = one_x[0]
        else:
            for one_x in x:
                ax.axvline(x=one_x[0], ymin=0, ymax=1,color="black")
                one_x[0]=round(one_x[0], 2)
                if one_x[0]!=x_1:
                    ax.annotate("{:.2f}".format(one_x[0]), (one_x[0], 0), size = 10, textcoords='offset points', xytext=(0,-20), ha='center', va='bottom')
                    x_1 = one_x[0]
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ty = fuzzy_obj.set_types[i]
        if ty==0:
            x_max = x_1
            x_min = -x_max
        elif ty==1:
            x_max = x_1
            x_min = 0
        else:
            x_min = ((y-abset[0][1])/abset[0][0])[0]
            x_max = 0
        ax.vlines(0, 0, 1,color="gray",linestyles='dotted')
        ax.hlines(y=[0, 1], xmin=x_min, xmax=x_max,color="gray",linestyles='dotted')
        
        
def print_membership(Pop,Kis,obj_control):
    lms = None
    member_confirm_obj = obj_control
    member_confirm_obj.fuzzy_produce(Kis,lms)
    member_func_print(member_confirm_obj.fuzzy1)
    member_func_print(member_confirm_obj.fuzzy2)

def data_to_text(fuzzy_class):
    rule = fuzzy_class.rule
    rule_arr  = copy.deepcopy(rule)
    text_data = rule_arr.tolist()
    set_types = fuzzy_class.set_types
    set_nums = fuzzy_class.set_nums
    set_set = []
    axis_set = []
    for i in range(len(set_nums)):
        axis=[]
        axis.append("ZO")
        set_type = set_types[i]
        if set_type==0:
            num = set_nums[i]//2
            for j in range(num):
                axis.append("P"+str(j))
                axis.insert(0,"N"+str(j))
        elif set_type==1:
            num = set_nums[i]-1
            for j in range(num):
                axis.append("P"+str(j))
        elif set_type==-1:
            num = set_nums[i]-1
            for j in range(num):
                axis.insert(0,"N"+str(j))
        axis_set.append(axis)
        
    set_type = set_types[-1]
    
    if len(set_nums)-1==2:
        for k in range(len(text_data)):
            for j in range(len(text_data[0])):
                value = text_data[k][j] 
                text_data[k][j] = axis_set[-1][value]
    elif len(set_nums)-1==3:
        for k in range(len(text_data)):
            for j in range(len(text_data[0])):
                for i in range(len(text_data[0][0])):
                    value = text_data[k][j][i] 
                    text_data[k][j][i] = axis_set[-1][value]
        
    return axis_set,text_data,rule_arr

def print_rule_text(fuzzy_class):
    axis_set,text_data,data = data_to_text(fuzzy_class)
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    cm = plt.get_cmap('coolwarm')
    color = cm(norm_data)
    if len(fuzzy_class.set_nums)-1==2:
        rule_2_text(text_data,axis_set,color)
    elif len(fuzzy_class.set_nums)-1==3:
        rule_3_text(text_data,axis_set,color)
    return True

def rule_2_text(text_data,axis_set,color):
    # fig準備
    fig = plt.figure(figsize=(8,1))
    ax1 = fig.add_subplot(111)

    # 表の定義
    ax1.axis('off')
    the_table = ax1.table(cellText=text_data,cellLoc="center",colWidths=[0.05]*8, colLabels=axis_set[0], rowLabels=axis_set[1],
          loc="center",cellColours=color)
    the_table.scale(2,2)
    the_table.set_fontsize(14)
    # 表示
    plt.show()
    plt.close()
    return True
    
def rule_3_text(text_data,axis_set,color):
    # fig準備
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
 
    # 表の定義
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    the_table1 = ax1.table(cellText=text_data[0],cellLoc="center",colWidths=[0.05]*8, colLabels=axis_set[1], rowLabels=axis_set[2],
          loc="center",cellColours=color[0])
    the_table2 = ax2.table(cellText=text_data[1],cellLoc="center",colWidths=[0.05]*8, colLabels=axis_set[1], rowLabels=axis_set[2],
          loc="center",cellColours=color[1])
    the_table3 = ax3.table(cellText=text_data[2],cellLoc="center",colWidths=[0.05]*8, colLabels=axis_set[1], rowLabels=axis_set[2],
          loc="center",cellColours=color[2])

    the_table1.scale(4,3)
    the_table1.set_fontsize(14)
    the_table2.scale(4,3)
    the_table2.set_fontsize(14)
    the_table3.scale(4,3)
    the_table3.set_fontsize(14)
 
    # 表示
    print(axis_set[0])
    plt.show()
    plt.close()

def print_result(member_result,rule_result,obj_control,Pop,Kis):
    if rule_result:

        print_rule_text(obj_control.fuzzy1)
        print_rule_text(obj_control.fuzzy2)
    if member_result:
        print_membership(Pop,Kis,obj_control)

def drive_test_orbit(obj_control):
    Xlist = obj_control.li1
    Ylist = obj_control.li2
    Tlist = obj_control.li3
    fig, ax = plt.subplots(figsize=(5, 5))
    for x,y,t in zip(Xlist,Ylist,Tlist):
        patch = patches.RegularPolygon(
            xy=(x, y), numVertices=3, radius=0.1, orientation=t-math.pi/2,color="white",ec="black",alpha=0.5)
        ax.add_patch(patch)
    ax.scatter(obj_control.lms[:,0],obj_control.lms[:,1], c='red', ec="red",label='orbit_point',s=10)
    ax.plot(obj_control.lms[:,0],obj_control.lms[:,1] ,color="red", label="orbit",linestyle="dashed",linewidth=0.5)
    ax.autoscale()
    ax.legend(loc='upper left')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1, 1)
    ax.autoscale()
    plt.show()

def save_drive_orbit(obj_control,Kis,fpos,lms,tim,display,goal_no):
    Obj = obj_control
    Obj.time_limit = tim
    Obj.fpos = fpos
    Obj.pose = fpos
    Obj.fuzzy_produce(Kis,lms)
    Obj.move_trial(goal_no)
    Xlist = Obj.li1
    Ylist = Obj.li2
    Tlist = Obj.li3
    lmxy_max = np.max(lms,axis=0)
    lmxy_min = np.min(lms,axis=0)
    if fpos[0] > lmxy_max[0]:
        lmxy_max[0] = fpos[0]
    elif fpos[0] < lmxy_min[0]:
        lmxy_min[0] = fpos[0] 
    if fpos[1] > lmxy_max[1]:
        lmxy_max[1] = fpos[1]
    elif fpos[1] < lmxy_min[1]:
        lmxy_min[1] = fpos[1] 

    if display:
        fig=plt.figure(figsize=(4,4))
        ax=fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim(lmxy_min[0]-0.2,lmxy_max[0]+0.2)
        ax.set_ylim(lmxy_min[1]-0.2,lmxy_max[1]+0.2)
        ax.set_xlabel("X",fontsize=10)
        ax.set_ylabel("Y",fontsize=10)
        for i in range(0,len(Xlist),5):
            patch = patches.RegularPolygon(xy=(Xlist[i], Ylist[i]), numVertices=3, radius=0.1, orientation=Tlist[i]-math.pi/2,color="white",ec="black",alpha=0.5)
            ax.add_patch(patch)
        ax.scatter(Obj.lms[:,0],Obj.lms[:,1], c='red', ec="red",label='orbit_point',s=10)
        ax.plot(Obj.lms[:,0],Obj.lms[:,1] ,color="red", label="orbit",linestyle="dashed",linewidth=0.5)
        ax.legend(loc='upper left')
        plt.show()

    return [Xlist,Ylist,Tlist]

def display_oribit(Xlist,Ylist,Tlist,lms,figure_size):
    fpos = [Xlist[0],Ylist[0],Tlist[0]]
    lmxy_max = np.max(lms,axis=0)
    lmxy_min = np.min(lms,axis=0)
    if fpos[0] > lmxy_max[0]:
        lmxy_max[0] = fpos[0]
    elif fpos[0] < lmxy_min[0]:
        lmxy_min[0] = fpos[0] 
    if fpos[1] > lmxy_max[1]:
        lmxy_max[1] = fpos[1]
    elif fpos[1] < lmxy_min[1]:
        lmxy_min[1] = fpos[1] 
    fig=plt.figure(figsize=figure_size)
    ax=fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_xlim(lmxy_min[0]-0.2,lmxy_max[0]+0.2)
    ax.set_ylim(lmxy_min[1]-0.2,lmxy_max[1]+0.2)
    ax.set_xlabel("X",fontsize=10)
    ax.set_ylabel("Y",fontsize=10)
    for i in range(0,len(Xlist),5):
        patch = patches.RegularPolygon(xy=(Xlist[i], Ylist[i]), numVertices=3, radius=0.1, orientation=Tlist[i]-math.pi/2,color="white",ec="black",alpha=0.5)
        ax.add_patch(patch)
    ax.scatter(lms[:,0],lms[:,1], c='red', ec="red",label='orbit_point',s=10)
    ax.plot(lms[:,0],lms[:,1] ,color="red", label="orbit",linestyle="dashed",linewidth=0.5)
    ax.legend(loc='upper left')
    plt.show()


