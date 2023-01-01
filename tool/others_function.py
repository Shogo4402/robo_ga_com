import numpy as np
import random
import math
import tool.robot_function as rof

#[0,0]と任意の2点の軌道(評価用)
def lm_pos(r_max):
    #r = np.array([random.uniform(0, r_max) for i in range(2)])
    #th = np.array([random.uniform(0,2*math.pi) for i in range(2)])
    r = np.array([random.uniform(0, r_max) for i in range(3)])
    th = np.array([random.uniform(0,2*math.pi) for i in range(3)])
    x = r*np.cos(th)
    y = r*np.sin(th)
    lm2 = np.stack([x, y],axis=1)
    return lm2

def lms_produce(r_max,num):
    lms = np.array([lm_pos(r_max) for i in range(int(num))])
    return lms

def circle_pos_produce(n,r):
    flm = [0,0]
    slm = [0,-1]
    lms = []
    for i in range(0,n):
        lm = []
        lm.append(flm)
        lm.append(slm)
        theta = i*math.pi/n
        x = r*math.cos(theta+math.pi/2)
        y = r*math.sin(theta+math.pi/2)
        lm.append([x,y])
        lms.append(lm)
    return np.array(lms)

def circle_fpos_produce(n,r):
    fposs = []
    for i in range(0,n):
        theta = i*math.pi/n
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        fposs.append(np.array([x,y,theta]))
    return fposs

def efficient_lms_produce():
    first = [0,-0.5]
    second = [0,-0.2]
    third = [0,0]
    lms_list = []
    r = 0.5
    for i in range(-5,5):
        theta = i*math.pi/10
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        x_last = 1*math.cos(theta)
        y_last = 1*math.sin(theta)
        lms_list.append([first,second,third,[x,y],[x_last,y_last]])
    lms = np.array(lms_list)
    return lms

def efficient_fpos_produce():
    fpos_list = []
    for i in range(10):
        fx = (i+1)*0.1
        fy = -0.5
        fpos_xyt = [fx,fy,0]
        fpos_list.append(fpos_xyt)
    return np.array(fpos_list)

def pos_conversion(pos,sp,ep):
    x,y,theta = pos
    ep = ep-sp
    delth = math.atan2(ep[1],ep[0])
    R = np.array([[np.cos(-delth), -np.sin(-delth)],
                  [np.sin(-delth), np.cos(-delth)]])
    v = (x-sp[0],y-sp[1])
    new_pos = np.dot(R,v)
    new_pos = np.append(new_pos, theta-delth)
    return new_pos

def pop_No_update(res,no):
    if no!=None:
        pop_no = np.argmin(res.F[:,no])
    else:
        pop_no=np.argmin(np.sum(res.F,1))
    print("pop_update -> <POP {}>".format(pop_no))
    return pop_no

def fuzzy_control_new_set(problem,pop_no,pop):
    if pop_no != None:
        genes = pop[pop_no].get("X")
        genes = genes.tolist()
        if (problem.ruch_count1 != 0) or (problem.ruch_count2!=0) == int:
            Kis = problem.Ks_divi_list(genes[0:-(problem.ruch_count1+problem.ruch_count2)],problem.divi_list)
            ruge_list = genes[len(genes[0:-(problem.ruch_count1+problem.ruch_count2)]):]
            problem.obj_control.fuzzy_rule1 = problem.rule_changer(problem.obj_control.fuzzy_rule1,problem.ruch_count1,
                                                        problem.ruch_countarr1,ruge_list[0:problem.ruch_count1])
            problem.obj_control.fuzzy_rule2 = problem.rule_changer(problem.obj_control.fuzzy_rule2,problem.ruch_count2,
                                                        problem.ruch_countarr2,ruge_list[problem.ruch_count1:problem.ruch_count1+problem.ruch_count2])
        else:
            Kis = problem.Ks_divi_list(genes,problem.divi_list)         
    
    else:
        Kis = [[0.781],[308],[147],[106],[0.715],[188],[187]]
        out_level1 = 3
        fuzzy_rule1 = np.array([[[0,1,2],[0,1,1],[0,0,0]],[[1,2,2],[1,1,1],[0,0,0]],[[2,2,2],[1,1,1],[0,0,0]]])
        set_types1 = [1,1,1,-1]
        out_level2 =7
        fuzzy_rule2 = np.array([[0,0,0,1,2,3,4],[0,0,1,1,2,3,4],[0,1,1,2,3,4,5],[0,1,2,3,4,5,6],[1,2,3,4,5,5,6],[2,3,4,5,5,6,6],[2,3,4,5,6,6,6]])
        set_types2 = [0,0,0]
        problem.obj_control = rof.obj_func(fuzzy_rule1 = fuzzy_rule1,set_types1 = set_types1, out_level1=out_level1,
                                 fuzzy_rule2 = fuzzy_rule2,set_types2 = set_types2,out_level2=out_level2,object_items=problem.object_items)
        problem.obj_control.fuzzy_produce(Kis,None)
    return problem,Kis

def fpos_decision(a,b,r):
    k = -1/((a[1]-b[1])/(a[0]-b[0]))
    s = -k*a[0]+a[1]
    x1 = (-k*s+math.sqrt(k*k*s*s-(k*k+1)*(s*s-r*r)))/(k*k+1)
    x2 = (-k*s-math.sqrt(k*k*s*s-(k*k+1)*(s*s-r*r)))/(k*k+1)
    y1 = k*x1+s
    y2 = k*x2+s
    if x1 >= x2:
        tmpx = x1
        tmpy = y1
        x1 = x2
        x2 = tmpx
        y1 = y2
        y2 = tmpy
    if a[1] < k*b[0]+s:
        ju = 1
        if y1>=0 and y2>=0:
            x_min = x1
            x_max = x2
        elif y1<0 and y2<0:
            x_min = -r
            x_max = r
        elif y1>=0 and y2<0:
            x_min = x1
            x_max = r
        elif y1<0 and y2>=0:
            x_min = -r
            x_max = x2
    else:
        ju=-1
        if y1>=0 and y2>=0:
            x_min = -r
            x_max = r
        elif y1<0 and y2<0:
            x_min = x1
            x_max = x2
        elif y1>=0 and y2<0:
            x_min = -r
            x_max = x2
        elif y1<0 and y2>=0:
            x_min = x1
            x_max = r 
    
    x_rd = random.uniform(x_min,x_max)
    y_rd1 = k*x_rd+s
    if math.sqrt(x_rd*x_rd+y_rd1*y_rd1) > r*r:
        y_rd1 = -ju*math.sqrt(r*r-x_rd*x_rd)
    y_rd2 = ju*math.sqrt(r*r-x_rd*x_rd)
    if y_rd1 >= y_rd2:
        y_min = y_rd2
        y_max = y_rd1
    else:
        y_min = y_rd1
        y_max = y_rd2

    y_rd = random.uniform(y_min,y_max)
    theta_rd = random.uniform(0,2*math.pi)
    return np.array([x_rd,y_rd,theta_rd])
