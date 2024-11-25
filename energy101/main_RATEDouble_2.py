import os
import sys
import optparse
import traci
import numpy as np
from sumolib import checkBinary
from Env_DQN.DuelingDQN  import *
import datetime
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

Training=True
num_hv = 95
num_av = 6
num_lane = 3
n_episodes = 300
# 定义warmup步长
Warmup_Steps = 10000

Testing=False
test_episodes = 10
load_dir='GRL_TrainedModels/DQN2022_09_18-20_23'
test_dir='GRL_TrainedModels/DQN2022_09_12-16:48/test'

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def get_step(id_info, information):
    N = information['n_vehicles']
    num_hv = information['n_hv']  # maximum number of HDVs
    num_lanes = information['n_lanes']
    ids = id_info['ids']
    AVs = id_info['AVs']
    HVs = id_info['HVs']
    Pos_x = id_info['Pos_x']
    Pos_y = id_info['Pos_y']
    Vel = id_info['Vel']
    LaneIndex = id_info['LaneIndex']
    Edge = id_info['Edge']

    # 初始化状态空间矩阵，邻接矩阵和mask
    states = np.zeros([N, 10 + num_lanes])
    adjacency = np.zeros([N, N])
    mask = np.zeros(N)
    intention_dic = {'HV': 0, 'AV': 1}
    light_dic = {'yrrGGGGrrrrryrrGGGGrrrrr': 0, 'grrrrrGGGrrrgrrrrrGGGrrr': 1, 'GrrrrrgrrGGGGrrrrrgrrGGG': 2,
                 'GGGrrrgrrrrrGGGrrrgrrrrr': 3,
                 'GrryyyGrrrrrGrryyyGrrrrr': 4, 'GrrrrrGyyrrrGrrrrrGyyrrr': 5, 'GrrrrrGrryyyGrrrrrGrryyy': 6,
                 'GyyrrrGrrrrrGyyrrrGrrrrr': 7}

    if AVs:
        # numerical data (speed, location)
        speeds = np.array(Vel).reshape(-1, 1)
        xs = np.array(Pos_x).reshape(-1, 1)
        ys = np.array(Pos_y).reshape(-1, 1)
        road = np.array(Edge).reshape(-1, 1)
        # categorical data 1 hot encoding: (lane location, intention)
        lanes_column = np.array(LaneIndex)  # 当前环境中的车辆所在车道的编号
        lanes = np.zeros([len(ids), num_lanes])  # 初始化车道onehot矩阵（当前车辆数量x车道数量）
        lanes[np.arange(len(ids)), lanes_column] = 1  # 根据每辆车当前所处的车道，在矩阵对应位置赋值为1
        types_column = np.array([intention_dic[traci.vehicle.getTypeID(i)] for i in ids])
        intention = np.zeros([len(ids), 2])  # 初始化intention矩阵（当前车辆数量x车辆种类）
        intention[np.arange(len(ids)), types_column] = 1
        types_column_l = np.array([light_dic[traci.trafficlight.getRedYellowGreenState('J3')] for i in ids])
        lights = np.zeros([len(ids), 8])  # 初始化light矩阵（当前车辆数量x车辆种类）
        lights[np.arange(len(ids)), types_column_l] = 1
        observed_states = np.c_[xs, ys, speeds, lanes, road,intention,lights[:,:4]]  # 将上述相关矩阵按列合成为状态观测矩阵

        # assemble into the NxF states matrix
        # 将上述对环境的观测储存至状态矩阵中
        states[:len(HVs), :] = observed_states[:len(HVs), :]
        states[num_hv:num_hv + len(AVs), :] = observed_states[len(HVs):, :]

        # 生成邻接矩阵
        # 使用sklearn库中的欧几里德距离函数计算环境中两两车辆的水平距离（x坐标，维度当前车辆x当前车辆）
        dist_matrix_x = euclidean_distances(xs)
        dist_matrix_y = euclidean_distances(ys)
        dist_matrix = np.sqrt(dist_matrix_x * dist_matrix_x + dist_matrix_y * dist_matrix_y)
        adjacency_small = np.zeros_like(dist_matrix)  # 根据dist_matrix生成维度相同的全零邻接矩阵
        adjacency_small[dist_matrix < 20] = 1
        adjacency_small[-len(AVs):, -len(AVs):] = 1  # 将RL车辆之间在邻接矩阵中进行赋值

        # assemble into the NxN adjacency matrix (这部分程序存疑)
        # 将上述small邻接矩阵储存至稠密邻接矩阵中
        adjacency[:len(HVs), :len(HVs)] = adjacency_small[:len(HVs), :len(HVs)]
        adjacency[num_hv:num_hv + len(AVs), :len(HVs)] = adjacency_small[len(HVs):, :len(HVs)]
        adjacency[:len(HVs), num_hv:num_hv + len(AVs)] = adjacency_small[:len(HVs), len(HVs):]
        adjacency[num_hv:num_hv + len(AVs), num_hv:num_hv + len(AVs)] = adjacency_small[len(HVs):,
                                                                        len(HVs):]

        # 构造mask矩阵
        mask[num_hv:num_hv + len(AVs)] = np.ones(len(AVs))

    return states, adjacency, mask


def check_state(rl_actions):
    ids = traci.vehicle.getIDList()
    EdgeList = traci.edge.getIDList()
    time_counter = traci.simulation.getTime()
    exist_V = []
    for num in range(len(traci.simulation.getArrivedIDList())):
        exist_V.append(traci.simulation.getArrivedIDList()[num])
    AVs = []
    HVs = []
    Pos_x = []
    Pos_y = []
    Vel = []
    LaneIndex = []
    Edge = []
    drastic_veh = []
    NO = 0
    for ID in ids:
        Pos_x.append(traci.vehicle.getPosition(ID)[0])
        Pos_y.append(traci.vehicle.getPosition(ID)[1])
        Vel.append(traci.vehicle.getSpeed(ID))
        current_lane = traci.vehicle.getLaneIndex(ID)
        if isinstance(rl_actions, torch.Tensor):

            rl_actions = rl_actions.cpu().numpy()
        # print(rl_actions)
        rl_actions2 = rl_actions.copy()
        rl_actions3 = rl_actions.copy()
        rl_actions2[rl_actions2 >2] = 1

        # print(next_lane)
        # print(next_lane)
        rl_actions3[rl_actions3< 3] = 14


        LaneIndex.append(current_lane)
        Edge.append(EdgeList.index(traci.vehicle.getRoadID(ID)))
        if traci.vehicle.getTypeID(ID) == 'AV':
            AVs.append(ID)

            # print(traci.vehicle.getSpeed(ID) + (rl_actions3[NO] - 18) / 2 * 0.1)

        elif traci.vehicle.getTypeID(ID) == 'HV':
            HVs.append(ID)
        rl_actions2 = rl_actions2[num_hv:num_hv + len(AVs)]
        rl_actions3 = rl_actions3[num_hv:num_hv + len(AVs)]
        if len(AVs) != 0:
            next_lane = np.clip(current_lane + rl_actions2[NO] - 1, 0, 2)
            traci.vehicle.setLaneChangeMode(ID, 0b000000000100)
            traci.vehicle.setSpeedMode(ID, 0b011111)
            traci.vehicle.changeLane(ID, next_lane,100)
            traci.vehicle.setSpeed(ID, traci.vehicle.getSpeed(ID) + (rl_actions3[NO] - 14)/2*0.3)
            # print(rl_actions3-8)
            NO += 1
        # if isinstance(rl_actions, torch.Tensor):
        #     rl_actions = rl_actions.cpu().numpy()
        #     # print(rl_actions)
        # rl_actions = rl_actions - 1
        # rl_actions2 = rl_actions.copy()
        # rl_actions3 = rl_actions2.copy()
        # rl_actions4 = rl_actions2.copy()
        # rl_actions3[rl_actions3 > 1] = 0  # langchange action
        #
        # rl_actions4[rl_actions4 < 2] = 18  # acc action
        #
        # # rl_actions3 = rl_actions3[num_hv:num_hv + len(AVs)]
        # # rl_actions4 = rl_actions4[num_hv:num_hv + len(AVs)]
        # next_lane = np.clip(current_lane + rl_actions3[NO], 0, 2)
        # # print(next_lane)
        #
        #
        #
        # NO += 1
        # LaneIndex.append(current_lane)
        # Edge.append(EdgeList.index(traci.vehicle.getRoadID(ID)))
        # if traci.vehicle.getTypeID(ID) == 'AV':
        #     AVs.append(ID)
        #     traci.vehicle.changeLane(ID, next_lane, 0)
        #     traci.vehicle.setAccel(ID, traci.vehicle.getSpeed(ID) + (rl_actions4[NO] - 18) / 2 * 0.1)
        # elif traci.vehicle.getTypeID(ID) == 'HV':
        #     HVs.append(ID)
        # print(current_lane)
    for ind, veh_id in enumerate(AVs):  # 这部分通过计算当前时间以及最后一次换道的时间间隔来检测车辆是否有激烈换到行为
        if rl_actions[ind] != 0 and (time_counter - traci.vehicle.getLastActionTime(veh_id) < 50):
            drastic_veh.append(veh_id)
    drastic_veh_id = drastic_veh
    id_list = {'ids': ids, 'EdgeList': EdgeList, 'exist_V': exist_V, 'AVs': AVs, 'HVs': HVs, 'Pos_x': Pos_x,
               'Pos_y': Pos_y, 'Vel': Vel, 'LaneIndex': LaneIndex, 'Edge': Edge, 'drastic_veh': drastic_veh,
               'drastic_veh_id': drastic_veh_id}






    return id_list


def get_reward(info_of_state, information):
    low_speed=10
    max_speed=30
    rei=0
    rep1=0
    rep2=0
    rti1 = 0
    rti2 = 0
    rti3 = 0
    rti4 = 0
    rtp = 0
    rti=0
    rtp1 = 0
    rtp2 = 0
    total = 0
    rci=0
    rcp1=0
    rcp2=0
    rcp=0
    rep=0
    r_task=0
    rd = 0
    rd2 = 0
    rd3 = 0
    rd5 = 0
    rdp = 0
    rdp1 = 0
    rdp2 = 0


    r_energy = 0
    r_wait1=0
    r_wait = 0
    r_wait2 = 0
    speed_reward = 0
    ids = traci.vehicle.getIDList()
    rl_ids = info_of_state['AVs']
    # print(rl_ids)
    max_av_speed = information['Max_speed_AV']
    if len(rl_ids) != 0:  # 若观测到RL车辆
        for rl_id in rl_ids:
            i = traci.vehicle.getSpeed(rl_id)
            y = traci.vehicle.getPosition(rl_id)
            if low_speed <= i <= max_speed:
                rei += 150*(i - low_speed)
                # print(rei)
            elif i > max_speed:
                rep1 += -(i - max_speed)
            elif i <= low_speed:
                rep2 -= 15*(low_speed-i)
        rep=rep1+rep2
        for rl_id in rl_ids:
            leader=traci.vehicle.getLeader(rl_id)
            follower=traci.vehicle.getFollower(rl_id)
            # print(follower)
            rl_x = traci.vehicle.getLanePosition(rl_id)
            rl_speed = traci.vehicle.getSpeed(rl_id)
            if leader != None :
                leader_speed = traci.vehicle.getSpeed(leader[0])
                leader_x = leader[1]
            else:
                leader_x=rl_x
                leader_speed=1
            if follower != ('', -1.0):
                follower=follower[0]
                if follower != None:

                  follower_speed=traci.vehicle.getSpeed(follower)
                  follower_x=traci.vehicle.getLanePosition(follower)
            else:
                follower_x=rl_x
                follower_speed=1

            tl = (leader_x - rl_x) / (rl_speed - leader_speed + 0.010)
            tf = (rl_x - follower_x) / (follower_speed - rl_speed + 0.011)
            if 15 > tf > 8:
                rti1 += (tf -8)
            if 15 > tl > 8:
                rti2 += (tl - 8)
            if tf < 0 or tf > 15:
                rti3 += 10
            if tl < 0 or tl > 15:
                rti4 += 10
            if 0 < tl < 8:
                rtp1 -= min(2**(8 - round(tl,1)),2**8)
            if 0 < tf < 8:
                rtp2 -= min(2**(8- round(tf,1)),2**8)
        rti=rti1+rti2+rti3+rti4
        for rl_id in rl_ids:
            time=traci.vehicle.getWaitingTime(rl_id)
            if time<300:
                r_wait1-=time/4
            if 800>time>300:
                r_wait2-=75+(time-300)*2.5
        r_wait=r_wait1+r_wait2
        # print(speed_reward)
        for rl_id in rl_ids:
            acc=traci.vehicle.getAcceleration(rl_id)
            if -4<acc<4:
                rci += min(5*np.exp(-abs(acc)/2),5)
        for id in ids:
            acc = traci.vehicle.getAcceleration(id)
            if acc<=-4:
                rcp2+= -500

        for rl_id in rl_ids:
            if traci.vehicle.getElectricityConsumption(rl_id)>0:
              r_energy -= traci.vehicle.getElectricityConsumption(rl_id)*50
            # print(r_energy)

        for rl_id in rl_ids:
            AV_currentedge=traci.vehicle.getLaneID(rl_id)
            speed=traci.vehicle.getSpeed(rl_id)
            # if AV_currentedge =='-E0_0' or'-E0_1' or'-E0_2':
            #     r_task+=10*speed
            diatance=traci.vehicle.getDistance(rl_id)

            # print(diatance)
            lane=traci.vehicle.getLaneIndex(rl_id)


            if 0<diatance<=710 and (lane=='0' or '0'):
                rd2 += 10
            if 800<diatance<=1300 and (lane=='2' or '2'):
                rd3 += 10

            if 1500 < diatance <=2400 and (lane == '0' or '0'):
                rd5 += 10
            if 1000 < diatance <= 1300 and (lane == '0' or '1'):
                rdp1 -= 10
            if 1500 < diatance <= 2400 and (lane == '2' or '1'):
                rdp2 -= 10
        rd=rd2+rd3+rd5
        rdp=rdp1+rdp2
            # penalty for frequent lane changing behavors
    # 这部分计算对频繁换道的处罚（负奖励）

    drastic_veh_id = info_of_state['drastic_veh_id']
    if drastic_veh_id:
        rcp1 -= 50*len(drastic_veh_id)
    rtp3 = 0
    crash_num = traci.simulation.getCollidingVehiclesNumber()
    # print(crash_num)
    rtp3 -= crash_num * 4000
    rcp=rcp1+rcp2
    rtp = rtp1 + rtp2+rtp3

    # 对于碰撞的惩罚（负奖励）
    # +rti+rtp+rci+rcp+r_energy+r_task
    qp = 0.4
    qi = 0.6
    ksi = qi
    kei = qi
    kci = qi
    kti = qi
    ksp = qp
    ktp = qp
    kep = qp
    kcp = qp
    if rtp < 0:
        ksi = qi * np.exp((rdp+rep+r_wait)/10000)
        kei = qi * np.exp(rdp/10000)
        kci = qi* np.exp(rdp+rep/10000)
    #     kti = qi * np.exp((rsp1 + rcp1) / 150000)
    # if rsi>=50*20*0.9:
    #     kcp=0.65*np.exp((900-rsi)/100)
    #     kep = 0.65 * np.exp((900 - rsi) / 100)
    #     ktp = 0.65 * np.exp((900 - rsi) / 100)

    # Ak1 = np.matrix([[0.2, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.35]])
    Ak1 = np.matrix([[0.3, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0.2, 0], [0, 0, 0, 0.3]])
    # Ak1 = np.matrix([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.2, 0], [0, 0, 0, 0.6]])
    # Ak1 = np.matrix([[0.4, 0, 0, 0], [0, 0.4, 0, 0], [0, 0, 0.05, 0], [0, 0, 0, 0.15]])
    Ak2 = np.matrix([[ksi, ksp, 0, 0, 0, 0, 0, 0],
                     [0, 0, kei, kep, 0, 0, 0, 0],
                     [0, 0, 0, 0, kci, kcp, 0, 0],
                     [0, 0, 0, 0, 0, 0, kti, ktp]]
                    )
    Ar = np.matrix([[float(rd), 0, 0, 0],
                    [float(r_energy+rdp), 0, 0, 0],
                    [0, float(rei), 0, 0],
                    [0, float(rep+r_wait), 0, 0],
                    [0, 0, float(rci), 0],
                    [0, 0, float(rcp), 0],
                    [0, 0, 0, float(rti)],
                    [0, 0, 0, float(rtp)]
                    ])
    # print(float(rsi))
    A = Ak1 * Ak2 * Ar
    return (A[0,0]+A[1,1]+A[2,2]+A[3,3])/10
    # return rtp


# 定义warmup步长记录
warmup_count = 0
now_time = datetime.datetime.now()
now_time = datetime.datetime.strftime(now_time, '%Y_%m_%d-%H:%M')
save_dir = 'GRL_TrainedModels/DQN' + now_time
try:
    os.makedirs(save_dir)
except:
    pass

info_dict = {'n_vehicles': num_hv + num_av, 'n_hv': num_hv, 'n_av': num_av, 'n_lanes': num_lane,
             'Max_speed_AV': 30, 'Max_speed_HV': 20}
if Training:
    Rewards = []  # 初始化奖励矩阵以进行数据保存
    Loss = []  # 初始化Loss矩阵以进行数据保存
    Episode_Steps = []  # 初始化步长矩阵保存每一episode的任务完成时的步长
    Average_Q = []  # 初始化平均Q值矩阵保存每一episode的平均Q值
    collision=[]
    energy=[]
    AVspeed=[]
    AVchangelane=[]

    GRL_Net, GRL_model = Create_DQN(num_hv, num_av, info_dict)

    print("#------------------------------------#")
    print("#----------Training Begins-----------#")
    print("#------------------------------------#")

    for i in range(1, n_episodes + 1):

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        if __name__ == "__main__":
            options = get_options()
            if options.nogui:
                sumoBinary = checkBinary('sumo')
            else:
                sumoBinary = checkBinary('sumo-gui')
            traci.start(['sumo' , "-c", "Env_Net/well.sumocfg"])
            action = np.zeros(GRL_Net.num_agents)  # 生成original动作
            list_info = check_state(action)
            obs = get_step(list_info, info_dict)
            R = 0  # 行为奖励
            t = 0  # 时间步长
            A_e=0
            A_s1=[]
            A_s=0
            A_change=0
            crash_name=0
            for step in range(0, 4000):
                # ------动作生成------ #
                if warmup_count <= Warmup_Steps:  # 进行warmup
                    action = np.random.choice(
                        np.arange(GRL_Net.num_outputs), GRL_Net.num_agents)  # 生成随机动作
                else:
                    action = GRL_model.choose_action(obs)  # agent与环境进行交互

                list_info = check_state(action)
                obs_next = get_step(list_info, info_dict)
                reward = get_reward(list_info, info_dict)
                exist_AV = []
                ids = traci.vehicle.getIDList()
                AV_energy=0
                rl_speed=0
                for id in ids:
                    if traci.vehicle.getTypeID(id) == 'AV':
                        AV_energy += traci.vehicle.getElectricityConsumption(id)
                        exist_AV.append(id)
                        rl_speed += traci.vehicle.getSpeed(id)
                if len(exist_AV)!=0:
                   AV_speed=rl_speed/len(exist_AV)
                   A_s1.append(AV_speed)
                A_change+=len(list_info['drastic_veh_id'])
                R += reward
                t += 1
                crash_num1 = traci.simulation.getCollidingVehiclesNumber()
                crash_name+=crash_num1
                A_e+=AV_energy

                warmup_count += 1
                done = ((len(exist_AV) == 0) and(step>1000))  or (crash_name>=600)
                # ------将交互结果储存到经验回放池中------ #
                GRL_model.store_transition(obs, action, reward, obs_next, done)
                # ------进行策略更新------ #
                GRL_model.learn()
                # ------环境观测更新------ #
                obs = obs_next
                traci.simulationStep()
                if done :
                    break
            traci.close()
            # ------记录训练数据------ #
            # 获取训练数据
            training_data = GRL_model.get_statistics()
            loss = training_data[0]
            q = training_data[1]
            # 记录训练数据
            for i in A_s1:
                A_s += i
            A_s=A_s/len(A_s1)
            Rewards.append(R)  # 记录Rewards
            Episode_Steps.append(t)  # 记录Steps
            Loss.append(loss)  # 记录loss
            Average_Q.append(q)  # 记录平均Q值
            collision.append(crash_name)
            energy.append(A_e)
            AVspeed.append(A_s)
            AVchangelane.append(A_change)

            if i % 1 == 0:
                print('Training Episode:', i, 'Reward:', R, 'Loss:', loss, 'Average_Q:', q)
            plt.figure(1)
            plt.subplot(2, 4, 1)
            plt.title('Rewards')
            plt.plot(Rewards)
            plt.subplot(2, 4, 2)
            plt.title('Episode_Steps')
            plt.plot(Episode_Steps)
            plt.subplot(2, 4, 3)
            plt.title('Loss')
            plt.plot(Loss)
            plt.subplot(2, 4, 4)
            plt.title('Average_Q')
            plt.plot(Average_Q)
            plt.subplot(2, 4, 5)
            plt.title('collision')
            plt.plot(collision)
            plt.subplot(2, 4, 6)
            plt.title('energy')
            plt.plot(energy)
            plt.subplot(2, 4, 7)
            plt.title('AVspeed')
            plt.plot(AVspeed)
            plt.subplot(2, 4, 8)
            plt.title('AVchangelane')
            plt.plot(AVchangelane)
            plt.show(block=False)
            plt.pause(1)
    print('Training Finished.')
    # 模型保存
    GRL_model.save_model(save_dir)
    # 保存训练过程中的各项数据
    np.save(save_dir + "/Rewards", Rewards)
    np.save(save_dir + "/Episode_Steps", Episode_Steps)
    np.save(save_dir + "/Loss", Loss)
    np.save(save_dir + "/Average_Q", Average_Q)
    np.save(save_dir + "/collision", collision)
    np.save(save_dir + "/energy", energy)
    np.save(save_dir + "/AVspeed", AVspeed)
    np.save(save_dir + "/AVchangelane", AVchangelane)
    np.savetxt(save_dir + '/rewards.txt', Rewards, delimiter=',')
    np.savetxt(save_dir + '/Episode_Steps.txt', Episode_Steps, delimiter=',')
    np.savetxt(save_dir + '/losses.txt', Loss, delimiter=',')
    np.savetxt(save_dir + '/AverageQ.txt', Average_Q, delimiter=',')
    np.savetxt(save_dir + '/collision.txt', collision, delimiter=',')
    np.savetxt(save_dir + '/energy.txt', energy, delimiter=',')
    np.savetxt(save_dir + '/AVspeed.txt', AVspeed, delimiter=',')
    np.savetxt(save_dir + '/AVchangelane.txt', AVchangelane, delimiter=',')
    plt.figure(1)
    plt.savefig(save_dir + '/datas.png')

if Testing:
    Rewards = []  # 初始化奖励矩阵以进行数据保存
    GRL_Net, GRL_model = Create_DQN(num_hv, num_av, info_dict)
    GRL_model.load_model(load_dir)

    print("#-------------------------------------#")
    print("#-----------Testing Begins------------#")
    print("#-------------------------------------#")
    for i in range(1, test_episodes + 1):
        traci.start(['sumo-gui', "-c", "Env_Net/well.sumocfg"])
        action = np.zeros(GRL_Net.num_agents)  # 生成original动作
        list_info = check_state(action)
        obs = get_step(list_info, info_dict)
        R = 0  # 行为奖励
        t = 0  # 时间步长
        for step in range(0, 2500):
            # ------动作生成------ #
             # 生成随机动作

            action = GRL_model.choose_action(obs)  # agent与环境进行交互

            list_info = check_state(action)
            obs_next = get_step(list_info, info_dict)
            reward = get_reward(list_info, info_dict)
            R += reward
            t += 1
            warmup_count += 1
            done = False
            # ------环境观测更新------ #
            obs = obs_next
            traci.simulationStep()
            if done:
                break
        traci.close()
        # traci.start(['sumo', "-c", "Env_Net/well.sumocfg"])
        # action = np.zeros(GRL_Net.num_agents)  # 生成original动作
        # list_info = check_state(action)
        # obs = get_step(list_info, info_dict)
        # if 'SUMO_HOME' in os.environ:
        #     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        #     sys.path.append(tools)
        # else:
        #     sys.exit("please declare environment variable 'SUMO_HOME'")
        #
        # R = 0
        # t = 0
        #
        # action = GRL_model.test_action(obs)
        # list_info = check_state(action)
        # obs_next = get_step(list_info, info_dict)
        # reward = get_reward(list_info, info_dict)
        # R += reward
        # t += 1
        # done = False
        # if done :
        #     break
        # traci.close()

        print('Evaluation Episode:', i, 'Reward:', R)

    print('Evaluation Finished')

    # 测试数据保存
    np.savetxt(test_dir + "/Test", Rewards)