import os
import sys
import optparse
import traci
import numpy as np
from sumolib import checkBinary


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


for i in range(1, 20):

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
        traci.start([sumoBinary, "-c", "well.sumocfg"])
        exist_V = []
        for step in range(0, 3600):
            for num in range(len(traci.simulation.getArrivedIDList())):
                exist_V.append(traci.simulation.getArrivedIDList()[num])
            # if len(traci.simulation.getArrivedIDList()) != 0:
            #     if traci.simulation.getArrivedIDList()[0] not in exist_V:
            #         exist_V.append(traci.simulation.getArrivedIDList()[0])
            IDlist = traci.vehicle.getIDList()
            EdgeList = traci.edge.getIDList()
            # exist_V = []
            # print(IDlist)
            # print(EdgeList)
            AVs = []
            HVs = []
            Pos_x = []
            Pos_y = []
            Vel = []
            LaneIndex = []
            Edge = []
            for ID in IDlist:
                # if ID not in exist_V and traci.vehicle.getRoadID(ID) == 'E11':
                #     exist_V.append(ID)
                Pos_x.append(traci.vehicle.getPosition(ID)[0])
                Pos_y.append(traci.vehicle.getPosition(ID)[1])
                Vel.append(traci.vehicle.getSpeed(ID))
                LaneIndex.append(traci.vehicle.getLaneIndex(ID))
                Edge.append(EdgeList.index(traci.vehicle.getRoadID(ID)))
                if traci.vehicle.getTypeID(ID) == 'AV':
                    AVs.append(ID)
                elif traci.vehicle.getTypeID(ID) == 'HV':
                    HVs.append(ID)
            # print(AVs)
            # print(HVs)
            # print(Pos_x)
            # print(Pos_y)
            # print(Vel)
            # print(LaneIndex)
            # print(Edge)
            print(exist_V)
            # if len(Pos) != 0:
            #     print(type(Pos[0]))
            # for ID in IDlist:
            #     print(traci.vehicle.getSpeed(ID))
            #        traci.vehicle.setSpeed('a12.5',10)
            #        print(traci.vehicle.getIDList())
            #        print(traci.edge.getIDList())
            #        print(traci.inductionloop.getVehicleData('abcd'))
            traci.simulationStep()
        traci.close()
