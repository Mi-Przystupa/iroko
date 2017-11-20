# utility functions
import sys
import os
print(os.path.dirname(os.path.abspath(__file__))+'/../Iroko')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../Iroko')
from DCTopo import FatTreeTopo
from mininet.util import makeNumeric
from DCRouting import HashedRouting
from topo_ecmp import Fattree

TOPOS = {'ft': FatTreeTopo, 'ft2': Fattree }
ROUTING = {'ECMP' : HashedRouting}



def buildTopo(topo):
    topo_name, topo_param = topo.split( ',' )
    return TOPOS[topo_name](makeNumeric(topo_param))


def getRouting(routing, topo):
    return ROUTING[routing](topo)
