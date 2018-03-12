Before running this experiment:
Install Mininet
Install Ryu
Guides should be available in the repos. 


This current system launches a ECMP fattree topology with a connected controller

To run mininet (and the simulation) type:

To run the controller in another window run:


In Mininet you can use 

pingall 
to ping all nodes

nodes
to list all nodes

x ping y
to ping a specifc node

x iperf y
to iperf to a node

iperfall
to blast the network with iperf (Caution will kill your machine)

I recommend reading Mininet guides to become comfortable with the system