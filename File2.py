#!/usr/bin/env python
# coding: utf-8

# # Question 1: Plague model

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Defning the Plague Function for the Model
def plague(ic, t):
    s_t, i_t = ic[0], ic[1]
    b, beta, k = 3, 3, 3
    dsdt = (b*s_t) - (beta*i_t*s_t)
    didt = (beta*i_t*s_t) - (k*i_t)
    return [dsdt, didt]

#Plotting the Function Plot for the Plague Model
for iic in [0.1,0.5,1.3]:
    # Stating the Initial Conditions
    ic = [1, iic]
    # Defining Time Space for the Model
    t = np.linspace(0,20,1000)
    # Obtaining the Integrated Result of the Differential Model
    result = odeint(plague, ic, t)
    s_t, i_t = result[:,0], result[:,1]
    # Plotting the Function Plot
    plt.title("Plague Function Plot When ($I_{0} = " + str(iic) + "$ )")
    plt.xlabel('t')
    plt.ylabel('S(t) I(t)')
    plt.plot(t,s_t,label='S')
    plt.plot(t,i_t,label='I')
    plt.legend(loc=1)
    # Saving the Plots into a Folder
    plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q1/Plague Function Plot when I(0) = {}.png'.format(iic))
    plt.show()


# In[4]:


#Plotting the Phase Plot for the Plague Model

# Plotting the Characteristics Line of the Model on the Phase Plot
st = np.linspace(0.0, 4.0, 20)
it = np.linspace(0.0, 4.0, 20)
St, It = np.meshgrid(st, it)
u, v = np.zeros(St.shape), np.zeros(It.shape)
n1, n2 = St.shape
for i in range(n1):
    for j in range(n2):
        x = St[i, j]
        y = It[i, j]
        res_t = plague([x, y], 0)
        u[i,j] = res_t[0]
        v[i,j] = res_t[1]
plt.quiver(St, It, u, v, color='r')
plt.xlabel('S(t)')
plt.ylabel('I(t)')
plt.xlim([0, 4])
plt.ylim([0, 4])

# Plotting the solutions of the Model given the Initial Conditions on the Phase Plot
for iic in [0.1, 0.5, 1.3]:    
    ic = [1, iic]
    t = np.linspace(0,20,1000)
    result = odeint(plague, ic, t)
    if iic==0.1:
        color='royalblue'
    elif iic == 0.5:
        color = 'darkorange'
    else:
        color = 'limegreen'
    plt.title("Phase Plane for the Plague Model")
    plt.plot(result[:,0], result[:,1], color=color, label='I(0)='+str(iic)) # path
    plt.plot([result[0,0]], [result[0,1]], 'o',color='blueviolet') # start
    plt.plot([result[-1,0]], [result[-1,1]], 's',color='fuchsia') # end
    
plt.xlim([0, 4])
plt.legend(loc=1)
# Saving the Phase Plot into a Folder
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q1/Phase Plane for the Plague Model.png')
plt.show()


# # Question 2: SIR model
W. O Kermack and A. G. McKendrick created a model in which they considered a fixed population with three compartments: susceptible S, infected I and removed R. The code is presented below with the Basic Reproductive Ratio.
# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function for the Disease Model by Kermack and McKendrick 
def kerken(ic, t, bet, r):
    s_t, i_t, r_t = ic[0], ic[1], ic[2]
    dsdt = -(bet*s_t*i_t)
    didt = (bet*s_t*i_t) - (r*i_t)
    drdt = r*i_t
    return [dsdt, didt, drdt]

# Basic Reproductive Ratio Function
def brr(bet, r, n):
    r0 = (bet*n)/r
    return r0

# Defining Starting Variables
n = 1000
t = np.linspace(0,1,100)
beta = 0.1
rr = 300
infect = 0
# Plotting the Function Plots of the Model within the Set Population
while infect < n:
    ic = [n,1,0]
    sir = odeint(kerken, ic, t, args=(beta, rr))
    s,i,r = sir[:,0], sir[:,1], sir[:,2]
    infect = sum(sir[:,1])
    rnot = brr(beta, rr, n)
    if rnot > 1:
        plt.title("Function Plots with $beta = " + str(beta) + ", $r = " + str(rr) + " - Status: Epidemic")
    else:
        plt.title("Function Plots with $beta = " + str(beta) + ", $r = " + str(rr) + " - Status: No Epidemic")
    plt.plot(t, s, label='S')
    plt.plot(t, i, label='I')
    plt.plot(t, r, label='R')
    plt.legend(loc=1)
    plt.grid(True, linestyle='dotted')
    plt.xlabel('t')
    plt.ylabel('N')
    # Saving the Function Plots into a Folder
    plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/kenker/function plots with beta = {} and r = {}.png'.format(beta,rr),transparent=False)
    plt.show()
    beta+=0.05
    rr-=5
        


# In[6]:


# Making the Function Plots into a Movie
import imageio 
import os
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/kenker/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/Kenker_Function_Plots.movie.gif", images,duration=1.2)

The basic SIR model can be reduced to a two-dimensional system, because the variable for recovered individuals does not appear in the equations of the other two variables. The code for the reduced SI system with varying initial conditions for the infection is presented below:
# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Defning the Function for the Reduced Model
def kerken2(ic, t, beta, r):
    s_t, i_t = ic[0], ic[1]
    dsdt = -(beta*s_t*i_t)
    didt = (beta*s_t*i_t) - (r*i_t)
    return [dsdt, didt]

# Plotting the Characteristics Line of the Model on the Phase Plot
beta, r = 0.2, 170
st = np.linspace(0.0, 1000, 30)
it = np.linspace(0.0, 200, 10)
St, It = np.meshgrid(st, it)
u, v = np.zeros(St.shape), np.zeros(It.shape)
n1, n2 = St.shape
for i in range(n1):
    for j in range(n2):
        x = St[i, j]
        y = It[i, j]
        res_t = kerken2([x, y], 0, beta, r)
        u[i,j] = res_t[0]
        v[i,j] = res_t[1]
plt.quiver(St, It, u, v, color='r')
plt.xlabel('S(t)')
plt.ylabel('I(t)')
plt.xlim([0, 1000])
plt.ylim([0, 200])

# Plotting the solutions of the Model given the Initial Conditions on the Phase Plot
for iic in [1, 50, 100, 150]:  
    ic = [1000, iic]
    t = np.linspace(0,20,1000)
    result = odeint(kerken2, ic, t, args=(beta, r))
    if iic== 1:
        color='gold'
    elif iic == 50:
        color = 'slateblue'
    elif iic == 100:
        color = 'turquoise'
    else:
        color = 'blue'
    plt.title("SIR Model - Two Dimensional with beta = 0.2 and r = 170")
    plt.plot(result[:,0], result[:,1], color=color, label='I(0)='+str(iic)) # path
    plt.plot([result[0,0]], [result[0,1]], 'o',color='c') # start
    plt.plot([result[-1,0]], [result[-1,1]], 's',color='k') # end
plt.xlim([0, 1000])
plt.legend(loc=2)
# Saving the Phase Plot into a Folder
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q2/SIR Model - Two Dimensional.png')
plt.show()

Determining the total number of infected individuals during an epidemic and plotting it against the basic reproductive ratio
# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function for the Disease Model by Kermack and McKendrick 
def kerken3(ic, t, beta, r):
    s_t, i_t, r_t = ic[0], ic[1], ic[2]
    dsdt = -(beta*s_t*i_t)
    didt = (beta*s_t*i_t) - (r*i_t)
    drdt = r*i_t
    return [dsdt, didt, drdt]

# Basic Reproductive Ratio Function
def brr(beta, r, n):
    r0 = (beta*n)/r
    return r0

# Estimating the Total Number of Infected Individuals and the Basic Reproductive Ratio
n = 1000
t = np.linspace(0,1,100)
inf=[]
rnot=[]
beta = 0.1
rr = 300
infect = 0
while infect < n:
    ic = [n,1,0]
    sir = odeint(kerken3, ic, t, args=(beta, rr))
    it = sir[:,1]
    rrn = brr(beta, rr, n)
    infect = sum(it)
    rnot.append(rrn)
    inf.append(infect)
    beta+=0.05
    rr-=5

# Plotting the Total Number of Infected Individuals against the Basic Reproductive Ratio
plt.title("Total Number of Infection Plot")
plt.plot(rnot, inf)
plt.grid(True, linestyle='dotted')
plt.xlabel('$R_{0}$')
plt.ylabel('Total Number of Infection')
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/kenker3/function plots I(t) against BRR.png',transparent=False)
plt.show()


# # Question 3: SIR Model on a Network
SIR model on a network You are given a graph representing the contact network. 
So an edge between the nodes v and w means that if v becomes infected at some point, the disease has the potential to spread directly to w. Each node may go through the Susceptible-Infectious-Removed cycle.
The progress of the epidemic is controlled by the contact network structure and an additional quantity: the probability of contagion p (for the sake of simplicity we assume that the length of the infection is exactly one time step)
# In[1]:


#A function to be use later to help the gif run the graph without mixing next graphs
def nmes(i,n):
    if i<=n:
        if (i<100) or (2<len(str(i))<4):
            t=int(str(i)[0:2])
            if 0<(i or t)<10:
                prf="a"
            elif 9<(i or t)<20:
                prf='b'
            elif 19<(i or t)<30:
                prf='c'
            elif 29<(i or t)<40:    
                prf='d'
            elif 39<(i or t)<50:    
                prf='e'
            elif 49<(i or t)<60:    
                prf='f'
            elif 59<(i or t)<70:    
                prf='g'
            elif 69<(i or t)<80:    
                prf='h'
            elif 79<(i or t)<90:    
                prf='k'
            elif 89<(i or t)<100:
                prf='l'      
            if len(str(i)) ==(3 or 4):
                prf='m'+ str(t)
    return prf


# In[7]:


import matplotlib.pyplot as plt
import networkx as nx 
import random
# Function for the SIR Model on the Network
def sirmodel(G, p, infection_start_node):
    pos=nx.spring_layout(G)
    infectnode = [infection_start_node]
    recovernode = []
    permanent = []
    susceptnode = G.nodes()
    i=0
    while len(infectnode) > 0:
        # Calculating and Printing the Number of Susceptible, Infected and Recovered Individuals
        susceptnodel = len(G.nodes()) - len(infectnode) - len(recovernode)
        print(susceptnodel, len(infectnode), len(recovernode))
        # Plotting and Saving the Graph of the Susceptible, Infected and Recovered Individuals
        plt.title("SIR Model on a Network: Time =" + str(i+1))
        nx.draw_networkx(G, pos=pos, node_color="blue",font_color="black",edge_color="black")
        nx.draw_networkx_nodes(G, pos=pos, nodelist=infectnode, node_color="red")
        nx.draw_networkx_nodes(G, pos=pos, nodelist=recovernode, node_color="green")
        z1=nmes(i+1,n)
        plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3image/SIR on a network=a'+str(z1)+'{}.png'.format(i+1),transparent=False)
        # Declaring a Dummy List for the Susceptible and Infection List
        susceptible_list = []
        infection_list = []
        # Getting the Infected Individuals (Neighbours of the Presently Infected Individuals) with probability p
        for infected_node in infectnode:
            for node in G.neighbors(infected_node):
                gp = random.uniform(0,1)
                if gp >= p:
                    if node not in infection_list:
                        infection_list.append(node)
        # New Recovered List: Turning the Previously Infected Nodes to Recovered
        for node in infectnode:
            if node not in recovernode:
                recovernode.append(node)
        # New Infected List: Making Sure Recovered Individuals do not Reflect in the Infection List Since we are considering SIR Model
        infectnode = []
        for node in infection_list:
            if node not in recovernode:
                infectnode.append(node)
        # New Susceptible List: Getting the Susceptible Individuals
        for node in G.nodes():
            if node not in recovernode or node not in infectnode:
                susceptible_list.append(node)
        susceptnode = susceptible_list
        # Additional Condition Imposed for Printing the Output When Infection is Zero
        if len(infectnode) == 0:
            susceptnode = len(G.nodes()) - len(infectnode) - len(recovernode)
            print(susceptnode, len(infectnode), len(recovernode))
            plt.title("SIR Model on a Network: Time =" + str(i+1))
            nx.draw_networkx(G, pos=pos, node_color="blue",font_color="black",edge_color="black")
            nx.draw_networkx_nodes(G, pos=pos, nodelist=infectnode, node_color="red")
            nx.draw_networkx_nodes(G, pos=pos, nodelist=recovernode, node_color="green")
            z1=nmes(i+1,n)
            plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3image/SIR on a network=a'+str(z1)+'{}.png'.format(i+1),transparent=False)        
        i+=1
    return


# In[8]:


# Simulation for the Random Graph
import imageio 
import os
G = nx.erdos_renyi_graph(100,0.7)
p = 0.6
n = 100
infection_start_node = 0
sirmodel(G, p, infection_start_node)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3image/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):      
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/SIR_on_RG_network.movie"+str(n)+".movie.gif", images,duration=1.2)


# In[9]:


# Simulation for the Watts Strogatz Graph
import imageio 
import os
G = nx.watts_strogatz_graph(100, 5, 0.7) 
p = 0.6
n = 100
infection_start_node = 0
sirmodel(G, p, infection_start_node)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3image/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):      
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/SIR_on_WS_network.movie"+str(p)+".movie.gif", images,duration=1.2)


# In[10]:


# Simulation for the Barabasi Albert Graph
import imageio 
import os
G = nx.barabasi_albert_graph(100,5) 
p = 0.6
n = 100
infection_start_node = 0
sirmodel(G, p, infection_start_node)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3image/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):      
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/SIR_on_BA_network.movie"+str(n)+".movie.gif", images,duration=1.2)

Record the fraction of infected nodes in the network at each time point for three different values of p. Because the SIR dynamics is stochastic, you will want to simulate each infection multiple times with the same starting node. Plot the average of this runs over time for each value of p.
# In[11]:


# THE FUNCTION WAS MODIFIED TO INCLUDE LIST OF SUSCEPTIBLE, INFECTED AND RECOVERED AS OUTPUT
import matplotlib.pyplot as plt
import networkx as nx 
import random, numpy
# Function for the SIR Model on the Network
def sirtmodel(G, p, t, infection_start_node):
    pos=nx.spring_layout(G)
    infectnode = [infection_start_node]
    recovernode = []
    permanent = []
    susceptnode = G.nodes()
    infectoutput = []
    recoveroutput = []
    susceptoutput = []
    for i in range(t):
        # Calculating and Printing the Number of Susceptible, Infected and Recovered Individuals
        susceptnodel = len(G.nodes()) - len(infectnode) - len(recovernode)
        infectoutput.append(len(infectnode))
        recoveroutput.append(len(recovernode))
        susceptoutput.append(susceptnodel)
        # Declaring a Dummy List for the Susceptible and Infection List
        susceptible_list = []
        infection_list = []
        # Getting the Infected Individuals (Neighbours of the Presently Infected Individuals) with probability p
        for infected_node in infectnode:
            for node in G.neighbors(infected_node):
                gp = random.uniform(0,1)
                if gp >= p:
                    if node not in infection_list:
                        infection_list.append(node)
        # New Recovered List: Turning the Previously Infected Nodes to Recovered
        for node in infectnode:
            if node not in recovernode:
                recovernode.append(node)
        # New Infected List: Making Sure Recovered Individuals do not Reflect in the Infection List Since we are considering SIR Model
        infectnode = []
        for node in infection_list:
            if node not in recovernode:
                infectnode.append(node)
        # New Susceptible List: Getting the Susceptible Individuals
        for node in G.nodes():
            if node not in recovernode or node not in infectnode:
                susceptible_list.append(node)
        susceptnode = susceptible_list
    infectoutpute, recoveroutpute, susceptoutpute = numpy.array(infectoutput),numpy.array(recoveroutput),numpy.array(susceptoutput)
    return (susceptoutpute,infectoutpute,recoveroutpute)


# In[12]:


n, t = 100, 10
G1 = nx.erdos_renyi_graph(n,0.7)
G2 = nx.watts_strogatz_graph(n, 5, 0.7) 
G3 = nx.barabasi_albert_graph(n,5) 
infection_start_node = 0
probs = [0.3,0.5,0.7]
trange = [i+1 for i in range(10)]

# Plotting Fraction of Infection Plot per Time for the Random Graph
for p in probs:
    infgrap = sirtmodel(G1, p, t, infection_start_node)
    plt.title("Random Graph: Fraction of Infection Plot per Time")
    plt.plot(trange, ((infgrap[1])/n), label = "p = "+str(p))
    plt.grid(True, linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Fraction of Infection')
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/RG Fracton Infection plot against Time.png',transparent=False)
plt.legend(loc=1)
plt.show()

# Plotting Fraction of Infection Plot per Time for the WS Graph
for p in probs:
    infgrap = sirtmodel(G2, p, t, infection_start_node)
    plt.title("WS Graph: Fraction of Infection Plot per Time")
    plt.plot(trange, ((infgrap[1])/n), label = "p = "+str(p))
    plt.grid(True, linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Fraction of Infection')
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/WS Fracton Infection plot against Time.png',transparent=False)
plt.legend(loc=1)
plt.show()

# Plotting Fraction of Infection Plot per Time for the BA Graph
for p in probs:
    infgrap = sirtmodel(G3, p, t, infection_start_node)
    plt.title("BA Graph: Fraction of Infection Plot per Time")
    plt.plot(trange, ((infgrap[1])/n), label = "p = "+str(p))
    plt.grid(True, linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Fraction of Infection')
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/BA Fracton Infection plot against Time.png',transparent=False)
plt.legend(loc=1)
plt.show()

Now select at least 20 different values of p. Simulate the SIR dynamics on the network starting with a random node. Measure the total propotion of the network that becomes infected, the time to clear infection and the time to the largest number of infected nodes. Be sure to simulate the infection enough times (each run starting from a different randomly chosen starting node) that you can reasonably estimate the mean of each of these measures. For each measure plot it as a function of p. Make one plot for each measure, including a separate line (labeled appropriately) for each network.
# In[13]:


n, t = 100, 10
G1 = nx.erdos_renyi_graph(n,0.7)
G2 = nx.watts_strogatz_graph(n, 5, 0.7) 
G3 = nx.barabasi_albert_graph(n,5) 
GU = [G1, G2, G3]
GUname = ["Random Graph","WS Graph","BA Graph"]
infection_start_node = 0
probs = [i*0.05 for i in range(1,21)]
for G in GU:
    total_prop_infect = []
    for p in probs:
        infgrap = sirtmodel(G, p, t, infection_start_node)
        count = 0
        sume = 0
        for i in infgrap[1]:
            if i != 0:
                count+=1
                sume = sume + i
        tot_prop_infect = sume/(n)
        total_prop_infect.append(tot_prop_infect)
    plt.title("Total Proportion of Infection Against Probability")
    plt.plot(probs, total_prop_infect, label = GUname[GU.index(G)])
    plt.grid(True, linestyle='dotted')
plt.xlabel('Probability of Infection')
plt.ylabel('Total Proportion of Infection ')
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/Total Proportion of Infection Against Probability.png',transparent=False)
plt.legend(loc='best')
plt.show()


# In[14]:


n, t = 100, 10
G1 = nx.erdos_renyi_graph(n,0.7)
G2 = nx.watts_strogatz_graph(n, 5, 0.7) 
G3 = nx.barabasi_albert_graph(n,5) 
GU = [G1, G2, G3]
GUname = ["Random Graph","WS Graph","BA Graph"]
infection_start_node = 0
probs = [i*0.05 for i in range(1,21)]
for G in GU:
    time_to_clear_infect = []
    for p in probs:
        infgrap = sirtmodel(G, p, t, infection_start_node)
        count = 0
        for i in infgrap[1]:
            if i != 0:
                count+=1
        time_to_clear_infect.append(count)
    plt.title("Time to Clear Infection Against Probability")
    plt.plot(probs, time_to_clear_infect, label = GUname[GU.index(G)])
    plt.grid(True, linestyle='dotted')
plt.xlabel('Probability of Infection')
plt.ylabel('Time to Clear Infection')
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/Time to Clear Infection Against Probability.png',transparent=False)
plt.legend(loc='best')
plt.show()


# In[15]:


n, t = 100, 10
G1 = nx.erdos_renyi_graph(n,0.7)
G2 = nx.watts_strogatz_graph(n, 5, 0.7) 
G3 = nx.barabasi_albert_graph(n,5) 
GU = [G1, G2, G3]
GUname = ["Random Graph","WS Graph","BA Graph"]
infection_start_node = 0
probs = [i*0.05 for i in range(1,21)]
for G in GU:
    time_to_largest_infect = []
    for p in probs:
        infgrap = sirtmodel(G, p, t, infection_start_node)
        count = 0
        max_val = max(infgrap[1])
        for i in infgrap[1]:
            if i == max_val:
                time_max = count
            if i != 0:
                count+=1
        time_to_largest_infect.append(time_max)
    plt.title("Time to Largest Infection Against Probability")
    plt.plot(probs, time_to_largest_infect, label = GUname[GU.index(G)])
    plt.grid(True, linestyle='dotted')
plt.xlabel('Probability of Infection')
plt.ylabel('Time to Largest Infection')
plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_5_257389_Segun_Light_Jegede/q3/Time to Largest Infection Against Probability.png',transparent=False)
plt.legend(loc='best')
plt.show()


# In[ ]:




