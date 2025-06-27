#!/usr/bin/env python
# coding: utf-8

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


# In[2]:


import numpy
import pylab
import random
import matplotlib.pyplot as plt

def rw(n,ap,an,bp,bn):
    x = [0]
    y = [0]
    for i in range(1, n+1):
        #Plotting the walk on the graph at every point of the iteration
        pylab.ylim(bn-1,bp+1) 
        pylab.xlim(an-1,ap+1)
        pylab.title("Random Walk with ($n = " + str(i) + "$ steps)")
        pylab.plot(x, y, "-r", marker='H',color='b')
        pylab.plot(x[0], y[0], "-r", marker='H',color='c')
        pylab.plot(x[i-1], y[i-1], "-r", marker='H',color='r')
        pylab.grid(True, linestyle='dotted')
        z1=nmes(i,n)
        pylab.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/rw/Random_Walk_at_step=a'+str(z1)+'{}.png'.format(i),transparent=False)
        a=x[i-1]
        b=y[i-1]
        #Stating a condition to break out of the loop (plot) when the maximum stated number of step is reached
        if i==n:
            break
        #Stating conditions to keep the walk within a particular frame
        else:
            if a==an and b==bp:
                val=random.choice([1,4])
            elif a==an and b==bn:
                val=random.choice([1,3])
            elif a==ap and b==bp:
                val=random.choice([2,4])
            elif a==ap and b==bn:
                val=random.choice([2,3])
            elif a==ap and b<bp and b>bn:
                val=random.choice([2,3,4])
            elif a==an and b<bp and b>bn:
                val=random.choice([1,3,4])
            elif b==bp and a<ap and a>an:
                val=random.choice([4,1,2])
            elif b==bn and a<ap and a>an:
                val=random.choice([3,1,2])
            else:
                val = random.randint(1, 4)
            #Varying conditions for walks; left or right step on a particular point
            if val == 1:
                c = a + 1
                d = b
            elif val == 2:
                c = a - 1
                d = b
            elif val == 3:
                c = a
                d = b + 1
            else:
                c = a
                d = b - 1
            #Appending new coordinates at every iteration
            x.append(c)
            y.append(d)


# In[3]:


#QUESTION 1 PROGRAM TEST: simulating a random walk of an agent on a square lattice with 500 walks
import imageio 
import os
n=500
rw(n,8,-6,8,-6)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/rw/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/Random_walk"+str(n)+".movie.gif", images,duration=1.2)


# In[2]:


import numpy as np
import pylab, random, math
import matplotlib.pyplot as plt
import seaborn as sns

def pprw(n,ap,an,bp,bn):
    x = [0]
    y = [0]
    for i in range(1, n+1):
        #Plotting the walk on the graph at every point of the iteration
        pylab.ylim(bn-1,bp+1) 
        pylab.xlim(an-1,ap+1)
        pylab.title("Pearson Random Walk with ($n = " + str(i) + "$ steps)")
        pylab.plot(x, y, "-r", marker='o',color='b') 
        pylab.plot(x[0], y[0], "-r", marker='o',color='c')
        pylab.plot(x[i-1], y[i-1], "-r", marker='o',color='r')
        pylab.grid(True, linestyle='dotted')
        z1=nmes(i,n)
        pylab.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/prw/Pearson_Random_Walk_at_step=a'+str(z1)+'{}.png'.format(i),transparent=False)
        a=x[i-1]
        b=y[i-1]
        #Stating a condition to break out of the loop (plot) when the maximum stated number of step is reached
        if i==n:
            break
        else:
            #Generating a random angle for next point on a axis
            ainc = math.sin(np.random.uniform(0,math.pi/2))
            #Estimating the addition length for the next point on the other axis to make a walk of length of one 
            aphy = math.sqrt(1-(ainc**2))
            #Random number to determine which quadrant direction of the walk will be next
            val = random.randint(1, 4)
            val2 = random.randint(1, 2)
            #Varying conditions for walks; left or right step on a particular point
            if val == 1:
                c = a + ainc
                if val2 == 1:
                    d = b + aphy
                else:
                    d = b - aphy
                #conditions to force the walk within a particular coordinate
                if c > ap:
                    c = a - ainc
                if d < an:
                    d = b + aphy
                elif d > ap:
                    d = b - aphy
                else:
                    d = d 
            elif val == 2:
                c = a - ainc
                if val2 == 1:
                    d = b - aphy
                else:
                    d = b + aphy
                #conditions to force the walk within a particular coordinate
                if c < an:
                    c = a + ainc
                if d < an:
                    d = b + aphy
                elif d > ap:
                    d = b - aphy
                else:
                    d = d 
            elif val == 3:
                d = b + ainc
                if val2 == 1:
                    c = a + aphy
                else:
                    c = a - aphy
                #conditions to force the walk within a particular coordinate
                if d > ap:
                    d = b - ainc
                if c < an:
                    c = a + aphy
                elif c > ap:
                    c = a - aphy
                else:
                    c = c
            else:
                d = b - ainc
                if val2 == 1:
                    c = a - aphy
                else:
                    c = a + aphy
                #conditions to force the walk within a particular coordinate
                if d < an:
                    d = b + ainc
                if c < an:
                    c = a + aphy
                elif c > ap:
                    c = a - aphy
                else:
                    c = c
            #Appending new coordinates at every iteration 
            x.append(c)
            y.append(d)
    return (x,y)        

#The Histogram Plot of Random Walk on (x > 0)
def histpprwx(xa,yb):
    xan = []
    xybn = []
    n = len(xa)
    for j in range(n):
        if xa[j] > 0:
            xan.append(xa[j])  
    anres = len(xan)/n
    plt.title("Histogram of the Pearson Random Walk (x > 0) and ($n = " + str(n) + "$ steps)")
    plt.grid(True, linestyle='dotted')
    sns.distplot(xan, bins = int(180/15),kde_kws={"color": "r"},hist_kws={"color": "b"},axlabel='Random Walk on x>0')
    plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/Histogram_x_greater_0_Pearson_Random_Walk_at_step={}.png'.format(n),transparent=False)
    return anres

#The Histogram Plot of Random Walk on (x > 0 and y > 0)
def histpprwxy(xa,yb):
    xan = []
    xybn = []
    n = len(xa)
    for j in range(n):
        if xa[j] > 0 and yb[j] > 0:
            xybn.append(xa[j])
    bnres = len(xybn)/n
    plt.title("Histogram of the Pearson Random Walk (x,y > 0) and ($n = " + str(n) + "$ steps)")
    sns.distplot(xybn, bins = int(180/15),kde_kws={"color": "r"},hist_kws={"color": "b"},axlabel='Random Walk on x>0 and y>0')
    plt.grid(True, linestyle='dotted')
    plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/Histogram_xy_greater_0_Pearson_Random_Walk_at_step={}.png'.format(n),transparent=False)
    return bnres


# In[3]:


#QUESTION 2 PROGRAM TEST: simulating Pearson’s random walk in the plane, where the steps have constant length a = 1 and uniformly distributed random angles
import imageio 
import os
n = 500
cord = pprw(n,8,-6,8,-6)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/prw/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/Pearson_Random_walk"+str(n)+".movie.gif", images,duration=1.2)
f = open("Pearson_Random_Walk_File.txt", "a+")
f.write("Simulation %d\r\n" % (1))
f.write("\n\n")


# In[4]:


#Plotting normalized histograms (i.e. the PDFs) of AN - the fraction of time steps when the walker is in right half plane.
atn = histpprwx(cord[0],cord[1])
print('The probability of time the walker is in the x > 0 plane is', atn)
#Wriing out the fraction of time steps when the walker is in right half plane.
f.write("The probability of time the walker is in the x > 0 plane is %d\r\n" % (atn))
f.write("\n\n")


# In[5]:


#Plotting normalized histograms (i.e. the PDFs) of BN - the fraction of time steps when the walker is in the first quadrant.
btn = histpprwxy(cord[0],cord[1])
print('The probability of time the walker is in the x > 0 and y > 0 plane is', btn)
#Wriing out the fraction of time steps when the walker is in the first quadrant.
f.write("The probability of time the walker is in the x > 0 and y > 0 plane is %d\r\n" % (btn))
f.write("\n\n")


# In[6]:


#SECOND SIMULATION
#QUESTION 2 PROGRAM TEST: simulating Pearson’s random walk in the plane, where the steps have constant length a = 1 and uniformly distributed random angles
import imageio 
import os
n = 500
cord = pprw(n,8,-6,8,-6)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/prw/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/Pearson_Random_walk"+str(n)+".movie.gif", images,duration=1.2)
f = open("Pearson_Random_Walk_File.txt", "a+")
f.write("Simulation %d\r\n" % (2))
f.write("\n\n")


# In[7]:


#Plotting normalized histograms (i.e. the PDFs) of AN - the fraction of time steps when the walker is in right half plane.
atn = histpprwx(cord[0],cord[1])
print('The probability of time the walker is in the x > 0 plane is', atn)
#Wriing out the fraction of time steps when the walker is in right half plane.
f.write("The probability of time the walker is in the x > 0 plane is %d\r\n" % (atn))
f.write("\n\n")


# In[8]:


#Plotting normalized histograms (i.e. the PDFs) of BN - the fraction of time steps when the walker is in the first quadrant.
btn = histpprwxy(cord[0],cord[1])
print('The probability of time the walker is in the x > 0 and y > 0 plane is', btn)
#Wriing out the fraction of time steps when the walker is in the first quadrant.
f.write("The probability of time the walker is in the x > 0 and y > 0 plane is %d\r\n" % (btn))
f.write("\n\n")


# In[9]:


#THIRD SIMULATION
#QUESTION 2 PROGRAM TEST: simulating Pearson’s random walk in the plane, where the steps have constant length a = 1 and uniformly distributed random angles
import imageio 
import os
n = 500
cord = pprw(n,8,-6,8,-6)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/prw/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/Pearson_Random_walk"+str(n)+".movie.gif", images,duration=1.2)
f = open("Pearson_Random_Walk_File.txt", "a+")
f.write("Simulation %d\r\n" % (3))
f.write("\n\n")


# In[10]:


#Plotting normalized histograms (i.e. the PDFs) of AN - the fraction of time steps when the walker is in right half plane.
atn = histpprwx(cord[0],cord[1])
print('The probability of time the walker is in the x > 0 plane is', atn)
#Wriing out the fraction of time steps when the walker is in right half plane.
f.write("The probability of time the walker is in the x > 0 plane is %d\r\n" % (atn))
f.write("\n\n")


# In[11]:


#Plotting normalized histograms (i.e. the PDFs) of BN - the fraction of time steps when the walker is in the first quadrant.
btn = histpprwxy(cord[0],cord[1])
print('The probability of time the walker is in the x > 0 and y > 0 plane is', btn)
#Wriing out the fraction of time steps when the walker is in the first quadrant.
f.write("The probability of time the walker is in the x > 0 and y > 0 plane is %d\r\n" % (btn))
f.write("\n\n")


# In[12]:


import matplotlib.pyplot as plt
import networkx as nx 
import json
import random
#Writing a function to generate a random walk on a given graph
def graphrw(G, n, start_node):
    pos=nx.spring_layout(G)
    randnode = start_node
    nodelist = G.nodes()
    #Creatiing a dictionary to hold the hitting times of a node with the keys as the graph node
    nodecount = dict.fromkeys(nodelist,0 )
    #Creating a loop for every step taken during the random walk
    for i in range(n):
        #plotting and saving the graph at the ith walk
        plt.title("Random Walk on Graph ($n = " + str(i+1) + "$ steps)")
        nx.draw_networkx(G, pos=pos, node_color="blue",font_color="yellow",edge_color="grey")
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[randnode], node_color="red")
        z1=nmes(i+1,n)
        plt.savefig('/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/grw/Graph_Random_Walk_at_step=a'+str(z1)+'{}.png'.format(i+1),transparent=False)
        #The present selected node point on the ith walk
        next_node = randnode
        #Empty list for the neigbours of the chosen node and loop for appending each neighbour
        spec_a = []
        for node in G.neighbors(next_node):
            spec_a.append(node)
        #Choosing a node at random from the neighbour of the current node - the present point in the walk
        randnode=random.choice(spec_a)
        #Empty list for unselected nodes and loop for appending each node
        other = []
        for node in G.nodes():
            if node != randnode:
                other.append(node)
        #Condition which increease the value of a node choosen on the ith walk
        if randnode in nodecount.keys():
            nodecount[randnode]+=1
    #Converting the keys of the hitting times dictionary to string and appending the word "node"
    nodecount={'node '+str(k): v for k, v in nodecount.items()}
    #Writing the hitting times of each nodes in a file
    f = open("Hitting_Times_On_Node.txt", "a+")
    f.write("Average hitting times on each node is presented in the dictionary as nodes:hitting_times and starting node is %d\r\n" % (start_node))
    f.write(str(nodecount))
    f.write("\n\n\n")
    print("Average hitting times on each node is presented in the dictionary as nodes:hitting_times and starting node is", start_node)
    print(nodecount)
    return


# In[13]:


#QUESTION 3 PROGRAM TEST: simulating a random walk on a random graph
import imageio 
import os
G = nx.erdos_renyi_graph(20,0.5) 
n = 100
start_node = 0
graphrw(G, n, start_node)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/grw/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/Graph_Random_walk_with_steps_"+str(n)+".movie.gif", images,duration=1.2)


# In[14]:


#QUESTION 3 PROGRAM TEST: simulating a random walk on a Watt Strogatz graph
import imageio 
import os
G = nx.watts_strogatz_graph(20,5,0.6,seed=None)
n = 100
start_node = 3
graphrw(G, n, start_node)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/grw/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/Watt_Strogatz_and_Graph_Random_walk"+str(n)+".movie.gif", images,duration=1.2)


# In[15]:


#QUESTION 3 PROGRAM TEST: simulating a random walk on a Barabasi Albert graph
import imageio 
import os
G = nx.barabasi_albert_graph(20,4,seed=None)
n = 100
start_node = 5
graphrw(G, n, start_node)
png_dir = '/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/grw/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave("/Users/LIGHT/Desktop/Light Work/Diffusion GitLab/diffusion-processes/List_4_257389_Segun_Light_Jegede/grw/Barabasi_Albert_Graph_with_random_walk"+str(n)+".movie.gif", images,duration=1.2)


# In[ ]:




