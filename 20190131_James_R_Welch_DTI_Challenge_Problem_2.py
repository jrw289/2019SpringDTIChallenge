# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 23:13:48 2019

@author: Jake Welch
"""
import time

#This is a brute force method for solving this problem
#    to demonstrate how it can be accomplished by generating vectors
#The approach is to generate lists, with a probability of occurrence,
#    then generate a new list for each available move
#To prevent the list of lists from becoming too big, all identical
#    lists are combined between each generation step 
#Note that this was not optimized for computation as well as it could be.
#   If I were continuing to use this file, I would look at things such as
#   NumPy arrays to increase the computational speed

#This generates the initial track and car positions, as well as
#    a term for the probability of this configuration in the last bin
def generate_init(n: int, m: int):
    z = [0 for i in range(n + 1)] #all the positions, plus a prob. term
    for j in range(m):
        z[j] = 1
    z[-1] = 1 #probability of initial configuration is 1 at t = 0
    print('Initial Configuration: {}'.format(z))    
    return [z], m #Need m for denominator of averages later on

#This function takes a list of lists representing positions and
    #probabilities, then generates a list of lists with all the 
    #new positions after an additional move, with modified probabilities
def moves(l):
    a = [] #output list of lists
    b = len(l) #split at the list of lists level
    
    #For each list in l, find all possible moves available
    #Then a new list for each move, modifying the prob. accordingly
    for c in l:
        
        f = len(c) - 1 #skip the probability term
        h = [] #indices for the switches 
        
        #Find all the 
        for d in range(f):

            if (c[d % f] == 1) & (c[(d + 1) % f]==0):            

                h.append(d)
        
        scal = len(h)
        
        for x in h:
            
            temp = c.copy()
            temp[x % f] = 0 #Flip the position of 0 and 1 
            temp[(x + 1) % f] = 1
            temp[-1] = temp[-1]*(1/scal) #Since all equally likely, 
                                         #1/len(h) prob. of each 
                                         
            a.append(temp) #Add the list to the output list of lists
            
    return a

#The output of each moves operation has mutiple entries
#    for the same configuration, so here, identical distributions
#    and their probabilities are combined     
def combineProbs(q):
    r = [] 
    temp = q.copy() #Copy list to remove elements as they are summed
    
    while temp:
        
        temp_2 = temp[0]        
        temp.remove(temp_2)
        
        if len(temp)> 0:        
            b = [] #list of items to be removed            
            for a in temp:
                
                if temp_2[:-1] == a[:-1]: #If positions are the same       
                    temp_2[-1] = temp_2[-1] + a[-1] #add prob. to temp_2
                    b.append(a) #add the list to the removal
            
            for c in b:            
                temp.remove(c) #remove all flagged lists
                
        r.append(temp_2)
    
    return r

#Average of the n_cars car positions
#Labeled 0 to n_cars - 1
def ave(o):
    temp = o[:-1] #drop probability term 
    weighted = [i*j for i,j in enumerate(temp)]
    av = sum(weighted)/n_cars
    return av

#Standard deviation of the n_cars car positions             
def stDev(p):
    
    temp = p[:-1] #drop probability term 

    inds = [k for k, l in enumerate(temp) if l == 1] #indices of car pos.    
    weighted = [i*j for i,j in enumerate(temp)] #car positions 
    av = sum(weighted)/n_cars
    #print(weighted)
    
    rel_vals = [weighted[x] for x in inds] #here be cars 
    w2 = [(a - av)**2/n_cars for a in rel_vals] #variance of the car pos.
    #print(w2)
    stDev = sum(w2)**(0.5) #standard deviation 

    return(stDev)

#Probability of each vector
def prob(q):
    prob = q[-1]
    return prob

#Expectation value generator
#Inputs: variable list, probability list
def expVal(var_l,prob_l):
    exp_vec = [a*b for a,b in zip(var_l,prob_l)]
    #print(exp_vec)
    return sum(exp_vec)

#Standard deviation of the standard deviations 
#Inputs: variable list, probability list, variable expectation value    
def stDev_2(var_m,prob_m,ave):
    std_vec = [p*(x - ave)**2 for x,p in zip(var_m,prob_m)]
    return sum(std_vec)


################################################################

#First setup: T = 20, N = 10, M = 5

t = 20 #Number of time steps 
temp_4, n_cars = generate_init(10,5) #generate init. and n_cars 

start = time.time()

#Main loop for performing the simulation 
for z in range(t):
    temp_3 = moves(temp_4)
    temp_4 = combineProbs(temp_3)

end = time.time()
print('Main Loop Time: {} s'.format(end-start))                         
      
temp_4.sort()

#Checks that the probability isn't getting messed up
#Creates lists of averages, standard devs, and probabilities
#    for each position vector 
tot_prob_2 = 0
aves = []
stDevs = []
probs = []

start = time.time()
    
for j in temp_4:
    tot_prob_2 = tot_prob_2 + j[-1]   
    stDevs.append(stDev(j))
    aves.append(ave(j))
    probs.append(prob(j))

end = time.time()
print('Aggregation Loop Time: {} s'.format(end-start))   
    
start = time.time()
#Expectation value of averages for first case 
q1 = expVal(aves,probs)

#Standard deviation of averages for first case
q2 = stDev_2(aves,probs,q1)

#Expectation value of standard deviation for first case 
q3 = expVal(stDevs,probs)

#Standard deviation of standard deviations for first case
q4 = stDev_2(stDevs,probs,q3)

end = time.time()
print('Answer Calculation Time: {} s'.format(end-start))

####################################################################

#Second setup: T = 50, N = 25, M = 10
#Computational time was increasingly very nonlinearly, so I added
#    some subloops to split the operation into a set of batches
#This greatly decreased the computation time, which wasn't a problem
#    for the previous problem  


t_2 = 30 #Number of time steps
#Around 38 is when the time computation shoots up
#At 30, 22 to 25 s to complete and 13,400 elements in temp_7
#At 35, about 120 s (alt method takes )
#40 has not been attemped (takes 5 minutes with alt method)

temp_8, n_cars = generate_init(25,10) #generate init. and n_cars 

start = time.time()

#for z in range(t_2):
#    start = time.time()
#    xx = len(temp_8)
#    temp_7 = moves(temp_8)
#    temp_8 = combineProbs(temp_7)
#    end = time.time()
#    print('Starting size: {}\n Time: {} s\n'.format(xx,(end-start)))


for z in range(t_2):
    
    xx = len(temp_8)
    #start = time.time()
    
    if xx > 150:
        
        b = []
        batch_out = []
        not_end = True
        i = 0
        
        while not_end:
            
            if 150*(i+1) < xx:
                b.append(temp_8[150*i:150*(i+1)])
                i = i + 1
            else:
                b.append(temp_8[150*i:])
                not_end = False
        

        
        for c in b: 
            temp_in = moves(c)
            batch_out = batch_out + combineProbs(temp_in)
        
        temp_8 = combineProbs(batch_out)
        
    else: 
        temp_7 = moves(temp_8)
        temp_8 = combineProbs(temp_7)
    
    #end = time.time()    
    #print('Starting size: {}\n Time: {} s\n'.format(xx,(end-start)))

        
end = time.time()
print('Main Loop Time: {} s'.format(end-start))                         
      
temp_8.sort()

#Checks that the probability isn't getting messed up
#Creates lists of averages, standard devs, and probabilities
#    for each position vector 
tot_prob_4 = 0
aves_2 = []
stDevs_2 = []
probs_2 = []

start = time.time()
    
for j in temp_8:
    tot_prob_4 = tot_prob_4 + j[-1]   
    stDevs_2.append(stDev(j))
    aves_2.append(ave(j))
    probs_2.append(prob(j))

end = time.time()
print('Aggregation Loop Time: {} s'.format(end-start))   
    
start = time.time()
#Expectation value of averages for second case 
q5 = expVal(aves_2,probs_2)

#Standard deviation of averages for second case
q6 = stDev_2(aves_2,probs_2,q5)

#Expectation value of standard deviation for second case 
q7 = expVal(stDevs_2,probs_2)

#Standard deviation of standard deviations for second case
q8 = stDev_2(stDevs_2,probs_2,q7)

end = time.time()
print('Answer Calculation Time: {} s'.format(end-start))