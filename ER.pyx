# cython: language_level=3
# cython: initializedcheck = False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++
cimport cython
import numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from mc_lib.rndm cimport RndmWrapper
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from anytree import NodeMixin, RenderTree
from anytree import find_by_attr, PreOrderIter
from numpy.random.c_distributions cimport random_poisson
# import pickle
import networkx as nx
import queue
from networkx.drawing.nx_pydot import graphviz_layout
import errno

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

    
cdef extern from "time.h":
    long int time(int)
    

cdef extern from "math.h":
    double exp(double)
    
    
    
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef Py_ssize_t  random_choice(double[:] probs):
    '''
    Returns random number from 0 to n-1 according to probs
        Check if probs do not sum up to 1!
    '''
    
    # cdef int Py_ssize_t
    cdef double x = drand48()
    cdef double cum_probs = 0
    cdef Py_ssize_t n = 0
    while x > cum_probs:
        cum_probs += probs[n]
        n += 1
    n -= 1
    return n    
    
cdef class Event:
    cdef public:
        double x, y, time

    def __init__(self, double x, double y, double time):
        self.x = x
        self.y = y
        self.time = time

        
cdef class Events:
    cdef public:
        Py_ssize_t size, ptr
        double[::1] xs, ys, times

    def __init__(self):
        self.size = 0
        self.ptr = 0#pointer to the first empty cell
        
    def __next__(self):
        if self.ptr < self.size:
            self.ptr += 1
            return self.GetEvent(self.ptr-1)
        else:
            raise StopIteration

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # cdef void AddEvent(self, double x_, double y_, double time_):
    #     self.times[ self.ptr ] = time_
    #     self.xs[self.ptr] = xs_
    #     self.ys[self.ptr] = ys_
    #     self.ptr+=1
    #     self.size+=1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Event GetEvent(self, Py_ssize_t e_id):
        ev = Event(self.xs[ e_id ], self.ys[ e_id ], self.times[ e_id ])
        return(ev)
    
    cpdef void pprint(self):
        self.CreateEvents(100, 1, 1)
        for i in range(self.size):
            print(self.xs[i])
    

    cpdef void CreateEvents(self, Py_ssize_t iterations, double lamda, double L):
        self.times = np.cumsum(np.random.exponential(lamda, iterations))
        self.xs = np.random.uniform(0, L, iterations)
        self.ys = np.random.uniform(0, L, iterations)
        self.size = iterations


cdef class State:
    '''
    State class describe state of model:
    all individuals who have lived and died, number of alive individuals and those who died.
    This class provide you with information about indiviuals. Where were they born, who are their parents, and what are their genotype.
    In the end, everything pass away, so States contains death_time.
    '''
    cdef public:
        Py_ssize_t max_size, size, n_alive, n_dead
        Py_ssize_t[:] ids, p_ids, ids_alive, ids_to_die
        Py_ssize_t n_alleles # It should be inhereted by SLFVP.
        double[:] xs, ys, times, death_times
        Py_ssize_t[:, :] genotypes
        
        
    def __init__(self, double rho, double u0, Py_ssize_t t, Py_ssize_t n_alleles, pd, double l=1):
        self.size = 0
        self.max_size = int(1.05 * (rho + rho * pd * t) * l**2)
        self.ids = np.empty(self.max_size, dtype = int)
        self.p_ids = np.empty(self.max_size, dtype = int)
        self.xs = np.empty(self.max_size)
        self.ys = np.empty(self.max_size)
        self.times = np.empty(self.max_size)
        self.death_times = np.empty(self.max_size)
        self.genotypes = np.empty((self.max_size, n_alleles), dtype = int)
        self.n_alive = 0
        self.n_dead = 0
        self.n_alleles = n_alleles
        
        self.ids_alive = np.full(self.max_size, -1, dtype = int)
        self.ids_to_die = np.full(self.max_size, -1, dtype = int)
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    cdef void add(self, Py_ssize_t p_id, double time, double x, double y, Py_ssize_t[:] i_type):
        self.ids[self.size] = self.size
        self.p_ids[self.size] = p_id
        self.xs[self.size] = x
        self.ys[self.size] = y
        self.times[self.size] = time
        self.death_times[self.size] = -1
        for k in range(self.n_alleles):
            self.genotypes[self.size, k] = i_type[k]
        self.size += 1

        
    cpdef void generate(self, double L, double rho, Py_ssize_t n_alleles, double proport):
        cdef Py_ssize_t i
        N_points = np.random.poisson(rho*L**2)
        xs = np.random.uniform(0, L, N_points)
        ys = np.random.uniform(0, L, N_points)
        alleles = np.random.choice([0, 1], (N_points, n_alleles), p = [1 - proport, proport])
        for i in range(N_points):
            self.ids_alive[self.size] = self.size
            self.ids[self.size] = self.size
            self.p_ids[self.size] = -1
            self.xs[self.size] = xs[i]
            self.ys[self.size] = ys[i]
            self.times[self.size] = 0
            self.death_times[self.size] = -1
            for k in range(self.n_alleles):
                self.genotypes[self.size, k] = alleles[i,k]
            self.size += 1
            self.n_alive += 1
        
   
    
        
    cpdef pprint(self):
        for i in self.ids_alive:
            print(self.individuals[i])
            
            
            

cdef inline double tor2_distance(double x1, double x2, double y1, double y2,  double w=1, double h=1):
    #squared!!!
    return (min(abs(x1 - y1), w - abs(x1 - y1))**2 + min(abs(x2 - y2), h - abs(x2 - y2))**2)


cdef class ER:
    cdef public:
        State state
        Events events
        double L, rho, lamda, theta, alpha, time, u0, integral, beta
        double [:] probs
        double [:,:] points 
        Py_ssize_t n_alleles, n_epoch, died, born
        Py_ssize_t [:] ids_newborn
        RndmWrapper seed
        
    def __init__(self, double L=1, double lamda=1, double u0=0.4, double rho=1000, double theta=0.5,
                 double alpha=1, Py_ssize_t n_alleles=10, n_epoch = 1000):
        self.rho=rho
        self.L=L
        self.u0=u0
        self.theta=theta
        self.alpha=alpha
        self.lamda = lamda
        self.died = 0
        self.born = 0
        self.beta = 1
        self.state = State(rho, u0, n_epoch, n_alleles, L)
        self.n_epoch = n_epoch
        self.n_alleles = n_alleles
        self.time = 0
        self.integral = dblquad(lambda x, y: self.u(self.L/2, self.L/2, x,y), -self.L/2, self.L/2, -self.L/2,  self.L/2 )[0]
        self.state = State(rho, u0, n_epoch, n_alleles, self.integral, L)
        self.probs = np.empty(int(rho*L**2)*3 + 1)
        self.ids_newborn = np.empty(int(rho*L**2)*3 + 1, dtype = int)
        self.seed = RndmWrapper(seed=(123, 0))
        self.points = np.empty((int(rho*L**2)*10,3), dtype = float)
        self.events = Events()
        
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef Py_ssize_t choose_parent(self, double z1, double z2):
        '''
        This method chooses parent for new generation, and return its id
        '''
        
        cdef double sum_p = 0
        cdef Py_ssize_t i = 0, k = 0
        
        
        while self.state.ids_alive[i]!=-1:
            self.probs[i] = self.v(z1, z2,
                                    self.state.xs[self.state.ids_alive[i]], self.state.ys[self.state.ids_alive[i]])
            sum_p += self.probs[i]
            i += 1
            
        for k in range(self.state.n_alive): # Probability sums up to 1
            self.probs[k] /= sum_p
        p_id_index = random_choice(self.probs)
        
        
        # assert(p_id_index >= 0 and p_id_index < self.state.n_alive)
        return self.state.ids_alive[p_id_index]
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void extinction(self, double z1, double z2, double time):
        '''
        All shall fade once.
        This method go through all alive individuals, and wheighing their live. With probability, depending on distance from calamity every individual dies: its time is written in state, and index(position in ids_alive) is written in ids_to_die 
        '''
        cdef double x1, x2
        cdef Py_ssize_t i = 0, k=0
        self.died = 0
        while self.state.ids_alive[i] != -1:
            x1 = self.state.xs[self.state.ids_alive[i]] 
            x2 = self.state.ys[self.state.ids_alive[i]]
            if drand48() < self.u(z1, z2, x1, x2):  
                self.state.ids_to_die[k] = i
                k+=1
                self.state.death_times[self.state.ids_alive[i]] = time # Model remember the time of death
                self.died += 1
            i += 1
                
                
            
        self.state.ids_to_die[k] = -1
        
        # self.state.n_alive -= k

        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void recolonization(self, double z1, double z2, double time):
        '''
        Life has always been the end, while it is wisdom that shall be the means.
        This method uses rejection sampling and creates new individuals. Data about them is written in State, and id is written in ids_newborn 
        '''
        cdef Py_ssize_t n_points, generated, k,       
        cdef Py_ssize_t[:] i_type
        cdef double x1, x2
        
        p_id = self.choose_parent(z1, z2)
        i_type = self.state.genotypes[p_id]
        max_intensity = self.u(z1, z2, z1, z2)
        # n_points = self.died
        n_points = random_poisson(self.seed.rng, self.rho * self.integral)
        
        generated = 0
        while generated < n_points:
            x1 = self.L * drand48()
            x2 = self.L * drand48()

            if self.L**2 * self.u(z1, z2, x1, x2) >= max_intensity * drand48():#Rejection sampling for inhomogenous Poisson point process
                self.state.add(p_id, time, x1, x2, i_type)
                self.ids_newborn[generated] = self.state.size-1
                generated += 1
        self.ids_newborn[generated] = -1 #Indicates end
        # print(f'died = \t{self.died}\tborn = \t{n_points}')
        self.born = n_points
                
    

                
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef replace(self):
        '''
        Death is not the opposite of life, but a part of it.
        
        This method replaces Individuals, who died in extinction method to those, who were born in recolonization method. If the number of died prevales over the number of born, then ids of dead are erased, and gaps are shifted.
        '''
        cdef Py_ssize_t length1, length2, k=0, j=0, s=0, p = 0

        
        while self.state.ids_to_die[k]!= -1 and self.ids_newborn[j]!= -1: #while there are both who should pass away and who are to be born.
            self.state.ids_alive[self.state.ids_to_die[k]] = self.ids_newborn[j]
            k+=1
            j+=1
            
        
        
        while self.ids_newborn[j]!= -1: # If number of born is bigger the number of died, then newborn are written in the end.
            self.state.ids_alive[self.state.n_alive+s] = self.ids_newborn[j]
            j+=1
            s+=1
            
        
        
        #Если j < k то часть живых особей в ids_alive разделена -1. Таких пробелов должно быть k - j
        
        if self.ids_newborn[j]== -1 and self.state.ids_to_die[k]!= -1:
            while self.state.ids_to_die[k] + s < self.state.n_alive:   
                if self.state.ids_alive[self.state.ids_to_die[k] + s] == -1:
                    p += 1
                else:
                    self.state.ids_alive[self.state.ids_to_die[k] + s - p] = self.state.ids_alive[self.state.ids_to_die[k] + s]
                s += 1
                
                
        self.state.n_alive += self.born - self.died
        
        self.state.ids_alive[self.state.n_alive] = -1
        
        
        

        
    
    cpdef void propagate(self, double z1, double z2, double time):# parameters -- list [L, rho, u0, alpha, theta]
        self.extinction(z1, z2, time)
        self.recolonization(z1, z2, time)
        self.replace()
    
    
    cpdef void run(self):
        '''But eternity is far too cruel fate for you, Ei'''
        cdef Py_ssize_t i
        for i in range(self.events.size):
        # for i in trange(self.events.size):
            self.propagate(self.events.xs[i], self.events.ys[i], self.events.times[i]) 
            
            
    cpdef double calc_hetero(self, allele=0):
        cdef Py_ssize_t k=0, i
        cdef double prop=0
        
        for i in self.state.ids_alive:
            if i == -1:
                break
            if self.state.genotypes[i, allele] == 1:
                k += 1
        prop = k/self.state.n_alive
        
        return 2 * prop*(1-prop)
            
            
    cpdef double[:] run_with_het(self):
        '''But eternity is far too cruel fate for you, Ei'''
        cdef double[:] hets
        cdef Py_ssize_t i
        hets = np.empty(self.events.size)
        for i in range(self.events.size):
        # for i in trange(self.events.size):
            hets[i] = self.calc_hetero()
            self.propagate(self.events.xs[i], self.events.ys[i], self.events.times[i])
        return hets
        
        
        
    cpdef void initiate(self, double proport):
        self.events.CreateEvents(self.n_epoch, self.lamda, self.L)
        self.state.generate(self.L, self.rho, self.n_alleles, proport)
        
        
    @cython.cdivision(True)    
    cdef inline double v(self, double z1, double z2, double x1, double x2):
        return exp(- tor2_distance(z1, z2, x1, x2)/(2 * self.alpha**2 * self.theta**2))
    
    
    @cython.cdivision(True)
    cdef inline double u(self, double z1, double z2, double x1, double x2):
        return self.u0 * exp(- tor2_distance(z1, z2, x1, x2)/(2 * self.theta**2))
    
    
    @cython.cdivision(True)
    cdef inline double h(self, double z1, double z2, double x1, double x2):
        return exp(-tor2_distance(z1, z2, x1, x2)/self.beta**2)
    
    
    
    '''
    You who were born with original sin. Go forth and search for the long-buried truth, before all is lost beneath the waves. I will use the past to judje the future. --Neuvilette
    
    Next methods nor affect evolution of system, but analyse it. Here you may see mathod to calculate some genomic characteristics, make plots and build geneaologic trees.
    
    '''
    cpdef coalescense_time(self, Py_ssize_t id1, Py_ssize_t id2):
            cdef:
                double birth_time, RMCA_time
            
            birth_time = (self.state.times[id1] + self.state.times[id2])*0.5
            
            while id1 != id2 and id1 != -1 and id2!=-1:
                if (self.state.times[id1] > self.state.times[id2]):
                    id1 = self.state.p_ids[id1]
                else: 
                    id2 = self.state.p_ids[id2]
            if id1 == -1 or id2==-1:
                return -1
            RMCA_time = self.state.times[id1]
            return birth_time - RMCA_time
        
    def coalescense_time_hist(self):
        times = list()
            
        for i in tqdm(range(self.state.n_alive)):
            for j in range(i+1,self.state.n_alive):
                assert(self.state.ids_alive[i] != -1 and self.state.ids_alive[j] != -1 )
                t = self.coalescense_time(self.state.ids_alive[i], self.state.ids_alive[j])
                if t != -1:
                    times.append(t)
                
        print(f"Mean coalescence time = {np.mean(times)}")
        return plt.hist(times, bins = 100, density = True);
    
    
    cpdef mean_lifetime(self):
        cdef:
            list list_dead = list(self.state.ids_dead)
            Py_ssize_t i
            double mean_time = 0
            
        for i in tqdm(list_dead):
            mean_time += self.state.individuals[i].death_time - self.state.individuals[i].time
        mean_time /= self.state.n_dead
        return mean_time
    
    
    def lifetime(self):
        ltime = list()
        for i in tqdm(range(self.state.size)):
            if self.state.death_times[i] != -1:
                ltime.append(self.state.death_times[i] - self.state.times[i])
        print(f"Mean-time = {np.mean(ltime)}")
        return plt.hist(ltime, bins = 100, range=(0, 10), density = True);
        
        
        
    def build_lines(self, ids):
        ax = plt.figure().add_subplot(projection='3d')
        for i in ids:
            ind = i
            xs, ys, ts = list(), list(), list()
            while ind != -1:
                xs.append(self.state.xs[ind])
                ys.append(self.state.ys[ind])
                ts.append(self.state.times[ind])
                
                ind = self.state.p_ids[ind]
                
            ax.plot(xs, ys, ts)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('t')
        
        plt.show()

                
            
                
                
        
        
                
            
    def density(self, z1, z2):
        denom = 0
        thetas = np.zeros(self.n_alleles)
        for i in self.state.ids_alive:
            if i == -1:
                break
            denom += self.h(z1, z2,
                            self.state.xs[i], self.state.ys[i])
            for k in range(self.n_alleles):
                if self.state.genotypes[i, k] == 1:
                    thetas[k] += self.h(z1, z2,
                            self.state.xs[i], self.state.ys[i])
        thetas = thetas / denom

        return thetas


    def plt_SFS1(self, z1, z2):
        d = self.density(z1, z2)
        N = 100
        y = []
        for i in range(N):
            y.append((d<(i+1)/100).sum())
        plt.plot(y)


    def plt_SFS2(self, z11, z12, z21, z22):
        d1 = self.density(z11, z12)
        d2 = self.density(z21, z22)
        N = 100
        y = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                y[N-i-1,j]=(np.logical_and(d1<(i+1)/N, d2<(j+1)/N).sum())
        plt.imshow(y, extent=[0,1,0,1])
        
    
    def plot_with_alleles(self, allele=0, alpha=0.5, name=None):
        cdef list xs1, xs2, ys1, ys2
        xs1 = list()
        xs2 = list()
        ys1 = list()
        ys2 = list()
    

        for i in self.state.ids_alive:
            if i == -1:
                break
            if self.state.genotypes[i, allele] == 0:
                xs1.append(self.state.xs[i])
                ys1.append(self.state.ys[i])
            else:
                xs2.append(self.state.xs[i])
                ys2.append(self.state.ys[i])
        plt.scatter(xs1, ys1, alpha, c='b', label ='0 allele')
        plt.scatter(xs2, ys2, alpha, c='g', label = '1 allele')
        plt.legend()
        if name!= None:
            plt.savefig(name)
        plt.show();
        
        
    cpdef plot_alleles(self, alpha=0.5):
        cdef list xs1, xs2, ys1, ys2
        xs1 = list()
        xs2 = list()
        xs3 = list()
        xs4 = list()
        ys1 = list()
        ys2 = list()
        ys3 = list()
        ys4 = list()
    
        for i in list(self.state.ids_alive):
            if i == -1:
                break
            if self.state.genotypes[i, 0] == 0 and self.state.genotypes[i, 1] == 0 :
                xs1.append(self.state.xs[i])
                ys1.append(self.state.ys[i])
            elif self.state.genotypes[i, 0] == 0 and self.state.genotypes[i, 1] == 1:
                xs2.append(self.state.xs[i])
                ys2.append(self.state.ys[i])
            elif self.state.genotypes[i, 0] == 1 and self.state.genotypes[i, 1] == 0:
                xs3.append(self.state.xs[i])
                ys3.append(self.state.ys[i])
            else:
                xs4.append(self.state.xs[i])
                ys4.append(self.state.ys[i])
        plt.scatter(xs1, ys1, alpha, label = '00 allele')
        plt.scatter(xs2, ys2, alpha, label = '01 allele')
        plt.scatter(xs3, ys3, alpha, label = '10 allele')
        plt.scatter(xs4, ys4, alpha, label = '11 allele')
        plt.legend()
        plt.show();
        
        
            
        
    def build_tree(self, ids):
        adj = []
        finished = set()
        q = queue.Queue()
        for i in ids:
            if i not in finished:
                q.put(i)
                finished.add(i)
        while not q.empty():
            idx = q.get()
            id_p = self.state.p_ids[idx]
            adj.append([idx, id_p])
            finished.add(idx)
            if id_p != -1 and id_p not in finished:
                q.put(id_p)
        G = nx.from_edgelist(adj)
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", root = -1)
        nx.draw(G, pos)
                
            
            
        
        
        
        
        