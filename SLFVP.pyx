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
# from anytree import NodeMixin, RenderTree
# from anytree import find_by_attr, PreOrderIter
from numpy.random.c_distributions cimport random_poisson
# import pickle

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
    '''Returns random number from 0 to n-1 according to probs
        Check if probs do not sum up to 1!'''
    
    cdef int Py_ssize_t
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



cdef class Individual:
    cdef public:
        Py_ssize_t i_id, p_id
        Py_ssize_t[:] i_type
        double x, y, time, death_time
        
    def __init__(self, Py_ssize_t p_id, Py_ssize_t i_id, double time, double x, double y, Py_ssize_t[:] i_type, double death_time=-1):
        self.p_id = p_id
        self.i_id = i_id
        self.time = time
        self.x = x
        self.y = y
        self.i_type = i_type
        self.death_time = death_time
        
    def __str__(self):
        return f'p_id:{self.p_id}, i_id:{self.i_id}, x:{self.x},y:{self.y}'


cdef class State:
    cdef public:
        Py_ssize_t max_size, size, n_alive, n_dead
        Py_ssize_t[:] ids, p_ids, alive_ids
        int n_alleles
        double[:] xs, ys, times, death_times
        Py_ssize_t[:, :] genotypes
        
        
    def __init__(self, rho, u0, t, n_alleles, l=1):
        self.size = 0
        self.max_size = 2 * (rho+ u0 * rho t) * l**2
        self.ids = np.array((max_size))
        self.p_ids = np.array((max_size))
        self.xs = np.array((max_size))
        self.ys = np.array((max_size))
        self.genotypes = np.array((max_size, n_alleles))
        self.n_alive = 0
        self.n_dead = 0
        self.n_alleles = n_alleles
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void add(self, Py_ssize_t p_id, double time, double x, double y, Py_ssize_t[:] i_type):
        
        # self.individuals[self.size] = Individual(p_id, self.size, time, x, y, i_type)
        
        self.alive_ids[n_alive] = self.size
        self.ids[self.size] = size
        self.p_ids[self.size] = p_id
        self.xs[self.size] = x
        self.ys[self.size] = y
        self.times[self.size] = time
        for k in self.n_alleles:
            self.genotypes[self.size, k] = i_type[k]
        self.size += 1
        self.n_alive += 1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void remove(self, Py_ssize_t i_id, death_time):
        self.n_alive -= 1
        self.n_dead += 1
        self.individuals[i_id].death_time = death_time
        self.ids_alive.remove(i_id)
        self.ids_dead.add(i_id)
        
        
    cpdef void generate(self, double L, double rho, Py_ssize_t n_alleles, double proport):
        cdef long i
        N_points = np.random.poisson(rho*L**2)
        xs = np.random.uniform(0, L, N_points)
        ys = np.random.uniform(0, L, N_points)
        alleles = np.random.choice([0, 1], (N_points, n_alleles), p = [1 - proport, proport])
        for i in range(N_points):
            self.add(
                p_id = 0,
                time = 0,
                x = xs[i],
                y = ys[i],
                i_type = alleles[i]
            )
        
   
    
        
    cpdef pprint(self):
        for i in self.ids_alive:
            print(self.individuals[i])
            
            
            

cdef inline double tor2_distance(double x1, double x2, double y1, double y2,  double w=1, double h=1):
    #squared!!!
    return (min(abs(x1 - y1), w - abs(x1 - y1))**2 + min(abs(x2 - y2), h - abs(x2 - y2))**2)


cdef class SLFVP:
    cdef public:
        RndmWrapper seed
        double L, rho, lamda, theta, alpha, time, u0, integral, beta
        Py_ssize_t n_alleles
        double[:] probs
        State state
        Events events
        double [:,:] points 
        
    def __init__(self, double L=1, double lamda=1, double u0=0.4, double rho=1000, double theta=0.5, double alpha=1, Py_ssize_t n_alleles=10):
        self.rho=rho
        self.L=L
        self.u0=u0
        self.theta=theta
        self.alpha=alpha
        self.lamda = lamda
        self.beta = 1
        self.state = State()
        self.n_alleles = n_alleles
        self.time = 0
        self.integral = dblquad(lambda x, y: self.u(self.L/2, self.L/2, x,y), 0, self.L, 0,  self.L )[0]
        self.probs = np.empty(int(rho*L**2)*3 + 1)
        self.seed = RndmWrapper(seed=(123, 0))
        self.points = np.empty((int(rho*L**2)*10,3), dtype = float)
        self.events = Events()
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef Individual choose_parent(self, double z1, double z2):
        
        cdef double sum_p = 0
        cdef Py_ssize_t i, p_id
        cdef list list_alive = list(self.state.ids_alive)
        for i in range(self.state.n_alive):
            self.probs[i] = self.v(z1, z2,
                                    self.state.individuals[list_alive[i]].x, self.state.individuals[list_alive[i]].y)
            sum_p += self.probs[i]
        for i in range(self.state.n_alive):
            self.probs[i] /= sum_p
        p_id = random_choice(self.probs)
        return self.state.individuals[list_alive[p_id]]
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void extinction(self, double z1, double z2, double time):
        cdef double x1, x2
        cdef Py_ssize_t i
        cdef list ids
        cdef list list_alive = list(self.state.ids_alive)
        for i in list_alive:
            x1 = self.state.individuals[i].x
            x2 = self.state.individuals[i].y
            if drand48() < self.u(z1, z2, x1, x2):
                self.state.remove(i, time)
                

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void recolonization(self, double z1, double z2, double time):
        cdef Py_ssize_t n_points, generated, k,
        cdef double x1, x2
        cdef Individual parent
        cdef Py_ssize_t[:] i_type
        parent = self.choose_parent(z1, z2)
        p_id = parent.i_id
        i_type = np.copy(parent.i_type)
        # for k in range(self.n_alleles):
            # i_type[k] = 
        max_intensity = self.u(z1, z2, z1, z2)
        # print(f"{total_intensity=}\n{max_intensity=}")
        n_points = random_poisson(self.seed.rng, self.rho * self.integral) # Тут вроде total
        # print(f'{rho * total_intensity=}')
        # print(f'recolonized {n_points=}')
        generated = 0
        while generated < n_points:
            x1 = self.L * drand48()
            x2 = self.L * drand48()

            if self.L**2 * self.u(z1, z2, x1, x2) >= max_intensity * drand48():
                self.state.add(p_id, time, x1, x2, i_type)
                generated += 1
                
                
    cpdef void propagate(self, double z1, double z2, double time):# parameters -- list [L, rho, u0, alpha, theta]
        self.extinction(z1, z2, time)
        self.recolonization(z1, z2, time)
    
    
    cpdef void run(self):
        cdef Py_ssize_t i
        for i in trange(self.events.size):
            self.propagate(self.events.xs[i], self.events.ys[i], self.events.times[i]) 
        
        
        
    cpdef void initiate(self, Py_ssize_t iterations, double proport):
        self.events.CreateEvents(iterations, self.lamda, self.L)
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
    
    
    
    cpdef coalescense_time(self, Py_ssize_t id1, Py_ssize_t id2):
            cdef:
                double birth_time, RMCA_time
            
            birth_time = (self.state.individuals[id1].time + self.state.individuals[id2].time)*0.5
            
            while id1 != id2 and id1 != 0 and id2!=0:
                if (self.state.individuals[id1].time > self.state.individuals[id2].time):
                    id1 = self.state.individuals[id1].p_id
                else: 
                    id2 = self.state.individuals[id2].p_id
            RMCA_time = self.state.individuals[id1].time
            
            return birth_time - RMCA_time
        
    cpdef mean_coalescense_time(self):
        cdef:
            list list_alive = list(self.state.ids_alive)
            Py_ssize_t i, j
            double mean_time = 0
            
        for i in tqdm(list_alive):
            for j in list_alive:
                if i<=j:
                    continue
                mean_time += self.coalescense_time(i, j)
        mean_time /= self.state.n_alive * (self.state.n_alive-1) * 0.5
        return mean_time
    
    
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
        list_dead = list(self.state.ids_dead)
        ltime = list()
        for i in tqdm(list_dead):
            ltime.append(self.state.individuals[i].death_time - self.state.individuals[i].time)
            
        return plt.hist(ltime)
        
        
        
        
        
                
            
    #-------------
    def density(self, z1, z2):
        denom = 0
        thetas = np.zeros(self.n_alleles)
        list_alive = list(self.state.ids_alive)
        for i in range(self.state.n_alive):
            denom += self.h(z1, z2,
                            self.state.individuals[list_alive[i]].x, self.state.individuals[list_alive[i]].y)
            for k in range(self.n_alleles):
                if self.state.individuals[list_alive[i]].i_type[k] == 1:
                    thetas[k] += self.h(z1, z2,
                            self.state.individuals[list_alive[i]].x, self.state.individuals[list_alive[i]].y)
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
                y[i,j]=(np.logical_and(d1<(i+1)/N, d2<(j+1)/N).sum())
        plt.imshow(y, extent=[0,1,0,1])
        
    
    def plot_with_alleles(self, allele=0, alpha=0.5):
        cdef list xs1, xs2, ys1, ys2
        xs1 = list()
        xs2 = list()
        ys1 = list()
        ys2 = list()
    
        list_alive = list(self.state.ids_alive)
        for i in range(self.state.n_alive):
            if self.state.individuals[list_alive[i]].i_type[allele] == 0:
                xs1.append(self.state.individuals[list_alive[i]].x)
                ys1.append(self.state.individuals[list_alive[i]].y)
            else:
                xs2.append(self.state.individuals[list_alive[i]].x)
                ys2.append(self.state.individuals[list_alive[i]].y)
        plt.scatter(xs1, ys1, alpha, label ='0 allele')
        plt.scatter(xs2, ys2, alpha, label = '1 allele')
        plt.legend()
        plt.show();
        
        
    def plot_alleles(self, alpha=0.5):
        cdef list xs1, xs2, ys1, ys2
        xs1 = list()
        xs2 = list()
        xs3 = list()
        xs4 = list()
        ys1 = list()
        ys2 = list()
        ys3 = list()
        ys4 = list()
    
        list_alive = list(self.state.ids_alive)
        for i in range(self.state.n_alive):
            if self.state.individuals[list_alive[i]].i_type[0] == 0 and self.state.individuals[list_alive[i]].i_type[1] == 0 :
                xs1.append(self.state.individuals[list_alive[i]].x)
                ys1.append(self.state.individuals[list_alive[i]].y)
            elif self.state.individuals[list_alive[i]].i_type[0] == 0 and self.state.individuals[list_alive[i]].i_type[1] == 1:
                xs2.append(self.state.individuals[list_alive[i]].x)
                ys2.append(self.state.individuals[list_alive[i]].y)
            elif self.state.individuals[list_alive[i]].i_type[0] == 1 and self.state.individuals[list_alive[i]].i_type[1] == 0:
                xs3.append(self.state.individuals[list_alive[i]].x)
                ys3.append(self.state.individuals[list_alive[i]].y)
            else:
                xs4.append(self.state.individuals[list_alive[i]].x)
                ys4.append(self.state.individuals[list_alive[i]].y)
        plt.scatter(xs1, ys1, alpha, label ='00 allele')
        plt.scatter(xs2, ys2, alpha, label = '01 allele')
        plt.scatter(xs3, ys3, alpha, label = '10 allele')
        plt.scatter(xs4, ys4, alpha, label = '11 allele')
        plt.legend()
        plt.show();
        
        
        
        
        
        