import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.integrate as integr
from tqdm import tqdm
# from icecream import ic
from anytree import NodeMixin, RenderTree
from anytree import find_by_attr, PreOrderIter
import pickle

class Node(NodeMixin):
    def __init__(self, index=0, time=0, parent=None, children=None):
        self.id = index
        self.parent = parent
        self.time=time
        if children:  # set children only if given
             self.children = children
    def __repr__(self):
             return str(f'id:{self.id}, time:{self.time}')
    def __str__(self):
        return str(f'id:{self.id}, time:{self.time}')



class State:
    def __init__(self, time=0):
        self.time = 0
        self.individuals=None
        self.hist=None
        self.freeId = 1
        
        
    def create(self, individuals):
        Ids = np.arange(self.freeId, self.freeId+len(individuals))
        self.individuals = np.column_stack((Ids, individuals))
        self.hist = np.column_stack((Ids, individuals))
        self.freeId = self.freeId+len(individuals)
        
        
    def delete(self, ids):
        self.individuals = np.delete(self.individuals, ids, axis=0)
    
    
    def add(self, individuals):
        Ids = np.arange(self.freeId, self.freeId+len(individuals))
        self.individuals = np.append(self.individuals, np.column_stack((Ids, individuals)),axis=0)
        self.hist = np.append(self.hist, np.column_stack((Ids, individuals)),axis=0)
        self.freeId = self.freeId+len(individuals)
        
    
    def genealogy(self, ids ):
        
        
        table = self.hist[self.hist[:,0] == ids[0]]
        for idx in ids[1:]:
            table = np.row_stack((table, self.hist[self.hist[:,0] == idx]))
        allids = set(ids)
        newids = list(set(table[:,1]) - allids)
        allids = allids.union(set(newids))
        while len(newids)>0:
            for idx in newids:
                table = np.row_stack((table, self.hist[self.hist[:,0] == idx]))
            newids = list(set(table[:,1]) - allids)
            allids = allids.union(set(newids))
        return table
        
    
    
    
    def _build_coalesce_tree(self, data, parent=None, index=0):
    # data = [[id, parent_id, time]]
        root = None
        if parent is None:
            root = Node()
            parent = roo
        for x in data[data[:,1]==index]:
            node = Node(x[0],x[2],parent)
            self._build_coalesce_tree(data, node, x[0])
        return root
        
        
    def build_coalesce_tree(self, ids):
        data = self.genealogy(ids)
        return self._build_coalesce_tree(data)
            



class Model:
    def __init__(self, L=1, lamda=1, u0=0.4, rho=1000, theta=0.5, alpha=1, n_alleles=10):
        self.rho=rho
        self.L=L
        self.u0=u0
        self.theta=theta
        self.alpha=alpha
        self.lamda = lamda
        self.state = State()
        self.n_alleles = n_alleles
        self.dynamic=None
        
        
        
#--------------Initializating functions-------------------
        
        
    def generate_dynamic(self, n_epoch, time_init=0):
        '''Result is in (time, x, y)
        NOTE! MAYBE LAMBDA SHOULD BE RENORMALISED'''
        times = time_init + np.cumsum(np.random.exponential(self.lamda, n_epoch))
        xs = np.random.uniform(0, self.L, n_epoch)
        ys = np.random.uniform(0, self.L, n_epoch)
        self.dynamic = np.column_stack((xs, ys, times))
        
        
    def generate_initial_points(self):
        '''Fill state.individuals with points according to uniform poisson point process with density \\rho'''
        N_points = np.random.poisson(self.rho*self.L**2)
        xs = np.random.uniform(0, self.L, N_points)
        ys = np.random.uniform(0, self.L, N_points)
        return np.column_stack((xs, ys))
        
        
    def initiate(self, proport=0.5):
        N_points = np.random.poisson(self.rho*self.L**2)
        xs = np.random.uniform(0, self.L, N_points)
        ys = np.random.uniform(0, self.L, N_points)
        alleles = np.random.choice([0, 1], (N_points, self.n_alleles), p = [1 - proport, proport])
        pId = np.full(N_points, 0)
        times = np.full(N_points, 0)
        self.state.create(np.column_stack((pId, times, xs, ys ,alleles)))
        
    
    def choose_parent(self, z):
        probs = []
        for x in self.state.individuals:
            probs.append(self.v(z[0:2], x[3:5]))
        return self.state.individuals[np.random.choice(np.arange(len(self.state.individuals)), p = probs/np.sum(probs))]
    
    
    def choose_parent_type(self, z):
        probs = []
        for x in self.state.individuals:
            probs.append(self.v(z[0:2], x[3:5])) 
        return self.state.individuals[np.random.choice(np.arange(len(self.state.individuals)), p = probs/np.sum(probs))][5:]
    
    
    
#--------------Evolution functions--------------------------    
    def extinction(self, event):
        z, time = event[0:2], event[2]
        time = 0
        # time = event[2]
        ids = [] #ids to delete
        for i in range(len(self.state.individuals)):
            if np.random.uniform() < self.u(z, self.state.individuals[i,3:5]):
                ids.append(i)
                # print(f'{indicies=}')
        # print(f'survived {points[indicies].shape=}')
        self.state.delete(ids)
    
    
    def recolonization(self, event):
        z, time = event[0:2], event[2]
        parent = self.choose_parent(z)
        parentId = parent[0]
        parentType = parent[5:]
        intensity = lambda x, y: self.u(z, np.array([x,y]))
        max_intensity = intensity(z[0], z[1])
        total_intensity = integr.dblquad(intensity, 0, self.L, 0,  self.L )[0]
        # print(f"{total_intensity=}\n{max_intensity=}")
        n_points = np.random.poisson(self.rho * total_intensity) # Тут вроде total
        # print(f'{rho * total_intensity=}')
        # print(f'recolonized {n_points=}')
        points = []
        generated = 0
        while generated < n_points:
            x = np.random.uniform(0, self.L)
            y = np.random.uniform(0, self.L)

            if self.L**2 * intensity(x,y) >= np.random.uniform(0, max_intensity):
                points.append([x,y])
                generated += 1
        
        
        points = np.array(points,ndmin=2)
        points = np.column_stack(
            (
                np.full(n_points,parentId),
                np.full(n_points,time),
                points,
                np.full((n_points, self.n_alleles), parentType)
            )
        )
        self.state.add(points)
   

    
    
    
    def propagate(self, event):# parameters -- list [L, rho, u0, alpha, theta]
        # ic(event)
        # pass
        self.extinction(event)
        self.recolonization(event)
    
    
    def run(self):
        for event in tqdm(self.dynamic):
            self.propagate(event) 
    
#----------------Hat functions------------------------    
    def v(self, z, x):
        return np.exp(- np.linalg.norm(z-x)/(2 * self.alpha**2 * self.theta**2))
    
    
    def u(self, z, x):
        return self.u0 * np.exp(- np.linalg.norm(z-x)/(2 * self.theta**2))
    
    
    def h(self, z, x, beta = 1):
        return np.exp(-np.linalg.norm(z-x)/beta**2)
    
    
#---------------ANALYS FUNCTIONS----------------------- 
    def density(self, z, beta=1):
        points = self.state.individuals
        denom = 0
        thetas = np.zeros(self.n_alleles)
        for x in points:
            denom += self.h(z, x[3:5], beta)
            for k in range(self.n_alleles):
                if x[k+5] == 1:
                    thetas[k] += self.h(z, x[3:5], beta)
        thetas = thetas / denom

        return thetas


    def plt_SFS1(self, z, beta=1):
        d = self.density(z, beta=1)
        N = 100
        y = []
        for i in range(N):
            y.append((d<(i+1)/100).sum())
        plt.plot(y)


    def plt_SFS2(self, z1, z2,beta=1):
        d1 = self.density(z1, beta=1)
        d2 = self.density(z2, beta=1)
        N = 100
        y = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                y[i,j]=(np.logical_and(d1<(i+1)/N, d2<(j+1)/N).sum())
        plt.imshow(y, extent=[0,1,0,1])
        
        
    def plot_with_alleles(self, allele=0, alpha=0.5):
        points = self.state.individuals
        plt.scatter(points[points[:,5+allele]==0][:,3],points[points[:,5+allele]==0][:,4], alpha, label ='0 allele')
        plt.scatter(points[points[:,5+allele]==1][:,3],points[points[:,5+allele]==1][:,4], alpha, label = '1 allele')
        plt.legend()
        plt.show();
#--------------------Auxilary
    
    def save(self):
        return {'rho' :self.rho, 
                'L' : self.L,
                'u0' :self.u0,
                'theta': self.theta,
                'alpha':self.alpha,
                'lamda':self.lamda,
                'n_alleles':self.n_alleles,
                'dynamic':np.copy(self.dynamic),
                'individuals':np.copy(self.state.individuals),
                'hist': np.copy(self.state.hist)}

    
    def load(self, data):
        self.rho = data['rho'] 
        self.L = data['L']
        self.u0 = data['u0']
        self.theta = data['theta']
        self.alpha = data['alpha']
        self.lamda = data['lamda']
        self.n_alleles = data['n_alleles']
        self.dynamic = data['dynamic'],
        self.state.individuals = data['individuals']
        self.state.hist = data['hist']