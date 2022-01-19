import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.spatial.distance

class Simplex():
    
    def __init__(self, vertices, metric, inds):
        
        self.connected_simplices = []
        self.inds = inds
        self.vertices = vertices
        self.metric = metric
        V_0 = np.array([self.vertices[0, :] for i in range(self.vertices.shape[0] -1)])
        A = 2 * (V_0 - self.vertices[1:, :])
        zero = np.zeros((self.vertices.shape[1], 1))
        V_0n = self.metric(V_0[0, :], zero) ** 2 
        b = np.array([V_0n - self.metric(self.vertices[i+1, :], zero) ** 2 for i in range(self.vertices.shape[0] - 1)])
        try:
            centre = np.linalg.solve(A, b)
        except:
            print(self.inds)
        self.hyper_sphere = HyperSphere(centre, metric(centre, vertices[0, :]), self.metric) #O(d^3) :(
        self.facets = [frozenset([j for j in self.inds if j != i]) for i in self.inds]
    
    def __eq__(self, smp): # simplices are equal if they have the same vertices
        
        shared = [True for i in self.inds if i in smp.inds]
        return len(shared) == self.vertices.shape[1] + 1
    
    
    def plot_2D(self):
        
        plt.plot([self.vertices[2, 0], self.vertices[0, 0]], [self.vertices[2, 1], self.vertices[0, 1]])
        plt.plot([self.vertices[0, 0], self.vertices[1, 0]], [self.vertices[0, 1], self.vertices[1, 1]])
        plt.plot([self.vertices[1, 0], self.vertices[2, 0]], [self.vertices[1, 1], self.vertices[2, 1]])
        
    def getNode(self):
        try:
            return self.node
        except:
            return self.create_Graph()
        
    def added_simplex(self, num):
        
        num_vertices_lessthan = sum([1 for i in self.inds if i < num])
        
        return num_vertices_lessthan
        
    def create_Graph(self):
        if len(self.connected_simplices) == 0:
            raise Exception("Connected simplices have not been computed")
            
        self.node = Node(self.hyper_sphere.centre, [])
        self.node.Nodes = [smp.getNode() for smp in self.connected_simplices]
        
        return self.node
        
    def __str__(self):
        
        return str(self.inds)
    
    def __repr__(self):
        
        return str(self.inds)
        
class HyperSphere():
    
    def __init__(self, centre, radius, metric):
        
        self.centre = centre
        self.radius = radius
        self.metric = metric
        
    def contains(self, point):

        return self.metric(self.centre, point) < self.radius - 1e-10

class Node():
    
    def __init__(self, position, nodes):
        self.position = position
        self.Nodes = nodes
        
    def add_Node(self, Node):
        self.Nodes.append(Node)
        
    def plot_2D(self):
        for i in range(len(self.Nodes)):
            plt.plot([self.position[0], self.Nodes[i].position[0]], [self.position[1], self.Nodes[i].position[1]])
            
    def __repr__(self):
        
        return str((self.position, len(self.Nodes)))
    
    def __str__(self):
        
        return str(self.position) + " " + str(len(self.Nodes))
    
def bounding_simplex(points, metric):
    
    zero = np.zeros((points.shape[1],))
    dmins = [np.min(points[:, i]) - 0.1 for i in range(points.shape[1])]
    dmaxs = [np.max(points[:, i]) + 0.1 for i in range(points.shape[1])]
    
    centre = [(m+M)/2 for (m, M) in zip(dmins, dmaxs)]
    tcentre = []
    vertices = [[dmins[0]], [dmaxs[0]]]
    
    for i in range(1,points.shape[1]):
        dmin = dmins[i]
        dmax = dmaxs[i]
        tcentre.append(centre[i-1])
        tvert = [metric(np.array(vertices[j]) - np.array(tcentre), np.zeros((i,))) for j in range(len(vertices))]
        vertices = [((np.array(vertices[i]) - np.array(tcentre)) * (1 + (dmax - dmin)/tvert[i]) + np.array(tcentre)).tolist() for i in range(len(tvert))]
        vertices = [j for j in vertices if j.append(dmin) is None]
        vertices.append(tcentre.copy())
        vertices[-1].append(2 * dmax - dmin)
    
    vertices = np.array(vertices) * 1.1
    return vertices

def bowyer_watson(points, metric):
    
    simplices = []
    num = points.shape[0]
    index = [i for i in range(points.shape[0] + points.shape[1] + 1)]
    super_vert = bounding_simplex(points, metric)
    points = np.concatenate((points, super_vert), axis = 0)
    simplices.append(Simplex(super_vert, metric, [num + i for i in range(points.shape[1]+1)]))
    for i in range(num):
        badSimplices = []
        for simp in simplices:
            if simp.hyper_sphere.contains(points[i]):
                badSimplices.append(simp)
        polytope = set()
        for j in range(len(badSimplices)):
            not_shared = set(badSimplices[j].facets)
            for k in range(len(badSimplices)):
                if j == k:
                    continue
                shared = not_shared.intersection(badSimplices[k].facets)
                not_shared = not_shared.union(badSimplices[k].facets).difference(shared)

            polytope = polytope.union(not_shared)
            not_shared = set()
        
        for j in badSimplices:
            simplices.remove(j)
            
        for j in polytope:
            lst = list(j)
            lst.append(i)
            tsimp = Simplex(points[lst, :], metric, lst)
            simplices.append(tsimp)

    fin_simplices = simplices.copy()
    for j in simplices:
        # if j.added_simplex(num):
        #     fin_simplices.remove(j)
        #     continue
        for k in range(points.shape[0]):
            if j.hyper_sphere.contains(points[k, :]):
                fin_simplices.remove(j)
                break
                
    
            
    return fin_simplices

def connect_simplices(simplices, num, dims):
    new_simplices = []
    for j in range(len(simplices)):
        if simplices[j].added_simplex(num) < 1:
            continue
        not_shared = set(simplices[j].facets)
        for k in range(len(simplices)):
            if j == k or simplices[k].added_simplex(num) < 1:
                continue
            shared = not_shared.intersection(simplices[k].facets)
            if len(shared) > 0 and (simplices[j].added_simplex(num) == dims + 1 or simplices[k].added_simplex(num) == dims + 1):
                simplices[j].connected_simplices.append(simplices[k])
        if len(simplices[j].connected_simplices) > 0:
            new_simplices.append(simplices[j])
    return new_simplices

def create_voronoi(points, metric):

    smps = bowyer_watson(points, metric)
    smps = connect_simplices(smps, points.shape[0], points.shape[1])
    smps[0].create_Graph()
    
    voronoi = [i.node for i in smps]
    
    return voronoi
    