import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.spatial.distance

class Simplex():
    
    def __init__(self, vertices, metric, inds):
        
        self.connected_simplices = []
        self.inds = inds # index of points which defines the vertices, helps with checking simplex equality
        self.vertices = vertices
        self.metric = metric
        
        # Calculate the hypersphere that circumscribes the simplex
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
                     
        self.node = None
        self.facets = [frozenset([j for j in self.inds if j != i]) for i in self.inds]
    
    def __eq__(self, smp): # simplices are equal if they have the same vertices
        
        shared = [True for i in self.inds if i in smp.inds]
        return len(shared) == self.vertices.shape[1] + 1
    
    
    def plot_2D(self):
        
        plt.plot([self.vertices[2, 0], self.vertices[0, 0]], [self.vertices[2, 1], self.vertices[0, 1]])
        plt.plot([self.vertices[0, 0], self.vertices[1, 0]], [self.vertices[0, 1], self.vertices[1, 1]])
        plt.plot([self.vertices[1, 0], self.vertices[2, 0]], [self.vertices[1, 1], self.vertices[2, 1]])
        
    def getNode(self):
        if self.node is None:
            return self.create_Graph() # this creates the node and initialise others but we don't want this if we have already created the node
        else:
            return self.node
        
    def create_Graph(self): # initisialises the simplices node and all connected nodes
        if len(self.connected_simplices) == 0:
            raise Exception("Connected simplices have not been computed")
            
        self.node = Node(self.hyper_sphere.centre, []) # we must initialise the node before creating connected nodes
        self.node.Nodes = [smp.getNode() for smp in self.connected_simplices] # recursive step
        
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
            
    def __str__(self):
        if self is None:
            return "None"
        
        return str(self.position) + " " + str([i.position for i in self.Nodes])
    
    def __str__(self):
        if self is None:
            return "None"
        
        return str(self.position) + " " + str([i.position for i in self.Nodes])
        
    
def bounding_simplex(points, metric):
    """
    Construct a simplex which bounds the given points
    """
    
    zero = np.zeros((points.shape[1],))
    dmins = [-10 * abs(np.min(points[:, i]))  for i in range(points.shape[1])]
    dmaxs = [10 * abs(np.max(points[:, i])) for i in range(points.shape[1])]
    
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
    """
    Implementation of the Bowyer-Watson algorithm
    """
    
    simplices = []
    num = points.shape[0]
    index = [i for i in range(points.shape[0] + points.shape[1] + 1)]
    super_vert = bounding_simplex(points, metric)
    points = np.concatenate((points, super_vert), axis = 0)
    simplices.append(Simplex(super_vert, metric, [num + i for i in range(points.shape[1]+1)]))
    for i in range(num+1):
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

def acceptable_shared_facet(facets, num):
    """
    Because we added in a bounding simplex we added extra points. 
    This creates extra facets in the voronoi that do not partition points in the dataset.
    So facets with less then two vertices from the original data set are removed which
    is equivalent to not connecting the simplices
    """
    for i in facets:
        contains_original = [True for j in i if j < num]
        if len(contains_original) >= 2:
            return True
    
    return False

def connect_simplices(simplices, num, dims):
    """
    Connected the simplices which border each other. 
    This helps with constructing the graph.
    """
    new_simplices = []
    for j in range(len(simplices)):
        not_shared = set(simplices[j].facets)
        for k in range(len(simplices)):
            if j == k:
                continue
            shared = not_shared.intersection(simplices[k].facets)
            if len(shared) > 0 and (acceptable_shared_facet(shared, num)):
                simplices[j].connected_simplices.append(simplices[k])
        if len(simplices[j].connected_simplices) > 0:
            new_simplices.append(simplices[j])
    return new_simplices        

def create_voronoi(points, metric):
    """
    Given the points create the voronoi diagram.
    1. Construct Delauney Triangulation (Bowyer-Watson)
    2. Create Dual Graph (Connected Centres of Hyperspheres)
    """

    smps = bowyer_watson(points, metric)
    smps = connect_simplices(smps, points.shape[0], points.shape[1])
    for i in smps:
        if i.node is None:
            i.create_Graph()
    
    for i in smps:
        i.plot_2D()
    plt.show()
    
    voronoi = [i.node for i in smps]
    
    return voronoi
    
    