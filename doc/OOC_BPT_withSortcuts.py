import sys
import numpy as np
from time import time
import math
index_t = np.int32
import cProfile
import pstats
from pstats import SortKey

def seq2adic(n):
    res = np.empty((n,), dtype=index_t)
    res[0] = 0
    res[1] = 1
    t = 2
    notfinished = t!=n 
    i=2
    while notfinished:
        end = t * 2
        if end > n:
            end = n
            notfinished = False
        res[t:end] = res[0:end-t]
        if notfinished:
            res[end-1] = i
            i += 1
        t=end
    return res

class GrapheImg:
    def __init__(self, nLignes, nCols):
        self.nLignes = nLignes
        self.nCols = nCols
        self.nSoms = nLignes * nCols
        self.nAretes = 2 * self.nSoms
        
    def Edge(self, u):
        if ( self.isEdge(u) ): 
            if (u%2 == 0):
                    return (u//2,u//2+1)
            else :
                    return (u//2, u//2+self.nCols)
        else :
            return (-1,-1)

    def edgeIndex(self, u):
        
        if u[1] == u[0] + 1 : 
            return u[0] * 2
        else : 
            return u[0] * 2 + 1

    def edgeIndices(self, u):
        
        test = u[:, 1] != u[:, 0] + 1
        res  = u[:, 0] * 2
        res[test] += 1 
        return res
       
    
    def isEdge(self, u):
        if ( (u < self.nAretes) and ( u >= 0) ): 
            if (u%2 == 0):
                # arete horizontal
                if ( ((u//2) % self.nCols) == self.nCols -1) :
                    return False
            else:
                if ( (u//2) // self.nCols == self.nLignes-1):
                    return False
            return True
        else : return False

    def Poids(self):
        p = np.zeros(self.nAretes, dtype = np.int32)
        return p

    def PoidsAleatoires(self,pMax):
        # genere des poids aleatoires uniformement distribues en entre
        # 0 et pMax pour le graphe self
        p = np.random.randint(pMax,size=(self.nAretes)) 
        return p

    def PoidsQuadArbre(self):
        w = np.zeros((self.nAretes,), dtype=np.int32)
        seqh = seq2adic(self.nCols)
        wh = w[0::2].reshape((self.nLignes, self.nCols))
        wh[:,:] = np.expand_dims(seqh,0)
        seqv = seq2adic(self.nLignes)
        wv = w[1::2].reshape((self.nLignes, self.nCols)) 
        wv[:,:]= np.expand_dims(seqv,1)
        return w

    def PoidsImage(self, image):
        w = np.zeros((self.nAretes,), dtype=np.int32)
        image = np.pad(image,((0,1),(0,1)))
        w[0::2] = (np.abs(image[:-1,0:-1] - image[:-1,1:])).flatten()
        w[1::2] = (np.abs(image[0:-1,:-1] - image[1:,:-1])).flatten()
        return w

class Tile:
    def __init__(self, premLigne, nLignes, g):
        self.premSom = g.nCols * premLigne
        self.nSoms = g.nCols * nLignes
        
        self.premArete = 2 * self.premSom
        self.nAretes = 2 * self.nSoms

        self.premLigne = premLigne
        self.nLignes = nLignes
        
    def  isInTile(self, som):
        return (som >= self.premSom) & (som < self.premSom + self.nSoms) 
           

    def print(self):
        print('----------------------------')
        print('-----------TILE-------------')
        print('----------------------------')
        print('premLigne : ', self.premLigne)
        print('nLignes : ', self.nLignes)
        
        print('premSom : ', self.premSom)
        print('nSom : ', self.nSoms)
        
        print('premArete : ', self.premArete)
        print('nAretes : ', self.nAretes)


        print('----------------------------')



def depth(tree):
    par = tree.parent
    depth = np.zeros_like(par)
    for i, p in enumerate(par):
        if p!=-1:
            depth[p] = max(depth[p], depth[i] + 1)
    return np.max(depth)

class Tree:
    def __init__(self, nFeuilles, size, overSize=0):
        self.maxSize = size + overSize
        self.nFeuilles = nFeuilles
        self.size = size
        self.parent = np.full((self.maxSize,), -1, dtype=index_t)
        self.nNonFeuilles = 0


    def print(self, map):
        print('----------------------------')
        print('-----------TREE-------------')
        print('----------------------------')
        print()
        print('number of nodes: ', self.size, 'number of leaves : ', self.nFeuilles)
        print()
        print('----------------------------')

        for i in range(self.size):
            if i < self.nFeuilles :
                print(i, ': v', map[i], ' - parent : e', map[self.parent[i]]) 
            else :
                if self.parent[i] >= 0:
                    print(i, ': e', map[i], ' - parent : e',  map[self.parent[i]]  )
                else :
                    print(i, ': e', map[i], ' - parent : e', self.parent[i])

        print('----------------------------')

class Tarjan:
    def __init__(self, size):
        self.par = np.zeros((size,), dtype = index_t)
        self.par = np.full((size,), -1, dtype = index_t)
        self.rank = np.zeros((size,), dtype = index_t)

    def findRoot(self, x): 
        y = x
        while(self.par[y] >= 0): y = self.par[y]

        z = x
        while(self.par[z] >= 0): 
            tmp = z
            z = self.par[z]
            self.par[tmp] = y
        return y

    def union(self, cx, cy):
        if self.rank[cx] > self.rank[cy] :
           tmp = cx
           cx = cy
           cy = tmp
        if self.rank[cx] == self.rank[cy] :
           self.rank[cy] += 1
        self.par[cx] = cy
        return cy

class BPTree:
    def __init__(self, size):
        self.maxSize = size + (size-1)
        self.nFeuilles = size
        self.size = size
        self.parent = np.full((self.maxSize,), -1, dtype=index_t) 
        self.MST = np.full((size-1,2), -1, dtype=index_t)
        self.nNonFeuilles = 0

        
    #def findRoot(self, x): 
    #    y = x
    #    while(self.parent[y] >= 0): y = self.parent[y]
    #    return y
        

    def createParent_fast(self, cx, cy, u, tarjan, tarjanMap):
        rx = tarjanMap[cx]
        ry = tarjanMap[cy]

        i = self.size
        self.size +=1
        self.parent[rx] = i
        self.parent[ry] = i
        self.MST[self.nNonFeuilles] = u
        self.nNonFeuilles +=1
 
        c = tarjan.union(cx,cy)
        
        tarjanMap[c] = i

        return i
       

    def print(self):
        print('----------------------------')
        print('-----------TREE-------------')
        print('----------------------------')
        print()
        print('number of nodes:', self.size)
        print()
        print('----------------------------')

        for i in range(self.size):
            if i < self.nFeuilles :
                print('Feuille ', i, ' : parent[', i, '] = ', self.parent[i])
            else : 
                print('NonFeuille ', i, ' : parent[', i, '] = ', self.parent[i], ' - arete :', self.MST[i-self.nFeuilles])
        print('----------------------------')

class TooManySlicesError(BaseException):
    def __init__(self, nSlices, nLignes):
        self.nSlices = nSlices
        self.nLignes = nLignes

class OocBpt:

    def __init__(self, g, w, n):
        # where (g,w) is an edge weighted 4-adjacency graph and where n is
        # the number of desired slices
        # returns a distributed BPT
        


        if(n > g.nLignes): raise TooManySlicesError(n, g.nLignes)
        self.nTranches = n

        nLignesTranche = g.nLignes // n 
        self.tuile = [Tile(r * (nLignesTranche+1), nLignesTranche+1, g) for r in range(g.nLignes%n)] + [Tile( (g.nLignes%n)*(nLignesTranche+1) + (r) * nLignesTranche, nLignesTranche, g) for r in range(self.nTranches-g.nLignes%n)]
            
        self.bordDroit = [Tile(t.premLigne + t.nLignes -1, 2, g) for t in self.tuile]

        self.bordGauche = [self.bordDroit[t-1] for t in range(len(self.tuile))]

        self.bpt = [tileBPT(g,w,t) for t in self.tuile]
            
        #  print('on commence le parcours causal')

        self.arbreBordDroit = [(None,None) for i in range(n)]
        self.arbreBordGauche = [(None,None) for i in range(n)]
        self.mergeTree = [(None, None) for i in range(n-1)]

        self.arbreBordDroit[0] = selectTileFromBPTree(self.bordDroit[0], self.bpt[0], self.tuile[0], g)

        for i in range(1,n):
            #print('Parcours causal tuile no:',i)
            self.arbreBordGauche[i] = selectTileFromBPTree(self.bordGauche[i], self.bpt[i], self.tuile[i], g) 

            self.mergeTree[i-1] = merge_tree_new(self.arbreBordDroit[i-1][0], self.arbreBordDroit[i-1][1], self.arbreBordGauche[i][0], self.arbreBordGauche[i][1], self.bordGauche[i], w,g)

            (ttemp, mtemp) = selectTileFromTree(self.tuile[i], self.mergeTree[i-1][0],self.bordGauche[i],g, self.mergeTree[i-1][1])

            mbpt1 = np.zeros(self.bpt[i].size, dtype = index_t)
            mbpt1[0:self.bpt[i].nFeuilles] = np.arange(self.tuile[i].premSom, self.tuile[i].premSom+self.bpt[i].nFeuilles) 
            mbpt1[self.bpt[i].nFeuilles:self.bpt[i].size] =  g.edgeIndices(self.bpt[i].MST)# [g.edgeIndex(u) for u in self.bpt[i].MST[:]] 
            self.bpt[i] = update(self.bpt[i],ttemp, mbpt1, mtemp, w)
            if i < n-1:
                self.arbreBordDroit[i] = selectTileFromTree(self.bordDroit[i], self.bpt[i][0], self.tuile[i], g, self.bpt[i][1])


        # bpt[0] n'est pas au bon format, il faut le convertir :

        mbpt = np.zeros(self.bpt[0].size, dtype = index_t)
        mbpt[0:self.bpt[0].nFeuilles] = np.arange(self.tuile[0].premSom,self.tuile[0].premSom+self.bpt[0].nFeuilles) # [i for i in range(self.tuile[0].premSom,self.tuile[0].premSom+self.bpt[0].nFeuilles)]
        mbpt[self.bpt[0].nFeuilles:self.bpt[0].size] = g.edgeIndices(self.bpt[0].MST) # [g.edgeIndex(u) for u in self.bpt[0].MST[:]]

        self.bpt[0] = (self.bpt[0], mbpt)

        for i in reversed(range(0,n-1)):
            #print('Parcours anticausal, tuile no', i)
            (ttemp, mtemp) = selectTileFromTree(self.tuile[i], self.mergeTree[i][0], self.bordDroit[i], g, self.mergeTree[i][1])
            self.bpt[i] = update(self.bpt[i][0], ttemp, self.bpt[i][1], mtemp, w)
            
            if i > 0 :
                (ttemp, mtemp) = selectTileFromTree(self.bordGauche[i], self.bpt[i][0], self.tuile[i], g, self.bpt[i][1])
                self.mergeTree[i-1] = update(self.mergeTree[i-1][0], ttemp, self.mergeTree[i-1][1], mtemp, w)

            
def tileBPT(g, w, t):
    # retourne le BPT du graphe (g,w) sur le tile t

    edge_index = np.arange(t.premArete, t.premArete + t.nAretes)
    sorted_index = np.lexsort(( edge_index ,  w[t.premArete:t.premArete + t.nAretes]))

    bpt = BPTree(t.nSoms)
    tarjan = Tarjan(t.nSoms)
    tarjanMap = np.arange(t.nSoms) 

    found_edges = 0
    for i in sorted_index: 
        u = g.Edge(edge_index[i]) 
        if (t.isInTile(u[0])) & (t.isInTile(u[1])):
            rx = tarjan.findRoot(u[0]-t.premSom)
            ry = tarjan.findRoot(u[1]-t.premSom)
            if rx != ry :
                bpt.createParent_fast(rx, ry, u, tarjan, tarjanMap)
                found_edges += 1
                if found_edges == t.nSoms -1:
                    break
    return bpt



def selectTileFromBPTree(selTile, bpt, bptTile, g):
    # select a subpart of a bpt of a tile from a given tile where
    # selTile is the tile of that we want to select inside bpt and
    # where bptTile is the tile upon which bpt is built and where g is
    # the overal graph space

    # I doubt that this fiunction is generic enough to handle all
    # possible selection situations
    
    left = max(selTile.premSom, bptTile.premSom)
    right = min(selTile.premSom + selTile.nSoms, bptTile.premSom + bptTile.nSoms)
    flag = np.full((bpt.size,), -1, dtype = index_t)
    

    som_slice = slice(left - bptTile.premSom, right - bptTile.premSom)
    nFeuilles = som_slice.stop - som_slice.start
    flag[som_slice] = np.arange(nFeuilles)
    flag[bpt.parent[som_slice]] = bpt.size
    
    taille = nFeuilles

    for n in range(bpt.nFeuilles, bpt.size-1):
        # pour toutes les noeuds non feuille sauf la racine
        if flag[n] != -1 : 
            flag[n] = taille
            taille +=1
            flag[bpt.parent[n]] = bpt.size
    
    flag[bpt.size-1] = taille
    taille +=1

    sT = Tree(nFeuilles, taille)
    sT.nNonFeuilles = sT.size-sT.nFeuilles
   
    map = np.zeros(taille, dtype = index_t)
    for n in range(som_slice.start, som_slice.stop):
        if flag[n] >= 0 :
            map[flag[n]] = n
            sT.parent[flag[n]] = flag[bpt.parent[n]]
        
    for n in range(bpt.nFeuilles, bpt.size-1):
        if flag[n] > 0 :
            map[flag[n]] = n
            sT.parent[flag[n]] = flag[bpt.parent[n]]

    map[taille-1] = bpt.size-1

    map[0:sT.nFeuilles] = map[0:sT.nFeuilles] + bptTile.premSom
 
    for i in range(sT.nFeuilles, sT.size):
        u = bpt.MST[map[i] - bpt.nFeuilles]
        #map[i] = g.edgeIndex(u+bptTile.premSom)
        map[i] = g.edgeIndex(u)
    return sT, map

# This version is supposed to be more generic than the previous one, in the sense that it can handle a tree with a map array instead of a bpt (basically a bpt is a tree with a map array only for the non-leaf nodes)

def selectTileFromTree(selTile, t, tTile, g, tmap):
    # select a subpart of a tree t defined over the tile tTile tile
    # from a given tile where selTile is the tile of that we want to
    # select inside t and where tTile is the tile upon which t is
    # built and where g is the overal graph space. The array map is a
    # mapping between the elements of the tree t and the elements of
    # the graph g. The array tmap provides the mapping from the nodes
    # of t to the elements of g (the leaves are mapped to vertices of
    # g and non-leaf nodes are mapped to edges of g)

    # the nodes of t are supposed to be sorted
    
    # I hope that this function is generic enough to handle all
    # possible selection situations
    
    # left = max(selTile.premSom, bptTile.premSom)
    # right = min(selTile.premSom + selTile.nSoms, bptTile.premSom + bptTile.nSoms)
    left = max(selTile.premSom, tmap[0])
    right = min(selTile.premSom + selTile.nSoms-1, tmap[t.nFeuilles-1])
    flag = np.full((t.size,), -1, dtype = index_t)
    nFeuilles = 0

    for n in range(t.nFeuilles):
        if tmap[n] >= left and tmap[n] <= right: 
            flag[n] = nFeuilles
            nFeuilles +=1
            if t.parent[n] < t.nFeuilles :
                print(t.parent[n], 'is a wrong parent for', n, 'since there are', t.nFeuilles, 'feuilles')
                print('ce noeud a pour mapping:',  tmap[n])
             #   t.print(tmap)
                exit(0)
            flag[t.parent[n]] = t.size

    taille = nFeuilles
  
    for n in range(t.nFeuilles, t.size):
        # pour toutes les noeuds non feuille 
        if flag[n] != -1 : 
            flag[n] = taille
            taille +=1
            if t.parent[n] >= 0: 
                if flag[t.parent[n]] < taille and flag[t.parent[n]] != -1:
                    print('the tree t is not wrongly sorted')
                    exit(0)
                flag[t.parent[n]]=t.size


    sT = Tree(nFeuilles, taille)
    map = np.zeros(taille, dtype = index_t)

    for n in range(t.size):
        if flag[n] >= 0 :

            map[flag[n]] = tmap[n]
            if t.parent[n] >= 0 :
                sT.parent[flag[n]] = flag[t.parent[n]]
            else : 
                sT.parent[flag[n]] = t.parent[n]

    return sT, map


def printBorderTree(btree, bpt, map, t, bordGauche, g):

    print('----------------------------')
    print('-----------TREE-------------')
    print('----------------------------')
    print()
    print('Tuile - debut: ', t.premLigne, ' - nb de lignes: ', t.nLignes)
    print('Bord - debut: ', bordGauche.premLigne)
    print()
    print('----------------------------')
    print()
    print('number of nodes:', btree.size)
    print('number of leaves:', btree.nFeuilles)
    print()
    print('----------------------------')

    for i in range(btree.size):
        if i < btree.nFeuilles :
#            print('Feuille ', i, '- sommets :', map[i]+t.premSom, ' - parent[', i, '] = ', btree.parent[i]  )
            print('Feuille ', i, '- sommets :', map[i], ' - parent[', i, '] = ', btree.parent[i]  )
        else : 
#            print('NonFeuille ', i, ' - arete :', bpt.MST[map[i]-bpt.nFeuilles], ' : parent[', i, '] = ', btree.parent[i])
            print('NonFeuille ', i, ' - arete :', g.Edge(map[i]), ' : parent[', i, '] = ', btree.parent[i])
    print('----------------------------')


              
def infAncestor(bpt, map, w, x, u, shortCut, shortCutMap):
    cx = shortCut.findRoot(x)
    n = shortCutMap[x]
    while (bpt.parent[n] != -1) & ((w[map[bpt.parent[n]]], map[bpt.parent[n]]) < (w[u], u)):
        cn = shortCut.findRoot(bpt.parent[n])
        n = shortCutMap[cn]
        cx = shortCut.union(cx, cn)

    shortCutMap[cx] = n
    return n

#def infLexi(a1, a2, b1, b2) :
#    # retourne vrai si le couple (a1, a2) est inferieur au couple (b1, b2)
#    if a1 < b1 : return True
#    if a1 == b1 and a2 < b2 : return True
#    return False

def infLexi(a, b) :
    for i, j in zip(a,b):
        if i < j:
            return True
        elif i > j:
            return False
    return False


size_tree1 = []
size_tree2 = []
nbArretes = []
num_while = []
depth_tree1 = []





def merge_tree_new(bpt1, map1, bpt2, map2, border, w, g):

    # merge tree, max size = size bpt1 + size bpt 2 + size border
    num_border_edges = g.nCols
    num_leaves = bpt1.nFeuilles + bpt2.nFeuilles
    mt = Tree(num_leaves, num_leaves, bpt1.size + bpt2.size +  num_border_edges)
    
    # for a node n of mt
    #  - if n is a leaf, mst_map[n] is the index of n in the full graph
    #  - if n is not a lead, mst_map[n] is the index of the edge of the mst associated to n
    mst_map = np.full((bpt1.size + bpt2.size + num_border_edges,), -1, dtype= index_t)
    mst_map[:bpt1.nFeuilles] = map1[:bpt1.nFeuilles]
    mst_map[bpt1.nFeuilles:bpt1.nFeuilles+bpt2.nFeuilles] =  map2[:bpt2.nFeuilles] 

    # vertical edges between border tree leaves
    edge_slice = slice(1 + 2 * border.premSom, 1 + 2 * (border.premSom + num_border_edges), 2)# (border.premSom+ np.arange(nBAretes))*2+1
    ww = w[edge_slice]
    edge_index = np.arange(edge_slice.start, edge_slice.stop, 2)  
    sorted_edge_index = np.lexsort((edge_index, ww))  
    num_edges = len(sorted_edge_index)

    # for a node n, 
    #  - attr[n,0] is equal to the index of a leaf contained in the first child of n
    #  - attr[n,1] is equal to the index of a leaf contained in the second child of n
    #       or -1 if n has only one child
    def attribute_child_one_leaf_node(tree, shift):
        attr = np.full((tree.size, 2), -1, dtype= index_t)
        attr[:tree.nFeuilles, 0] = np.arange(shift, shift + tree.nFeuilles)
        for i in range(0, tree.size - 1):
            p = tree.parent[i]
            ip = 0 if attr[p, 0] == -1 else 1
            attr[p, ip] = attr[i, 0]
        return attr

    child_one_leaf_bpt1 = attribute_child_one_leaf_node(bpt1, shift=0)
    child_one_leaf_bpt2 = attribute_child_one_leaf_node(bpt2, shift=bpt1.nFeuilles)

    # current indices in the three edge sets
    ibpt1 = bpt1.nFeuilles
    ibpt2 = bpt2.nFeuilles
    iedge = 0

    infty = (float("inf"), float("inf"))

    tarjan = Tarjan(mt.nFeuilles)
    tarjanMap = np.arange(mt.nFeuilles) 
    firstNode = border.premSom

    while ibpt1 < bpt1.size or ibpt2 < bpt2.size or iedge < num_edges:
        wibpt1 = (w[map1[ibpt1]], map1[ibpt1]) if ibpt1 < bpt1.size else infty
        wibpt2 = (w[map2[ibpt2]], map2[ibpt2]) if ibpt2 < bpt2.size else infty
        wiedge = (w[edge_index[sorted_edge_index[iedge]]], edge_index[sorted_edge_index[iedge]]) if iedge < num_edges else infty
        rx, ry = None, None

        if wiedge < wibpt1 and wiedge < wibpt2: # smallest is a border edge : normal Tarjan
            u = g.Edge(wiedge[1])
            rx = tarjan.findRoot(u[0] - firstNode)
            ry = tarjan.findRoot(u[1] - firstNode)
            iedge += 1
            wedge = wiedge
        else: # smallest is a node of a border tree
            if wibpt1 < wibpt2: # find which one
                index = ibpt1
                wedge = wibpt1
                attr = child_one_leaf_bpt1

                ibpt1 += 1
            else:
                index = ibpt2
                wedge = wibpt2
                attr  = child_one_leaf_bpt2

                ibpt2 += 1
            
            # find canonical element associated to the first child of the node
            rx = tarjan.findRoot(attr[index, 0])
            if attr[index, 1] != -1: # find canonical element associated to the second child of the node (if it exists)
                ry = tarjan.findRoot(attr[index, 1])

        if rx != ry:
            root_rx = tarjanMap[rx]
            new_node = mt.size
            mt.size += 1
            mst_map[new_node] = wedge[1]
            mt.parent[root_rx] = new_node

            if ry is None:
                # the other part of the edge is not in the domain: no union
                tarjanMap[rx] = new_node
            else:
                root_ry = tarjanMap[ry]
                mt.parent[root_ry] = new_node
                c = tarjan.union(rx,ry)
                tarjanMap[c] = new_node

    return mt, mst_map

                
def sort_merge_tree(mt, map, bpt1, bpt2, w):
    i1 = 0
    i2 = 0
    r = 0

    

    Rank = np.zeros(mt.size, dtype = np.int32)

    nValidBAretes = mt.size - (bpt1.size + bpt2.size)
    
    while r < mt.nFeuilles:
        if i1 >= bpt1.nFeuilles :
            Rank[bpt1.nFeuilles+i2] = r
            i2 += 1
        elif i2 >= bpt2.nFeuilles :
            Rank[i1] = r
            i1 += 1
        elif map[i1] < map[bpt1.nFeuilles+i2]:
            Rank[i1] = r
            i1 +=1
        else:
            Rank[bpt1.nFeuilles+i2]
            i2 += 1
        r += 1
                

    nNonLeaf1 = bpt1.size - bpt1.nFeuilles
    nNonLeaf2 = bpt2.size - bpt2.nFeuilles
    i1 = 0
    i2 = 0
    i3 = 0

    # On prepare les valeurs a comparer
    def val(n):
        return w[map[n]], map[n]

    a = val(mt.nFeuilles)
    b = val(mt.nFeuilles + nNonLeaf1)
    c = val(bpt1.size + bpt2.size)
    infty = (float("inf"), float("inf"))
    while r < mt.size:
        if(infLexi(a, b)):
            if infLexi(a, c):
                # a est le plus petit :
                Rank[mt.nFeuilles+i1] = r
                i1 += 1
                if(i1 >= nNonLeaf1): a = infty
                else : a = val(mt.nFeuilles + i1) #(w[map[mt.nFeuilles + i1]], map[mt.nFeuilles + i1])
            else :
                Rank[bpt1.size + bpt2.size + i3] = r
                i3 += 1
                if (i3 >= nValidBAretes ) : c = infty
                else : c = val(bpt1.size + bpt2.size + i3) #(w[map[bpt1.size + bpt2.size + i3]], map[bpt1.size + bpt2.size + i3]) 
        elif infLexi(b, c):
            # b est le plus petit
            Rank[mt.nFeuilles+nNonLeaf1+i2] = r
            i2 += 1
            if(i2 >= nNonLeaf2): b = infty
            else : b = val(mt.nFeuilles + nNonLeaf1 + i2) # w[map[mt.nFeuilles + nNonLeaf1 + i2]], map[mt.nFeuilles + nNonLeaf1+i2]
        else:
            # c'est c le plus petut
            Rank[bpt1.size + bpt2.size + i3] = r
            i3 += 1
            if (i3 >= nValidBAretes) : c = infty
            else : c = val(bpt1.size + bpt2.size + i3)# (w[map[bpt1.size + bpt2.size + i3]], map[bpt1.size + bpt2.size + i3])

        r+=1


    mapo = np.zeros(mt.size, dtype = index_t)
    parento = np.zeros(mt.size, dtype = index_t)

    for i in range(mt.size):
        mapo[Rank[i]] = map[i]
        if mt.parent[i] < 0 : parento[Rank[i]] = mt.parent[i]
        else : parento[Rank[i]] = Rank[mt.parent[i]]
                
    mt.parent = parento

 
    return mt, mapo            

def mergeTree(bpt1, map1, bpt2, map2, border, w, g):
    nBAretes = g.nCols

   

#    print('Merge tree du bord suivant:')
 #   border.print()
    
    mt = Tree(bpt1.nFeuilles + bpt2.nFeuilles, bpt1.size + bpt2.size,  nBAretes)
    
    map = np.full((bpt1.size + bpt2.size + nBAretes,), -1, dtype= index_t)

    mt.parent[:bpt1.nFeuilles] = bpt1.parent[:bpt1.nFeuilles] + bpt2.nFeuilles

    map[:bpt1.nFeuilles] = map1[:bpt1.nFeuilles] 
    
    mt.parent[bpt1.nFeuilles:bpt1.nFeuilles+bpt2.nFeuilles] = bpt2.parent[:bpt2.nFeuilles] +  bpt1.size 

    map[bpt1.nFeuilles:bpt1.nFeuilles+bpt2.nFeuilles] =  map2[:bpt2.nFeuilles] 
   
    mt.parent[bpt1.nFeuilles + bpt2.nFeuilles:bpt1.size+bpt2.nFeuilles-1] = bpt1.parent[bpt1.nFeuilles:bpt1.size-1] + bpt2.nFeuilles
    map[bpt1.nFeuilles + bpt2.nFeuilles:bpt1.size+bpt2.nFeuilles] = map1[bpt1.nFeuilles:bpt1.size] 

    mt.parent[bpt1.size+bpt2.nFeuilles:bpt1.size + bpt2.size -1] = bpt2.parent[bpt2.nFeuilles:bpt2.size-1] + bpt1.size
    map[bpt1.size+bpt2.nFeuilles:bpt1.size + bpt2.size] = map2[bpt2.nFeuilles:bpt2.size]

    mt.nNonFeuilles = bpt1.nNonFeuilles + bpt2.nNonFeuilles
    
     

    # vertical edges
    edge_slice = slice(1 + 2 * border.premSom, 1 + 2 * (border.premSom + nBAretes), 2)# (border.premSom+ np.arange(nBAretes))*2+1
    ww = w[edge_slice]
    edge_index = np.arange(edge_slice.start, edge_slice.stop, 2)  
    sorted_edge_index = np.lexsort((edge_index, ww))  

    size_tree1.append(bpt1.size)
    size_tree2.append(bpt2.size)
    depth_tree1.append(depth(bpt1))
    nbArretes.append(len(sorted_edge_index))
    n_while = 0
    

    # Initialisation de la structure pour les racourcis de remontees dans l'arbre
    shortCut = Tarjan(mt.maxSize)
    shortCutMap = np.arange(mt.maxSize, dtype = np.int32)

    # Les arbres ne sont pas fusionnes la racine virtuelle est donc a un poids infini

    rootWeight = (257,0)
    
    #for p,u in ww:
    for i in sorted_edge_index:
        p =  ww[i]
        u = edge_index[i]
        #if (p,u) > rootWeight : break
        (x,y) = g.Edge(u)
        nx = x % g.nCols
        ny = y % g.nCols + bpt1.nFeuilles
        nx = infAncestor(mt, map, w, nx, u, shortCut, shortCutMap)
        ny = infAncestor(mt, map, w, ny, u, shortCut, shortCutMap)
        if nx != ny :
            p0 = mt.parent[nx]
            p1 = mt.parent[ny]
            map[mt.size] = u
            mt.parent[nx] = mt.size
            mt.parent[ny] = mt.size
            z = mt.size
            mt.size+=1
            mt.nNonFeuilles += 1            
            if p0 == p1 :
                mt.parent[z] = p0
            
            while z != -1 and p0 != p1:
                n_while += 1
                if p0 == -1 :
                    temp = p0
                    p0 = p1
                    p1 = temp
                elif p1 != -1 and ( w[map[p0]] > w[map[p1]] or (w[map[p0]] == w[map[p1]] and map[p0] > map[p1]) ):
                    temp = p0
                    p0 = p1
                    p1 = temp                
                mt.parent[z] = p0
                z = p0
                p0 = mt.parent[p0]          
            if z!= -1 and p0 != -1 : 
                mt.parent[z] = mt.parent[p0]
                mt.parent[p0] = -2
                if (mt.parent[z] == -1) : rootWeight = (w[map[z]],map[z])

    num_while.append(n_while)

    return sort_merge_tree(mt, map, bpt1, bpt2, w)



def update(t,u, mapt, mapu, w):
    # return the updated version of t according to u
    # The nodes of t and of u are assumed to be already sorted
    # w is the edge weight map of the underlying graph g

    # the nodes of the (returned) updated tree remains sorted 
    
    # I have a doubt concerning the way cancelled nodes are dealt, I think that there is nothing to do indeed ;-)
    
    # On donne un mumero a chaque noeud de t et de u par ordre
    # croissant

    # Le probleme des noeuds annules se posera 

#     print('We start by ranking the leaves of t and u')

    iu = 0
    it = 0
    r = 0
    ft = np.zeros(t.size, dtype = index_t)
    fu = np.zeros(u.size, dtype = index_t)


    R = np.zeros((t.size+u.size, 2), dtype=index_t)
    
    # Les noeud etant traites dans l'ordre
    
    while iu < u.nFeuilles or it < t.nFeuilles:
        if  iu < u.nFeuilles and it < t.nFeuilles and mapt[it] == mapu[iu] :
            ft[it] = r
            fu[iu] = r
            #R[r] = (iu,it)
            R[r, 0] = iu
            R[r, 1] = it
            it+=1
            iu+=1
        elif iu >= u.nFeuilles:
            # on marque comme vivant le parent de it dans t
            ft[t.parent[it]] = -2
            ft[it] = r
            #R[r] = (-1,it)
            R[r, 0] = -1
            R[r, 1] = it
            it+=1
        elif it >= t.nFeuilles:
            fu[iu] = r
            #R[r] = (iu,-1)
            R[r, 0] = iu
            R[r, 1] = -1
            iu+=1
        elif mapt[it] < mapu[iu] :
            # on marque comme vivant le parent de it dans t
            ft[t.parent[it]] = -2
            ft[it] = r
            #R[r] = (-1,it)
            R[r, 0] = -1
            R[r, 1] = it
            it+=1
        else:
            fu[iu] = r
            #R[r] = (iu,-1)
            R[r, 0] = iu
            R[r, 1] = -1
            iu+=1
        r+=1

    nFeuilles = r


    while iu < u.size or it < t.size:
        if iu < u.size and it < t.size and mapt[it] == mapu[iu]:
            ft[it] = r
            fu[iu] = r
            #R[r] = (iu,it)
            R[r, 0] = iu
            R[r, 1] = it
            it+=1
            iu+=1
        elif iu >= u.size or (it < t.size and infLexi((w[mapt[it]], mapt[it]), (w[mapu[iu]], mapu[iu]))):
            if ft[it] == -2:
                # on marque comme vivant le parent de it dans t
                if t.parent[it] >= 0: ft[t.parent[it]] = -2
                ft[it] = r
                R[r, 0] = -1
                R[r, 1] = it
            else : r -= 1
            it+=1
        else:
            fu[iu] = r
            #R[r] = (iu,-1)
            R[r, 0] = iu
            R[r, 1] = -1
            iu+=1
        r+=1

    nNoeuds = r
    

    v = Tree(nFeuilles, nNoeuds, 0)
    map = np.zeros(nNoeuds, dtype = index_t)
    
    for r in range(nNoeuds):
        if R[r,0] != -1:
            if u.parent[ R[r,0]] < 0: v.parent[r] = u.parent[ R[r,0]]
            else : v.parent[r] = fu[u.parent[ R[r,0]]]
            map[r] = mapu[R[r,0]]
        else :
            if t.parent[ R[r,1]] <0  : v.parent[r] = t.parent[ R[r,1]]
            else: v.parent[r] = ft[t.parent[ R[r,1]]]
            map[r] = mapt[R[r,1]]
    return v,map



def testTime (sizes, slices, iter=3):
    fo = open("times.csv", "w")
    #print( "OOC-BPT : Temps d'execution pour une image de taille de", nLignes, " lignes et de ", nCols, "colonnes",file=fo)
    print("taille; nombre de tuiles; taille moyen des arbres; temps moyen d'execution", file=fo)
    np.random.seed(42)
    res = np.zeros((len(sizes), len(slices)))
    for k, s in enumerate(sizes):
        for l, n in enumerate(slices):
            temps = 0
            taille = 0

            g = GrapheImg(s, s)
        
            for i in range(iter):
                print("test ", s, n, i)
                w = g.PoidsAleatoires(256)

                t0 = time()
                if n > 1:
                    A = OocBpt(g, w, n)
                else:
                    t = Tile(0, g.nLignes, g)
                    A = tileBPT(g, w, t)
                    taille += A.size
                t1 = time()
                temps += t1-t0
                if n > 1:
                    for i in range(n):
                        taille += A.bpt[i][0].size

    
            res[k,l] = temps/iter
            print(s , "; ", n, "; ", taille/n/iter, "; ", temps/iter, file=fo)
    fo.close()
    print(res)
    
    
def testBasique(size, n): 
    import imageio

    im = imageio.imread("fish_brain_4k.tif")
    im = im[:size[0], :size[1]]
    #g = GrapheImg(6,2)

    g = GrapheImg(*size)

  
    np.random.seed(42)
    #w = g.PoidsAleatoires(256)#10
    #print(w)
    w = g.PoidsImage(im)
    #w2 = g.PoidsQuadArbre()
    
    #w = w * 100 + w2


    t0 = time()
    A = OocBpt(g,w,n)
    t1 = time()
    print('on a cree un oocbpt')

    taille = 0
    for i in range(n):
        taille += A.bpt[i][0].size
    print("(nombre de noeuds du bpt, nombre moyend de noeuds du BPT distribuee, temps de construction du bpt distribue, nombre de tuile)")
    print(g.nSoms *2 -1, taille/n,  (t1-t0), n )

    


    #Verification automatique
    #On construit le BPT complet du graphe

    camarche = True

    t = Tile(0, g.nLignes,g)
    t2 = time()
    B= tileBPT(g,w,t)
    t3 = time()

    #return 
    print("temps BPT simple", t3-t2)
    print("profondeur BPT simple", depth(B), "log N", math.log(size[0]*size[1]))
    for i in range(n):
       (tbpt, mtbpt) = selectTileFromBPTree(A.tuile[i], B, t, g)


       if not (np.array_equal(A.bpt[i][0].parent, tbpt.parent) and np.array_equal(A.bpt[i][1], mtbpt)):
           camarche = False
           print('tuile ', i, ': resultat errone !!!')
           print('On trouve :')
           A.bpt[i][0].print(A.bpt[i][1])
           print('On devrait trouver :')
           tbpt.print(mtbpt)


    print("ok : ", camarche )

    #print("temps de calcul du bpt avec PlayingWithKruskal :", (t3-t2))
        
    #print("(nbre de noeuds du BPT, nombre moyend de npeuds du BPT distribuee, temps de construction du BP, temps de construction du bpt distribue, nombre de tuile)")
    #print(B.size , taille/n, (t3-t2), (t1-t0), n )

    # print("(nombre de noeuds du bpt, nombre moyend de noeuds du BPT distribuee, temps de construction du bpt distribue, nombre de tuile)")
    # print(g.nSoms *2 -1, taille/n,  (t1-t0), n )

#testTime([100,250,500], [1, 2, 3, 5, 10, 15, 30], iter=3)

def print_a(a):
    a = [str(e) for e in a]
    print(" ".join(a))

#cProfile.run('testBasique((500,500),20)','restats')
#p = pstats.Stats('restats')
#p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)


testBasique((4000,4000),4)

print_a(size_tree1)
#print_a(size_tree2)
#print_a(nbArretes)
print_a(depth_tree1)
print_a(num_while)


