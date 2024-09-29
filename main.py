import numpy as np
import uuid
from typing import Set
from collections import deque

import networkx as nx
import matplotlib.pyplot as plt

class Sample:
    def __init__(self, scores, name=None) -> None:
        self.name = name
        self.id = uuid.uuid4()
        self.scores = np.array(scores)
        self.children: Set['Sample'] = set()
        self.parents: Set['Sample'] = set()
    def __eq__(self,other):
        if not isinstance(other, Sample):
            return NotImplemented
        return self.id==other.id
    def __hash__(self) -> int: return hash(self.id)
    def __repr__(self): return self.name if self.name else f"[#{self.id} {self.scores}]"
    def __lt__(self, other): return any(self.scores<other.scores) and all(self.scores<=other.scores)
    def __gt__(self, other): return any(self.scores>other.scores) and all(self.scores>=other.scores)
    def addChild(self, c:"Sample"):
        self.children.add(c)
        c.parents.add(self)
    def addParent(self, p:"Sample"):
        self.parents.add(p)
        p.children.add(self)

class Pool:
    def __init__(self) -> None:
        self.head = Sample(np.full(3,np.inf), name="head")
        self.tail = Sample(np.full(3,-np.inf), name="tail")
        self.head.addChild(self.tail)
        self.nodes = set()
    def __len__(self):
        return len(self.nodes)
    def add(self,nNode:Sample):
        if nNode in self.nodes:
            raise ValueError("Samples must be unique")
        queue = deque([self.head])
        visited = set()
        while queue:
            cNode = queue.popleft()
            if cNode in visited:
                continue
            visited.add(cNode)
            self.nodes.add(nNode)
            dominatingChildren = [c for c in cNode.children if c>nNode]
            dominatedChildren = [c for c in cNode.children if c<nNode]
            if len(dominatingChildren)>0:
                queue.extend(dominatingChildren)
            else:
                for c in dominatedChildren:
                    c.parents.remove(cNode)
                    cNode.children.remove(c)
                    nNode.addChild(c)
                cNode.addChild(nNode)
    def pop(self,sNodes:Set[Sample]):
        outerChildren:Set[Sample] = set()
        self.nodes-=sNodes
        for p in sNodes:
            outerChildren.update(p.children)
        outerChildren-=sNodes
        for c in outerChildren:
            c.parents-=sNodes
            c.addParent(self.head)
        self.head.children-=sNodes
        return sNodes
    def sample(self, n):
        if len(self)==0:
            raise ValueError("Cannot sample from an empty pool")
        queue = deque(self.head.children)
        selected = set()
        while queue and len(selected)<n:
            cNode = queue.popleft()
            if cNode == self.tail or cNode in selected:
                continue
            selected.add(cNode)
            queue.extend(cNode.children)
        selected = list(self.pop(selected))
        return (selected * ((n // len(selected)) + 1))[:n]
    
    def viz(self):
        G = nx.DiGraph()
        G.add_node(self.head)
        G.add_node(self.tail)
        for node in self.nodes:
            G.add_node(node)
        
        for node in [self.head] + list(self.nodes):
            for child in node.children:
                G.add_edge(node, child)
        
        pos = nx.spring_layout(G)
        pos[self.head] = (1, 1)
        pos[self.tail] = (0, 0)
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, arrowsize=20, font_size=8)
        
        plt.title("Pool Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

A = Sample([1,2,3], name="A 1,2,3")
B = Sample([1,2,4], name="B 1,2,4")
C = Sample([5,5,5], name="C 5,5,5")
print(A,B,C)
pass

pool = Pool()
pass
pool.add(A)
pass
pool.add(B)
pass
pool.add(C)
pass
pool.viz()

pass