

import networkx as nx
import random
import numpy as np
from matplotlib import pyplot as plt

class Fourmi:
    def __init__(self, alpha=1, beta=3):
        self.graph = None
        self.colors = {}
        self.depart = None
        self.visited = []
        self.unvisited = []
        self.alpha = alpha
        self.beta = beta
        self.distance = 0 
        self.nombre_collisions = 0 
        self.couleurs_disponibles = []
        self.couleurs_attribuees = {}

    def initialiser(self, graphe, couleurs, depart=None):
        self.couleurs_disponibles = sorted(couleurs.copy())
        keys = [n for n in noeuds_graphe]
        self.couleurs_attribuees = {key: None for key in keys}
        if depart is None:
            self.depart = random.choice(noeuds_graphe)
        else:
            self.depart = depart
        self.visited = []
        self.unvisited = noeuds_graphe.copy()
        if len(self.visited) == 0:
            self.attribuer_couleur(self.depart, self.couleurs_disponibles[0])
        return self

    def attribuer_couleur(self, noeud, couleur):
        self.couleurs_attribuees[noeud] = couleur
        self.visited.append(noeud)
        self.unvisited.remove(noeud)

    def colorier(self):
        len_unvisited = len(self.unvisited)
        tabu_colors = []
        for i in range(len_unvisited):
            suivant = self.prochain_candidat()
            tabu_colors = []
            for j in range(nombre_noeuds):
                if adj_matrix[suivant, j] == 1:
                    tabu_colors.append(self.couleurs_attribuees[j])
            for k in self.couleurs_disponibles:
                if k not in tabu_colors:
                    self.attribuer_couleur(suivant, k)
                    break
        self.distance = len(set(self.couleurs_attribuees.values()))

    def dsat(self, noeud=None):
        if noeud is None:
            noeud = self.depart
        col_voisins = []
        for j in range(nombre_noeuds):
            if adj_matrix[noeud, j] == 1:
                col_voisins.append(self.couleurs_attribuees[j])
        return len(set(col_voisins))

    def pheromone_level(self, noeud, adj_noeud):
        return phero_matrix[noeud, adj_noeud]

    def prochain_candidat(self):
        if len(self.unvisited) == 0:
            candidat = None
        elif len(self.unvisited) == 1:
            candidat = self.unvisited[0]
        else:
            max_value = 0
            valeurs_heuristiques = []
            candidats = []
            candidats_disponibles = []
            for j in self.unvisited:
                valeurs_heuristiques.append((self.pheromone_level(self.depart, j) ** self.alpha) * (self.dsat(j) ** self.beta))
                candidats.append(j)
            max_value = max(valeurs_heuristiques)
            for i in range(len(candidats)):
                if valeurs_heuristiques[i] >= max_value:
                    candidats_disponibles.append(candidats[i])
            candidat = random.choice(candidats_disponibles)
        self.depart = candidat
        return candidat

    def trace_phero(self):
        phero_trail = np.zeros((nombre_noeuds, nombre_noeuds), float)
        for i in noeuds_graphe:
            for j in noeuds_graphe:
                if self.couleurs_attribuees[i] == self.couleurs_attribuees[j]:
                    phero_trail[i, j] = 1
        return phero_trail

    def collisions(self):
        collisions = 0
        for key in self.couleurs_attribuees:
            noeud = key
            col = self.couleurs_attribuees[key]
            for j in range(nombre_noeuds):
                if adj_matrix[noeud, j] == 1 and self.couleurs_attribuees[j] == col:
                    collisions = collisions + 1
        return collisions

def dessiner_graphe(graphe, couleurs):
    pos = nx.spring_layout(graphe)
    valeurs = [couleurs.get(noeud, 'blue') for noeud in graphe.nodes()]
    nx.draw(graphe, pos, with_labels=True, node_color=valeurs, edge_color='black', width=1, alpha=0.7)
    plt.show()

def init_couleurs(graphe):
    couleurs = []
    grundy = len(nx.degree_histogram(graphe))
    for c in range(grundy):
        couleurs.append(c)
    return couleurs

def init_pheromones(graphe):
    phero_matrix = np.ones((nombre_noeuds, nombre_noeuds), float)
    for noeud in graphe:
        for adj_noeud in graphe.neighbors(noeud):
            phero_matrix[noeud, adj_noeud] = 0
    return phero_matrix

def matrice_adjacence(graphe):
    adj_matrix = np.zeros((nombre_noeuds, nombre_noeuds), int)
    for noeud in noeuds_graphe:
        for adj_noeud in graphe.neighbors(noeud):
            adj_matrix[noeud, adj_noeud] = 1
    return adj_matrix

def creer_colonie():
    fourmis = []
    fourmis.extend([Fourmi().initialiser(graphe, couleurs) for i in range(nombre_fourmis)])
    return fourmis

def mis_a_jour_pheromone():
    for noeud in noeuds_graphe:
        for adj_noeud in noeuds_graphe:
            phero_matrix[noeud, adj_noeud] = phero_matrix[noeud, adj_noeud] * (1 - phero_decay)

def mis_a_jour():
    global phero_matrix
    meilleur_cout = 0
    fourmi_elite = None
    for fourmi in fourmis:
        if meilleur_cout == 0:
            meilleur_cout = fourmi.distance
            fourmi_elite = fourmi
        elif fourmi.distance < meilleur_cout:
            meilleur_cout = fourmi.distance
            fourmi_elite = fourmi
    elite_phero_matrix = fourmi_elite.trace_phero()
    phero_matrix = phero_matrix + elite_phero_matrix
    return fourmi_elite.distance, fourmi_elite.couleurs_attribuees

def executer(graph=None, nb_fourmis=10, iter=10, a=1, b=3, decay=0.8, nb_noeuds=20, prob_aretes=0.3):
    global graphe, nombre_fourmis, nombre_iterations, alpha, beta, phero_decay, adj_matrix, phero_matrix, couleurs, fourmis

    if graph is None:
        graph = nx.erdos_renyi_graph(nb_noeuds, prob_aretes)

    graphe = graph 
    nombre_fourmis = nb_fourmis
    nombre_iterations = iter
    alpha = a
    beta = b
    phero_decay = decay

    global nombre_noeuds, noeuds_graphe
    nombre_noeuds = nx.number_of_nodes(graphe)
    noeuds_graphe = sorted(graphe.nodes())
    adj_matrix = matrice_adjacence(graphe)
    couleurs = init_couleurs(graphe)
    phero_matrix = init_pheromones(graphe)

    meilleure_solution = {}
    cout_final = 0
    for i in range(nombre_iterations):
        fourmis = creer_colonie()
        for fourmi in fourmis:
            fourmi.colorier()
        mis_a_jour_pheromone()
        cout_elite, solution_elite = mis_a_jour()
        if cout_final == 0 or cout_elite < cout_final:
            cout_final = cout_elite
            meilleure_solution = solution_elite

    return cout_final, meilleure_solution

cout, solution = executer(nb_fourmis=10, iter=100, a=1, b=3, decay=0.8, nb_noeuds=20, prob_aretes=0.3)
dessiner_graphe(graphe, solution)
print("Cout final:", cout)
print("Solution finale:", solution)