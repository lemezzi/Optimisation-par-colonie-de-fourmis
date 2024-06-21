

import math
import random
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


class AntColonyTsp:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.distance = 0.0

        def _select_node(self):
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, colony_size=10, elitist_weight=1.0, min_scaling_factor=0.001, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, num_cities=10, nodes=None, labels=None):
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_cities = num_cities
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                initial_pheromone)
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def simulate(self, progress_var):
        distance_evolution = []  # Liste pour stocker l'évolution de la distance

        for step in range(self.steps):
            progress_var.set(step / self.steps * 100)  # Update progress bar
            root.update_idletasks()
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            distance_evolution.append(self.global_best_distance)  # Ajoutez la distance globale à la liste d'évolution

            # Mise à jour de l'affichage
            update_plot(self.nodes, self.global_best_tour, self.labels)
            root.update()  # Mise à jour de la fenêtre Tkinter

            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

        # Tracer l'évolution de la distance
        plot_distance_evolution(distance_evolution)

def clear_plot():
    # Delete all elements except the background image
    for item in canvas.find_all():
        if item != map_photo:
            canvas.delete(item)

def update_plot(nodes, tour, labels):
    background_id = canvas.find_withtag("background")  # Get the ID of the background image
    
    for item in canvas.find_all():
        if item not in background_id:  # Skip deleting the background image
            canvas.delete(item)  # Clear the plot before updating

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            canvas.create_line(x1, y1, x2, y2, fill="#dddddd")

    total_distance = 0.0
    for i in range(len(tour)):
        x1, y1 = nodes[tour[i]]
        x2, y2 = nodes[tour[(i + 1) % len(tour)]]

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance

        canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
        canvas.create_oval(x1 - 5, y1 - 5, x1 + 5, y1 + 5, fill="red", outline="blue")
        canvas.create_text(x1 + 10, y1, text=str(labels[tour[i]]), font=("Arial", 8, "bold"), fill="blue")

    distance_value.set(round(total_distance, 2))
    root.update_idletasks()
    root.update()

def show_final_distance(final_distance):
    # Créer une nouvelle fenêtre Tkinter
    final_distance_window = tk.Toplevel(root)
    final_distance_window.title("Distance finale")

    # Créer un label pour afficher la distance finale
    final_distance_label = tk.Label(final_distance_window, text=f"Distance finale : {final_distance}")
    final_distance_label.pack(padx=20, pady=10)


def reset_simulation():
   
    
   
    
   # Réinitialiser les valeurs des champs d'entrée
    colony_size_entry.delete(0, tk.END)
    colony_size_entry.insert(tk.END, '20')
    steps_entry.delete(0, tk.END)
    steps_entry.insert(tk.END, '500')
    num_cities_entry.delete(0, tk.END)
    num_cities_entry.insert(tk.END, '50')
    random_nodes_var.set(False)
    
    background_id = canvas.find_withtag("background")  # Get the ID of the background image
    
    for item in canvas.find_all():
        if item not in background_id:  # Skip deleting the background image
            canvas.delete(item)  # Clear the plot before updating

    manually_added_nodes.clear()




def update_num_cities_entry():
    num_cities_entry.delete(0, tk.END)
    num_cities_entry.insert(tk.END, str(len(manually_added_nodes)))

def start_simulation(progress_var):
    
    
    
    # Vérifier si l'utilisateur a choisi de générer des nœuds aléatoires
    if add_nodes_choice.get() == "manual" and len(manually_added_nodes) == 0:
        # Afficher un message d'erreur
        tk.messagebox.showerror("Error", "Please add nodes manually or select 'Generate Random Nodes'.")
        return  # Arrêter la fonction si une erreur est détectée

    # Vérifier la validité des valeurs entrées
    try:
        colony_size = int(colony_size_entry.get())
        if colony_size <= 0:
            raise ValueError("Colony size must be a positive integer.")

        steps = int(steps_entry.get())
        if steps <= 0:
            raise ValueError("Number of steps must be a positive integer.")

        alpha = float(alpha_entry.get())
        beta = float(beta_entry.get())
        rho = float(rho_entry.get())
        if alpha <=0 or beta <0 or rho <= 0:
            raise ValueError("Alpha, Beta, and Rho must be positive values.")

        if add_nodes_choice.get() == "manual" and len(manually_added_nodes) < 2:
            raise ValueError("Please add at least 2 nodes manually.")

        num_cities = len(manually_added_nodes) if add_nodes_choice.get() == "manual" else int(num_cities_entry.get())
        if num_cities <= 1:
            raise ValueError("Number of cities must be at least 2  .")
            
        for x, y in manually_added_nodes:
            if not (0 <= x <= 600) or not (0 <= y <= 600):
                raise ValueError("Node coordinates must be within the canvas boundaries (0-600).")
    except ValueError as e:
        # Afficher un message d'erreur pour les valeurs incorrectes
        tk.messagebox.showerror("Error", str(e))
        return  # Arrêter la fonction si une erreur est détectée


    
    colony_size = int(colony_size_entry.get())
    steps = int(steps_entry.get())
    alpha = float(alpha_entry.get())
    beta = float(beta_entry.get())
    rho = float(rho_entry.get())
    
    if add_nodes_choice.get() == "manual":
        num_cities = len(manually_added_nodes)
        nodes = manually_added_nodes
        # Mettre à jour le champ d'entrée du nombre de villes
        num_cities_entry.delete(0, tk.END)
        num_cities_entry.insert(tk.END, str(num_cities))
    else:
        num_cities = int(num_cities_entry.get())
        nodes = [(random.uniform(10, 590), random.uniform(10, 590)) for _ in range(num_cities)]

    acs = AntColonyTsp(colony_size=colony_size, steps=steps, num_cities=num_cities, nodes=nodes,
                           alpha=alpha, beta=beta, rho=rho)
    acs.simulate(progress_var)


def plot_distance_evolution(distance_evolution):
    root = tk.Tk()
    root.title("Distance Evolution Graph")
    root.configure(bg="#f5f5f5")  # Couleur de fond
    style = ttk.Style(root)
    style.theme_use('clam')  # Choisir un thème pour les widgets
    style.configure("TButton", background="#007bff", foreground="#ffffff")  # Style des boutons

    fig, ax = plt.subplots()
    ax.plot(range(len(distance_evolution)), distance_evolution)
    ax.set_xlabel('Steps', color="#333333")  # Couleur du texte
    ax.set_ylabel('Distance', color="#333333")  
    ax.set_title('Evolution of Distance over Steps', color="#333333")  
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    button = ttk.Button(root, text="Close", command=root.destroy)
    button.pack(pady=10)

    root.mainloop()


def add_city(event):
    x, y = event.x, event.y
    manually_added_nodes.append((x, y))
    canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="green")
    
    


if __name__ == '__main__':
    manually_added_nodes = []  

    root = tk.Tk()
    root.title("TSP Simulation")
    root.configure(bg="#f5f5f5")  # Background color
    style = ttk.Style(root)
    style.theme_use('clam')  # Choose a theme for the widgets
    style.configure("TLabel", background="#f5f5f5", foreground="#333333")  # Label style
    style.configure("TButton", background="#007bff", foreground="#ffffff")  # Button style

    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    canvas = tk.Canvas(main_frame, width=600, height=600, bg="#ffffff")  # Canvas background color
    canvas.grid(column=0, row=0, columnspan=2, padx=5, pady=5)  

    # Load the map image
    map_image = Image.open("back.png")  # Change "map.png" to your image file path
    
    # Resize the image to fit within the canvas
    canvas_width = canvas.winfo_reqwidth()
    canvas_height = canvas.winfo_reqheight()
    map_width, map_height = map_image.size
    aspect_ratio = map_width / map_height
    
    if canvas_width / canvas_height > aspect_ratio:
        new_width = int(canvas_height * aspect_ratio)
        new_height = canvas_height
    else:
        new_width = canvas_width
        new_height = int(canvas_width / aspect_ratio)
    
    map_image = map_image.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Calculate coordinates to center the image on the canvas
    x = (canvas_width - new_width) / 2
    y = (canvas_height - new_height) / 2
    
    global map_photo
    map_photo = ImageTk.PhotoImage(map_image)
    map_item_id = canvas.create_image(x, y, anchor=tk.NW, image=map_photo)
    canvas.itemconfig(map_item_id, tags=("background",))
    
    canvas.bind("<Button-1>", add_city)  

    distance_label = ttk.Label(main_frame, text="Distance parcourue:")
    distance_label.grid(column=0, row=6, sticky=tk.W, padx=5, pady=2)  
    distance_value = tk.StringVar()
    distance_display = ttk.Label(main_frame, textvariable=distance_value)
    distance_display.grid(column=1, row=6, sticky=tk.W, padx=5, pady=2)  

    colony_size_label = ttk.Label(main_frame, text="Colony Size:")
    colony_size_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=2)  
    colony_size_entry = ttk.Entry(main_frame, width=10)
    colony_size_entry.grid(column=1, row=1, padx=5, pady=2)  
    colony_size_entry.insert(tk.END, '20')

    steps_label = ttk.Label(main_frame, text="Steps:")
    steps_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=2)  
    steps_entry = ttk.Entry(main_frame, width=10)
    steps_entry.grid(column=1, row=2, padx=5, pady=2)  
    steps_entry.insert(tk.END, '500')

    num_cities_label = ttk.Label(main_frame, text="Number of Cities:")
    num_cities_label.grid(column=0, row=3, sticky=tk.W, padx=5, pady=2)  
    num_cities_entry = ttk.Entry(main_frame, width=10)
    num_cities_entry.grid(column=1, row=3, padx=5, pady=2)  
    num_cities_entry.insert(tk.END, '50')

    alpha_label = ttk.Label(main_frame, text="Alpha:")
    alpha_label.grid(column=0, row=9, sticky=tk.W, padx=5, pady=2)
    alpha_entry = ttk.Entry(main_frame, width=10)
    alpha_entry.grid(column=1, row=9, padx=5, pady=2)
    alpha_entry.insert(tk.END, '1.0')

    beta_label = ttk.Label(main_frame, text="Beta:")
    beta_label.grid(column=0, row=10, sticky=tk.W, padx=5, pady=2)
    beta_entry = ttk.Entry(main_frame, width=10)
    beta_entry.grid(column=1, row=10, padx=5, pady=2)
    beta_entry.insert(tk.END, '3.0')

    rho_label = ttk.Label(main_frame, text="Rho:")
    rho_label.grid(column=0, row=11, sticky=tk.W, padx=5, pady=2)
    rho_entry = ttk.Entry(main_frame, width=10)
    rho_entry.grid(column=1, row=11, padx=5, pady=2)
    rho_entry.insert(tk.END, '0.1')

    random_nodes_var = tk.BooleanVar()
    random_nodes_checkbox = ttk.Checkbutton(main_frame, text="Generate Random Nodes", variable=random_nodes_var)
    random_nodes_checkbox.grid(column=0, row=4, columnspan=2, padx=5, pady=2, sticky=tk.W)

    run_button = ttk.Button(main_frame, text="Run Simulation", command=lambda: start_simulation(progress_var))
    run_button.grid(column=0, row=5, columnspan=2, padx=5, pady=5)  
    
    add_nodes_choice = tk.StringVar(value="manual")

    manual_nodes_radio = ttk.Radiobutton(main_frame, text="Add Nodes Manually", variable=add_nodes_choice, value="manual", command=update_num_cities_entry)
    manual_nodes_radio.grid(column=0, row=4, padx=5, pady=2, sticky=tk.W)

    random_nodes_radio = ttk.Radiobutton(main_frame, text="Generate Random Nodes", variable=add_nodes_choice, value="random")
    random_nodes_radio.grid(column=1, row=4, padx=5, pady=2, sticky=tk.W)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100)
    progress_bar.grid(column=0, row=7, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
    
    reset_button = ttk.Button(main_frame, text="Reset Simulation", command=reset_simulation)
    reset_button.grid(column=0, row=8, columnspan=2, padx=5, pady=5)  

    root.mainloop()