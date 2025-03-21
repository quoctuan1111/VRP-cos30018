import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from typing import List, Dict
from src.models.route import Route

class RouteVisualizer:
    def __init__(self, root: tk.Tk, cities_coordinates: dict):
        self.root = root
        self.root.title("VRP Route Visualization")
        self.cities_coordinates = cities_coordinates
        
        # Create main frame with better padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        
        # Create toolbar frame
        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        
        # Create info panel
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Route Information", padding="5")
        self.info_text = tk.Text(self.info_frame, height=8, width=60)
        self.route_listbox = tk.Listbox(self.info_frame, height=5)
        
        # Layout using grid
        self.toolbar_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.toolbar.grid(row=0, column=0, sticky="w")
        
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.info_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)
        self.info_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.route_listbox.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure weights
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.info_frame.grid_columnconfigure(0, weight=3)
        self.info_frame.grid_columnconfigure(1, weight=1)
        
        # Store routes for later reference
        self.routes = []

    def plot_routes(self, routes: List[Route]):
        """Plot routes with improved styling"""
        self.routes = routes
        
        # Schedule the actual plotting in the main thread
        self.root.after(0, self._do_plot_routes)

    def _do_plot_routes(self):
        """Perform the actual route plotting in the main thread"""
        self.ax.clear()
        
        # Use a better color scheme
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.routes)))
        
        # Plot cities with improved markers
        self._plot_cities()
        
        # Plot routes with better styling
        self._plot_route_paths(colors)
        
        # Customize plot appearance
        self.ax.set_title("Vehicle Routing Solution", pad=20, fontsize=14)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_aspect('equal')
        
        # Add route selection handler
        self.route_listbox.delete(0, tk.END)
        for i, route in enumerate(self.routes):
            self.route_listbox.insert(tk.END, f"Route {route.vehicle_id}")
            
        self.route_listbox.bind('<<ListboxSelect>>', self._on_route_select)
        
        # Update canvas
        self.canvas.draw()
        
        # Update info panel
        self.update_info(self.routes)

    def _plot_cities(self):
        """Plot cities with improved styling"""
        cities = set()
        for route in self.routes:
            for loc in route.locations:
                cities.add(loc.city_name)
        
        for city in cities:
            x, y = self.cities_coordinates[city]
            if city == "WAREHOUSE":
                self.ax.scatter(x, y, c='red', s=200, marker='*', zorder=5)
                self.ax.annotate('WAREHOUSE', (x, y), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=12, fontweight='bold')
            else:
                self.ax.scatter(x, y, c='lightgray', s=100, edgecolor='black', zorder=4)
                self.ax.annotate(city, (x, y), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8)

    def _plot_route_paths(self, colors):
        """Plot route paths with improved styling"""
        for route, color in zip(self.routes, colors):
            route_coordinates = []
            for loc in route.locations:
                route_coordinates.append(self.cities_coordinates[loc.city_name])
            
            route_coordinates = np.array(route_coordinates)
            self.ax.plot(route_coordinates[:, 0], route_coordinates[:, 1], 
                        c=color, linewidth=2, label=f"Route {route.vehicle_id}",
                        zorder=3)

    def _on_route_select(self, event):
        """Handle route selection"""
        selection = self.route_listbox.curselection()
        if selection:
            route = self.routes[selection[0]]
            self.update_route_details(route)

    def update_route_details(self, route: Route):
        """Update info panel with selected route details"""
        self.info_text.delete('1.0', tk.END)
        info = f"Route Details: {route.vehicle_id}\n"
        info += "=" * 50 + "\n"
        info += f"Number of Parcels: {len(route.parcels)}\n"
        info += f"Total Distance: {route.total_distance:.2f} km\n"
        info += f"Total Cost: ${route.total_cost:.2f}\n"
        info += f"Vehicle Capacity: {route.vehicle_capacity:.2f} kg\n"
        info += f"Current Load: {route.get_total_weight():.2f} kg\n"
        info += f"Capacity Utilization: {(route.get_total_weight()/route.vehicle_capacity)*100:.1f}%\n"
        self.info_text.insert(tk.END, info)

    def update_info(self, routes: List[Route]):
        """Update overall solution information"""
        total_cost = sum(route.total_cost for route in routes)
        total_distance = sum(route.total_distance for route in routes)
        total_parcels = sum(len(route.parcels) for route in routes)
        
        info = "Solution Summary\n"
        info += "=" * 50 + "\n"
        info += f"Total Routes: {len(routes)}\n"
        info += f"Total Distance: {total_distance:.2f} km\n"
        info += f"Total Cost: ${total_cost:.2f}\n"
        info += f"Total Parcels Delivered: {total_parcels}\n"
        
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert(tk.END, info)
