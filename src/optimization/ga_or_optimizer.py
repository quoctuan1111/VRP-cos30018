import time
import copy
import random
from typing import List, Dict, Tuple, Optional

from src.models.route import Route
from src.optimization.base_optimizer import BaseOptimizer
from src.optimization.ga_optimizer import GAOptimizer
from src.optimization.or_ml_optimizer import ORToolsMLOptimizer
from src.visualization.fitness_plotter import FitnessPlotter

class GAOROptimizer(BaseOptimizer):
    """
    GA + OR-Tools optimizer without fitness modification.
    This is Method 3 in the four-method system.
    
    This combines the Genetic Algorithm with OR-Tools solutions
    as part of the initial population but keeps the original
    GA fitness function.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the GA + OR-Tools optimizer.
        
        Args:
            data_processor: DataProcessor containing problem data
        """
        super().__init__(data_processor)
        
        # Create individual optimizers
        self.ga_optimizer = GAOptimizer(data_processor)
        self.or_optimizer = ORToolsMLOptimizer(data_processor)
        
        # GA parameters - can be different from pure GA
        self.population_size = 40
        self.generations = 80
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism_count = 5
        
        # How many OR-Tools solutions to include
        self.or_tools_solutions_count = 10
        
        # Fitness plotter for tracking evolution
        self.fitness_plotter = FitnessPlotter()
        self._running = True
    
    def optimize(self) -> List[Route]:
        """
        Execute the GA + OR-Tools optimization.
        
        Returns:
            List[Route]: The best solution found
        """
        print("\nStarting GA + OR-Tools (without fitness modification) optimization...")
        start_time = time.time()
        
        # First, get OR-Tools solutions
        print("Getting OR-Tools solutions for initial population...")
        or_solutions = self._get_or_tools_solutions()
        
        # Initialize population with mix of OR-Tools and random solutions
        population = self._initialize_hybrid_population(or_solutions)
        
        # Evaluate initial population
        fitness_scores = [self.calculate_fitness(solution) for solution in population]
        
        # Track best solution
        best_idx = fitness_scores.index(max(fitness_scores))
        self.best_solution = copy.deepcopy(population[best_idx])
        self.best_fitness = fitness_scores[best_idx]
        
        # Record initial fitness data
        self.fitness_plotter.add_generation_data(0, fitness_scores, self.best_fitness)
        
        print(f"Initial best fitness: {self.best_fitness:.4f}")
        
        # Main GA loop - similar to pure GA but with hybrid population
        for generation in range(self.generations):
            if not self._running:
                print("Optimization stopped by user.")
                break
                
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best solutions
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
            for i in range(self.elitism_count):
                new_population.append(copy.deepcopy(population[sorted_indices[i]]))
            
            # Crossover and Mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = selected[random.randint(0, len(selected) - 1)]
                parent2 = selected[random.randint(0, len(selected) - 1)]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = copy.deepcopy(parent1)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring = self._mutate(offspring)
                
                # Repair solution if needed
                offspring = self._repair_solution(offspring)
                
                # Add to new population
                new_population.append(offspring)
            
            # Update population
            population = new_population[:self.population_size]
            
            # Evaluate new population
            fitness_scores = [self.calculate_fitness(solution) for solution in population]
            
            # Update best solution
            current_best_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[current_best_idx] > self.best_fitness:
                self.best_solution = copy.deepcopy(population[current_best_idx])
                self.best_fitness = fitness_scores[current_best_idx]
                print(f"Generation {generation+1}: New best fitness: {self.best_fitness:.4f}")
            
            # Record fitness data for this generation
            self.fitness_plotter.add_generation_data(generation + 1, fitness_scores, self.best_fitness)
            
            # Progress reporting
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation+1}/{self.generations} completed. Best fitness: {self.best_fitness:.4f}")
        
        # Finalize and report
        elapsed_time = time.time() - start_time
        print(f"\nGA + OR-Tools optimization completed in {elapsed_time:.2f} seconds.")
        print(f"Final best fitness: {self.best_fitness:.4f}")
        
        evaluation = self.evaluate_solution(self.best_solution)
        print(f"Parcels delivered: {evaluation['parcels_delivered']}")
        print(f"Total cost: ${evaluation['total_cost']:.2f}")
        
        # Save fitness evolution plot
        self.fitness_plotter.plot(save_path='results/ga_or_fitness_evolution.png')
        
        return self.best_solution
    
    def _get_or_tools_solutions(self) -> List[List[Route]]:
        """
        Get OR-Tools solutions for the initial population.
        
        Returns:
            List of solutions from OR-Tools
        """
        # Use OR-Tools optimizer to generate solutions
        # We just need to generate a few solutions for seeding
        or_solutions = self.or_optimizer._generate_or_tools_solutions(num_variations=3)
        
        # If we didn't get enough, duplicate and modify them
        if len(or_solutions) < self.or_tools_solutions_count:
            original_count = len(or_solutions)
            for i in range(self.or_tools_solutions_count - original_count):
                if not or_solutions:
                    break
                    
                # Clone and slightly modify a random solution
                source_solution = random.choice(or_solutions)
                modified = self._create_modified_solution(source_solution)
                or_solutions.append(modified)
        
        return or_solutions
    
    def _create_modified_solution(self, solution: List[Route]) -> List[Route]:
        """
        Create a slightly modified version of a solution.
        
        Args:
            solution: Original solution
            
        Returns:
            Modified solution
        """
        modified = copy.deepcopy(solution)
        
        # Skip if empty
        if not modified:
            return modified
        
        # Apply random modifications
        modification_type = random.choice(['swap', 'change_truck', 'reorder'])
        
        if modification_type == 'swap' and len(modified) >= 2:
            # Swap some parcels between two routes
            idx1, idx2 = random.sample(range(len(modified)), 2)
            route1, route2 = modified[idx1], modified[idx2]
            
            if route1.parcels and route2.parcels:
                # Swap up to 20% of parcels
                swap_count = max(1, min(len(route1.parcels), len(route2.parcels)) // 5)
                
                # Select random parcels to swap
                parcels1 = random.sample(route1.parcels, min(swap_count, len(route1.parcels)))
                parcels2 = random.sample(route2.parcels, min(swap_count, len(route2.parcels)))
                
                # Remove swapped parcels
                route1.parcels = [p for p in route1.parcels if p not in parcels1]
                route2.parcels = [p for p in route2.parcels if p not in parcels2]
                
                # Add swapped parcels
                route1.parcels.extend(parcels2)
                route2.parcels.extend(parcels1)
                
                # Recalculate routes
                self._recalculate_route(route1)
                self._recalculate_route(route2)
                
        elif modification_type == 'change_truck':
            # Change truck type for a random route
            if modified:
                idx = random.randint(0, len(modified) - 1)
                route = modified[idx]
                
                truck_types = ['9.6', '12.5', '16.5']
                current_type = route.vehicle_id.split('_')[1]
                
                if current_type in truck_types:
                    truck_types.remove(current_type)
                    new_type = random.choice(truck_types)
                    
                    # Check if new truck can handle the load
                    total_weight = route.get_total_weight()
                    new_capacity = self.data_processor.truck_specifications[new_type]['weight_capacity']
                    
                    if total_weight <= new_capacity:
                        # Update truck type
                        route.vehicle_id = f"V_{new_type}_{route.vehicle_id.split('_')[-1]}"
                        route.vehicle_capacity = new_capacity
                        
                        # Recalculate cost
                        self._recalculate_route(route)
        
        elif modification_type == 'reorder':
            # Reorder a random route
            if modified:
                idx = random.randint(0, len(modified) - 1)
                route = modified[idx]
                
                if route.parcels and len(route.parcels) > 1:
                    # Shuffle parcels
                    random.shuffle(route.parcels)
                    
                    # Rebuild route
                    locations = [route.locations[0]]  # Start with warehouse
                    
                    for parcel in route.parcels:
                        if parcel.source and parcel.source not in locations:
                            locations.append(parcel.source)
                        if parcel.destination and parcel.destination not in locations:
                            locations.append(parcel.destination)
                    
                    if locations[-1] != route.locations[0]:
                        locations.append(route.locations[0])  # End with warehouse
                    
                    route.locations = locations
                    
                    # Recalculate route
                    self._recalculate_route(route)
        
        return modified
    
    def _recalculate_route(self, route: Route):
        """
        Recalculate route distances and costs.
        
        Args:
            route: Route to recalculate
        """
        route.calculate_total_distance()
        truck_type = route.vehicle_id.split('_')[1]
        route.total_cost = route.total_distance * \
            self.data_processor.truck_specifications[truck_type]['cost_per_km']
    
    def _initialize_hybrid_population(self, or_solutions: List[List[Route]]) -> List[List[Route]]:
        """
        Initialize population with a mix of OR-Tools and random solutions.
        
        Args:
            or_solutions: OR-Tools solutions
            
        Returns:
            Initial population
        """
        population = []
        
        # Add OR-Tools solutions
        for solution in or_solutions[:min(len(or_solutions), self.or_tools_solutions_count)]:
            population.append(solution)
        
        # Fill the rest with GA solutions
        while len(population) < self.population_size:
            # Use GA's random initialization
            solution = self.ga_optimizer._create_randomized_solution()
            population.append(solution)
        
        return population
    
    def _selection(self, population: List[List[Route]], fitness_scores: List[float]) -> List[List[Route]]:
        """
        Tournament selection for parent selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for the population
            
        Returns:
            Selected parents
        """
        # Use GA's selection method
        return self.ga_optimizer._selection(population, fitness_scores)
    
    def _crossover(self, parent1: List[Route], parent2: List[Route]) -> List[Route]:
        """
        Route-based crossover operator.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            Offspring solution
        """
        # Use GA's crossover method
        return self.ga_optimizer._crossover(parent1, parent2)
    
    def _mutate(self, solution: List[Route]) -> List[Route]:
        """
        Apply mutation operators to a solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        # Use GA's mutation method
        return self.ga_optimizer._mutate(solution)
    
    def _repair_solution(self, solution: List[Route]) -> List[Route]:
        """
        Repair infeasible solutions.
        
        Args:
            solution: Solution to repair
            
        Returns:
            Repaired solution
        """
        # Use GA's repair method
        return self.ga_optimizer._repair_solution(solution)
    
    def stop(self):
        """Stop the optimization process"""
        self._running = False