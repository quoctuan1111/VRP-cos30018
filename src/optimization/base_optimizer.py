from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.models.route import Route

class BaseOptimizer(ABC):
    """
    Abstract base class for all VRP optimization methods.
    Defines the common interface that all optimizers must implement.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the optimizer with a data processor.
        
        Args:
            data_processor: DataProcessor instance containing problem data
        """
        self.data_processor = data_processor
        self.best_solution: List[Route] = []
        self.best_fitness = 0.0
        
    @abstractmethod
    def optimize(self) -> List[Route]:
        """
        Execute the optimization algorithm and return the best solution found.
        
        Returns:
            List[Route]: The best solution found
        """
        pass
    
    def evaluate_solution(self, routes: List[Route]) -> Dict[str, Any]:
        """
        Evaluate a solution with various metrics.
        
        Args:
            routes: List of Routes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Count total parcels delivered
        parcels_delivered = sum(len(route.parcels) for route in routes if route.is_feasible)
        
        # Calculate total cost
        total_cost = sum(route.total_cost for route in routes if route.is_feasible)
        
        # Calculate total distance
        total_distance = sum(route.total_distance for route in routes if route.is_feasible)
        
        # Count infeasible routes
        infeasible_routes = sum(1 for route in routes if not route.is_feasible)
        
        # Calculate load factors
        if routes:
            weights = [route.get_total_weight() for route in routes if route.is_feasible]
            capacities = [route.vehicle_capacity for route in routes if route.is_feasible]
            load_factors = [w/c for w, c in zip(weights, capacities) if c > 0]
            avg_load_factor = sum(load_factors) / len(load_factors) if load_factors else 0
        else:
            avg_load_factor = 0
        
        return {
            'parcels_delivered': parcels_delivered,
            'total_cost': total_cost,
            'total_distance': total_distance,
            'num_routes': len(routes),
            'infeasible_routes': infeasible_routes,
            'avg_load_factor': avg_load_factor
        }
    
    def calculate_fitness(self, solution: List[Route]) -> float:
        """
        Calculate fitness score for a solution with improved differentiation.
        """
        if not solution:
            return 0.0

        # Get solution metrics
        evaluation = self.evaluate_solution(solution)
        total_parcels = evaluation['parcels_delivered']
        total_cost = evaluation['total_cost']
        total_distance = sum(route.total_distance for route in solution)
        
        # Calculate normalized metrics with non-linear scaling
        if total_parcels > 0:
            # Exponential reward for more parcels delivered
            parcels_score = (total_parcels / len(self.data_processor.time_windows)) ** 1.5
            
            # Progressive penalty for cost
            cost_per_parcel = total_cost / total_parcels
            max_acceptable_cost = 5000  # Example threshold
            cost_score = 1.0 - min(1.0, (cost_per_parcel / max_acceptable_cost) ** 0.8)
            
            # Route efficiency (penalize too many or too few routes)
            optimal_routes = max(1, len(self.data_processor.time_windows) // 5)  # Assume average 5 parcels per route
            route_ratio = len(solution) / optimal_routes
            route_score = 1.0 - abs(1.0 - route_ratio) ** 0.5
            
            # Load balancing score
            weights = [route.get_total_weight() for route in solution]
            if weights:
                avg_weight = sum(weights) / len(weights)
                weight_variations = [abs(w - avg_weight) / avg_weight if avg_weight > 0 else 1.0 for w in weights]
                balance_score = 1.0 - min(1.0, sum(weight_variations) / len(weights))
            else:
                balance_score = 0.0
            
            # Distance efficiency
            avg_distance_per_parcel = total_distance / total_parcels
            max_acceptable_distance = 100  # Example threshold per parcel
            distance_score = 1.0 - min(1.0, (avg_distance_per_parcel / max_acceptable_distance) ** 0.7)
            
            # Combine scores with adjusted weights
            weights = {
                'parcels': 0.35,  # Increased importance of delivering parcels
                'cost': 0.25,    # Balanced cost consideration
                'routes': 0.15,  # Route count importance
                'balance': 0.15, # Load balancing importance
                'distance': 0.10 # Distance efficiency
            }
            
            fitness = (weights['parcels'] * parcels_score +
                      weights['cost'] * cost_score +
                      weights['routes'] * route_score +
                      weights['balance'] * balance_score +
                      weights['distance'] * distance_score)
            
            # Apply penalty for infeasible solutions
            infeasible_routes = sum(1 for route in solution if not route.is_feasible)
            if infeasible_routes > 0:
                penalty = (infeasible_routes / len(solution)) ** 2
                fitness *= (1.0 - penalty)
            
            return max(0.0, min(1.0, fitness))
        return 0.0
    
    def get_best_solution(self) -> List[Route]:
        """Return the best solution found during optimization"""
        return self.best_solution
    
    def get_best_fitness(self) -> float:
        """Return the fitness of the best solution"""
        return self.best_fitness