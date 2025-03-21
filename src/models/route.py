from typing import List, Optional
from .location import Location
from .parcel import Parcel
from ..data.data_processor import DataProcessor

class Route:
    def __init__(self, vehicle_id: str, locations: List[Location], parcels: List[Parcel], 
                 data_processor: Optional[DataProcessor] = None, vehicle_capacity: float = 50000.0):
        self.vehicle_id = vehicle_id
        self.vehicle_capacity = vehicle_capacity
        self.data_processor = data_processor
        self.is_feasible = True
        self.violation_reason = ""
        self.total_distance = 0.0
        self.total_cost = 0.0
        
        # Validate and set locations
        if not locations:
            raise ValueError("Route must have at least one location")
            
        # If data_processor is provided, check warehouse location
        if data_processor is not None:
            if locations[0].city_name != data_processor.warehouse_location:
                locations.insert(0, Location(city_name=data_processor.warehouse_location))
            if locations[-1].city_name != data_processor.warehouse_location:
                locations.append(Location(city_name=data_processor.warehouse_location))
        
        self.locations = locations
        self.parcels = parcels
        
        # Calculate initial distance
        self.calculate_total_distance()

    def calculate_total_distance(self) -> float:
        """Calculate total route distance"""
        total = 0.0
        if self.data_processor and len(self.locations) > 1:
            for i in range(len(self.locations) - 1):
                source_idx = self.data_processor.city_to_idx.get(self.locations[i].city_name, 0)
                dest_idx = self.data_processor.city_to_idx.get(self.locations[i + 1].city_name, 0)
                total += self.data_processor.distance_matrix[source_idx][dest_idx]
        self.total_distance = total
        return total

    def get_total_weight(self) -> float:
        """Calculate total weight of all parcels"""
        return sum(parcel.weight for parcel in self.parcels)

    def __str__(self) -> str:
        return (f"Route {self.vehicle_id}: {len(self.parcels)} parcels, "
                f"Distance: {self.total_distance:.2f} km, "
                f"Cost: ${self.total_cost:.2f}")