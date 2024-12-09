import numpy as np

# Fitness Function
def fitness_function(green_time, vehicle_count):
    """
    Calculate the remaining vehicles after a green light duration.

    Parameters:
        green_time: Green light duration (in seconds).
        vehicle_count: Number of vehicles waiting at the signal.

    Returns:
        Remaining vehicles after the green time.
    """
    vehicles_passed = (green_time // 4) * 2  # 2 vehicles pass every 4 seconds
    remaining_vehicles = max(0, vehicle_count - vehicles_passed)  # Vehicles still in queue
    return remaining_vehicles

# PSO Optimization for Green Light Duration
def pso_optimize_green(vehicle_count, num_particles=15, num_iterations=50):
    """
    Optimize the green light duration for a traffic signal using PSO.

    Parameters:
        vehicle_count: Number of vehicles waiting at the signal.
        num_particles: Number of candidate solutions (particles).
        num_iterations: Number of optimization iterations.

    Returns:
        Optimized green light duration (in seconds).
    """
    particles = np.arange(5, 61, 5)[:num_particles]  # Green times: 5, 10, ..., 60 seconds
    pBest = particles.copy()
    gBest = particles[0]

    for _ in range(num_iterations):
        for i in range(len(particles)):
            current_fitness = fitness_function(particles[i], vehicle_count)
            if current_fitness < fitness_function(pBest[i], vehicle_count):
                pBest[i] = particles[i]
            if current_fitness < fitness_function(gBest, vehicle_count):
                gBest = particles[i]

            if fitness_function(gBest, vehicle_count) == 0:
                return gBest

    return gBest

# Traffic Light Optimization Algorithm
def traffic_light_algorithm(vehicle_data):
    """
    Optimize the green and red light timings for a junction with multiple roads.

    Parameters:
        vehicle_data: List of dictionaries, where each dictionary contains:
            - 'index': Road index (e.g., 'Road 1').
            - 'vehicles': Number of vehicles waiting at the signal.
            - 'emergency': Boolean indicating if there is an emergency vehicle.

    Returns:
        A list of dictionaries for each road, sorted by priority, with:
            - 'index': Road index.
            - 'green_time': Optimized green light duration.
            - 'red_time': Calculated red light duration.
            - 'priority': Priority order (1 = highest priority).
    """
    # Separate roads with emergency vehicles and those without
    emergency_roads = [(road['index'], road['vehicles']) for road in vehicle_data if road['emergency']]
    normal_roads = [(road['index'], road['vehicles']) for road in vehicle_data if not road['emergency']]

    # Sort roads with emergency vehicles first, followed by roads with highest congestion
    sorted_roads = sorted(emergency_roads, key=lambda x: x[1], reverse=True) + \
                   sorted(normal_roads, key=lambda x: x[1], reverse=True)

    green_times = {}
    red_times = {}
    priority_list = []

    # Initialize red times for all roads
    for road in vehicle_data:
        red_times[road['index']] = 0

    # Calculate green and red timings based on priority
    for priority, (road_idx, vehicle_count) in enumerate(sorted_roads, start=1):
        green_time = pso_optimize_green(vehicle_count)
        green_times[road_idx] = green_time
        priority_list.append({'index': road_idx, 'green_time': green_time, 'priority': priority})

        # Update red light timings for other roads
        for road in vehicle_data:
            if road['index'] != road_idx:
                red_times[road['index']] += green_time

    # Merge results into a single list
    results = []
    for entry in priority_list:
        road_idx = entry['index']
        results.append({
            'index': road_idx,
            'green_time': entry['green_time'],
            'red_time': red_times[road_idx],
            'priority': entry['priority']
        })

    return results
