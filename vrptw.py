from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
import os
import pandas as pd


def print_solution(data, manager, routing, solution, coordinates, instance_dir):
    """Saves solution to a CSV file."""
    time_dimension = routing.GetDimensionOrDie("Time")
    routes = []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = []
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_coordinates = []
        total_time = 0  # Reset the total time to 0 at the start of each route
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            node = manager.IndexToNode(index)
            route.append({
                'node': node,
                'time_min': solution.Min(time_var),
                'time_max': solution.Max(time_var),
                'coordinate': coordinates[node]  # Add the coordinate to the route
            })
            plan_output += (
                f"{node}"
                f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                " -> "
            )
            route_coordinates.append(coordinates[node])
            total_time += solution.Min(time_var)
            index = solution.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        node = manager.IndexToNode(index)
        route.append({
            'node': node,
            'time_min': solution.Min(time_var),
            'time_max': solution.Max(time_var),
            'coordinate': coordinates[node]
        })
        plan_output += (
            f"{node}"
            f" Time({solution.Min(time_var)},{solution.Max(time_var)})\n"
        )
        plan_output += f"Time of the route: {solution.Min(time_var)}min\n"
        print(plan_output)
        routes.append({
            'vehicle_id': vehicle_id,
            'route': route,
            'Time_of_the_route': plan_output
        })

    # Extract the instance number from the directory name
    cluster_id = instance_dir.split('_')[-1]

    # Create a DataFrame from the routes
    routes_df = pd.DataFrame(routes)
    solution_path = os.path.join('clusters', instance_dir, f'solution_{cluster_id}.csv')
    # Save the DataFrame to a CSV file
    routes_df.to_csv(solution_path, index=False)

    # Create a map
    m = folium.Map(location=route[0]['coordinate'], zoom_start=12)
    # Add a line for the route
    folium.PolyLine([node['coordinate'] for node in route], color="red", weight=2.5, opacity=1).add_to(m)
    # Add markers for each location in the route
    for node in route:
        folium.Marker(node['coordinate'],
                      popup=f"Address ID: {node['node']}, Time Window: ({node['time_min']}, {node['time_max']})").add_to(
            m)

    # Add a marker at the top of the map with a popup displaying the total number of IDs
    popup = folium.Popup(f"Total number of IDs: {len(route)}", max_width=300)
    folium.Marker(
        [max(node['coordinate'][0] for node in route), route[0]['coordinate'][1]],
        popup=popup
    ).add_to(m)
    # Save it as html
    route_map_path = os.path.join('clusters', instance_dir, f'route_map_{cluster_id}.html')
    m.save(route_map_path)



def main():
    """Solve the VRP with time windows."""
    # Get the list of instance directories
    instance_dirs = os.listdir('clusters')

    # Loop over each instance directory
    for instance_dir in instance_dirs:
        # Extract the instance number from the directory name
        cluster_id = instance_dir.split('_')[-1]

        # Load the data and distance matrix from the CSV files
        data_path = os.path.join('clusters', instance_dir,
                                 f'data_{cluster_id}.csv')
        distance_matrix_path = os.path.join('clusters', instance_dir,
                                            f'distance_matrix_{cluster_id}.csv')
        data = pd.read_csv(data_path)
        distance_matrix = pd.read_csv(distance_matrix_path)

        # Extract the necessary columns
        new_distance_matrix_list = distance_matrix.values.tolist()
        start_times = data['Von1'].tolist()
        end_times = data['Bis1'].tolist()
        latitudes = data['Latitude'].tolist()
        longitudes = data['Longitude'].tolist()

        # Combine start and end times to create time windows
        time_windows = list(zip(start_times, end_times))

        # Combine latitudes and longitudes to create coordinates
        coordinates = list(zip(latitudes, longitudes))

        def create_data_model():
            """Stores the data for the problem."""
            data = {}
            data["time_matrix"] = new_distance_matrix_list
            data["time_windows"] = time_windows
            data["num_vehicles"] = 1
            data["depot"] = 0
            return data



        # Instantiate the data problem.
        data = create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["time_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Time Windows constraint.
        time = "Time"
        routing.AddDimension(
            transit_callback_index,
            20000,  # allow waiting time
            288000,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time,
        )
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data["time_windows"]):
            if location_idx == data["depot"]:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        depot_idx = data["depot"]
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
            )

        # Instantiate route start and end times to produce feasible times.
        for i in range(data["num_vehicles"]):
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
       # routing_enums_pb2.LocalSearchMetaheuristic.A
        # Set the time limit to 5 minutes (300 seconds)
        search_parameters.time_limit.seconds = 500
        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            print_solution(data, manager, routing, solution, coordinates, instance_dir)
        else:
            # If no solution was found, save a message to the solutions folder
            cluster_id = instance_dir.split('_')[-1]
            solution_path = os.path.join('clusters', instance_dir,
                                         f'solution_{cluster_id}.txt')
            with open(solution_path, 'w') as f:
                f.write('Unable to find a solution within the time limit.')
            print('No solution found within the time limit.')


if __name__ == "__main__":
    main()
