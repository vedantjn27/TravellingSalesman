import streamlit as st 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import math
import random

st.set_page_config(layout="wide", page_title="TSP Solver")

st.title("Traveling Salesman Problem Solver")
st.write("""
This application solves the Traveling Salesman Problem using Dynamic Programming and provides 
an interactive visualization of the solution process.
""")

# Sidebar for inputs
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose input method:", ["Random Cities", "Manual Entry"])

# Random city generation
if input_method == "Random Cities":
    num_cities = st.sidebar.slider("Number of Cities", min_value=3, max_value=15, value=6)
    random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)
    random.seed(random_seed)
    
    # Generate random cities
    cities = {}
    for i in range(num_cities):
        cities[i] = (random.uniform(0, 100), random.uniform(0, 100))
else:
    # Manual city entry
    st.sidebar.subheader("Manual City Entry")
    st.sidebar.write("Enter city coordinates as x,y pairs (e.g., '10,20')")
    
    # Start with 3 cities by default
    if 'num_manual_cities' not in st.session_state:
        st.session_state.num_manual_cities = 3
    
    # Button to add more cities
    if st.sidebar.button("Add City"):
        st.session_state.num_manual_cities += 1
    
    # Button to remove cities
    if st.session_state.num_manual_cities > 3 and st.sidebar.button("Remove City"):
        st.session_state.num_manual_cities -= 1
    
    # Input fields for each city
    cities = {}
    for i in range(st.session_state.num_manual_cities):
        city_coord = st.sidebar.text_input(f"City {i} coordinates", value="0,0" if i == 0 else f"{random.randint(0, 100)},{random.randint(0, 100)}", key=f"city_{i}")
        try:
            x, y = map(float, city_coord.split(','))
            cities[i] = (x, y)
        except ValueError:
            st.sidebar.error(f"Invalid format for City {i}. Please use 'x,y' format.")
            cities[i] = (0, 0)

# Calculate distances between cities
def calculate_distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
               dist_matrix[i][j] = math.sqrt((cities[i][0] - cities[j][0])**2 + (cities[i][1] - cities[j][1])**2)

    return dist_matrix

distances = calculate_distance_matrix(cities)

# Animation speed control
animation_speed = st.sidebar.slider("Animation Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Function to solve TSP using dynamic programming
def solve_tsp_dynamic(distances):
    n = len(distances)
    # Maps each subset of the nodes to the cost to reach that subset, as well as what node it came from
    # to reach this subset with minimal cost
    C = {}
    
    # Initialize all subsets consisting of just one node
    for k in range(n):
        C[(1 << k, k)] = (distances[0][k], 0)
    
    solution_steps = []
    
    # Iterate subsets of increasing size and store intermediate results
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Create a bit mask representing this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            
            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    res.append((C[(prev, m)][0] + distances[m][k], m))
                C[(bits, k)] = min(res)
                
                # Store this step for visualization
                solution_steps.append({
                    'subset': list(subset),
                    'current_node': k,
                    'previous_node': C[(bits, k)][1],
                    'cost': C[(bits, k)][0]
                })
    
    # Find optimal cost from all bits to end state
    bits = (2**n - 1) - 1  # all nodes except 0
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + distances[k][0], k))
    opt, parent = min(res)
    
    # Backtrack to find full path
    path = [0]
    end = bits
    while end:
        path.append(parent)
        new_end = end & ~(1 << parent)
        parent = C[(end, parent)][1]
        end = new_end
    
    path.append(0)  # Return to start
    return opt, path, solution_steps

# Import the itertools module which we need for combinations
import itertools

# Create visualization of cities and the optimal path
def visualize_tsp_solution(cities, path, solution_steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # Plot 1: City locations and path
    ax1.set_title("TSP Solution Path")
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    
    # Plot city points
    for i, (x, y) in cities.items():
        ax1.scatter(x, y, c='blue', s=100, zorder=2)
        ax1.annotate(f"City {i}", (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Connect path
    path_x = [cities[i][0] for i in path]
    path_y = [cities[i][1] for i in path]
    ax1.plot(path_x, path_y, 'r-', zorder=1)
    
    # Draw distance graph
    G = nx.DiGraph()
    for i in cities:
        G.add_node(i, pos=cities[i])
    
    for i in range(len(path)-1):
        G.add_edge(path[i], path[i+1])
    
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue', 
            node_size=500, arrows=True, arrowsize=15, arrowstyle='-|>')
    ax2.set_title("TSP Path Graph")
    
    return fig

# Function to animate the solution process
def create_solution_animation(cities, path, solution_steps):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot city points
    for i, (x, y) in cities.items():
        ax.scatter(x, y, c='blue', s=100, zorder=2)
        ax.annotate(f"City {i}", (x, y), xytext=(5, 5), textcoords='offset points')
    
    line, = ax.plot([], [], 'r-', zorder=1, lw=2)
    title_text = ax.set_title('')
    
    # Set axis limits
    x_min, x_max = min(x for x, y in cities.values()), max(x for x, y in cities.values())
    y_min, y_max = min(y for x, y in cities.values()), max(y for x, y in cities.values())
    margin = max((x_max - x_min), (y_max - y_min)) * 0.1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # Initialize with empty path
    def init():
        line.set_data([], [])
        title_text.set_text('Initializing...')
        return line, title_text
    
    # Update function for animation
    def update(frame):
        if frame < len(solution_steps):
            step = solution_steps[frame]
            title_text.set_text(f"Processing subset {step['subset']}, Current node: {step['current_node']}, Previous: {step['previous_node']}")
            
            # Show current path consideration
            current_path = [step['previous_node'], step['current_node']]
            line.set_data([cities[i][0] for i in current_path], [cities[i][1] for i in current_path])
        else:
            # Show final path
            title_text.set_text("Final TSP Path")
            line.set_data([cities[i][0] for i in path], [cities[i][1] for i in path])

        return line, title_text
    
    # Create animation
    frames = min(len(solution_steps) + 30, len(solution_steps) * 2)  # Limit frames for performance
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=100/animation_speed)
    
    return fig, anim

# Solve button
if st.button("Solve TSP"):
    with st.spinner("Solving TSP and generating visualization..."):
        # Solve the TSP
        optimal_cost, optimal_path, solution_steps = solve_tsp_dynamic(distances)
        
        # Display city information
        st.subheader("City Information")
        
        # Create a dataframe for cities
        cities_df = pd.DataFrame.from_dict(
            {f"City {i}": [x, y] for i, (x, y) in cities.items()}, 
            orient='index', columns=['X Coordinate', 'Y Coordinate']
        )
        st.dataframe(cities_df)
        
        # Display distance matrix
        st.subheader("Distance Matrix")
        dist_df = pd.DataFrame(distances, 
                             index=[f"City {i}" for i in range(len(cities))],
                             columns=[f"City {i}" for i in range(len(cities))])
        st.dataframe(dist_df)
        
        # Display solution
        st.subheader("TSP Solution")
        st.write(f"*Optimal Path:* {' → '.join([f'City {i}' for i in optimal_path])}")
        st.write(f"*Optimal Cost:* {optimal_cost:.2f}")

        # Create and display visualization
        fig = visualize_tsp_solution(cities, optimal_path, solution_steps)
        st.pyplot(fig)
        
        # Create and display animation
        st.subheader("Solution Process Animation")
        animation_fig, animation = create_solution_animation(cities, optimal_path, solution_steps)
        st.pyplot(animation_fig)
        
        # Explain the solution process
        st.subheader("Explanation of Dynamic Programming Solution")
        st.write("""
        The Traveling Salesman Problem (TSP) is solved using dynamic programming with the following approach:
        
        1. *State Representation*: Each state is represented as (S, i) where S is a subset of cities and i is the last city visited.
        
        2. *Base Case*: For each city i, calculate the cost to go directly from the starting city (0) to city i.
        
        3. *Recursive Relation*: For each subset S of cities and each city i in S, we compute the minimum cost to visit all cities in S ending at city i.
        
        4. *Backtracking*: Once all states are computed, we backtrack to find the optimal path.
        
        The visualization shows the process of building up the optimal solution by considering different subsets of cities and finding the optimal path within each subset.
        """)