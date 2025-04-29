import dash
from dash import dcc, html, Input, Output, State, callback 
import plotly.graph_objects as go
import numpy as np
import time
import random
import networkx as nx
import pandas as pd
from dash.exceptions import PreventUpdate
import math
from itertools import combinations

# Custom color palette
COLOR_PALETTE = {
    "background": "#f8f9fa",
    "panel": "#ffffff",
    "primary": "#3366cc",
    "secondary": "#dc3545",
    "accent": "#28a745",
    "text": "#212529",
    "lightgray": "#e9ecef",
    "visited": "#17a2b8",
    "optimal": "#fd7e14"
}

class TSPSolver:
    def _init_(self):
        self.cities = []
        self.distances = None
        self.solution = []
        self.solution_cost = float('inf')
        self.animation_data = []
        self.execution_time = 0
        
    def set_cities(self, cities):
        """Set the city coordinates and calculate distances."""
        self.cities = cities
        self._calculate_distances()
        
    def _calculate_distances(self):
        """Calculate the distance matrix between all cities."""
        n = len(self.cities)
        self.distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    self.distances[i][j] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                else:
                    self.distances[i][j] = float('inf')  # Prevent self-loops
                    
    def solve_tsp_dp(self):
        """Solve the TSP using dynamic programming (Held-Karp algorithm)."""
        start_time = time.time()
        self.animation_data = []
        n = len(self.cities)
        
        # Initialize DP table
        # dp[(mask, i)] = minimum distance to visit all cities in mask and end at city i
        dp = {}
        
        # Base case: start at city 0
        for i in range(1, n):
            dp[(1 << i, i)] = self.distances[0][i]
            self.animation_data.append({
                'type': 'subproblem',
                'mask': 1 << i,
                'end': i,
                'cost': self.distances[0][i],
                'path': [0, i]
            })
        
        # Fill the dp table
        for mask in range(1, 1 << n):
            # Skip if city 0 is in the mask (except when it's the only city)
            if mask & 1 and mask != 1:
                continue
                
            for end in range(1, n):
                # Skip if the end city is not in the mask
                if not (mask & (1 << end)):
                    continue
                
                # If only one city in the mask, already handled in base case
                if mask == (1 << end):
                    continue
                
                # Try all possible previous cities
                prev_mask = mask & ~(1 << end)
                min_dist = float('inf')
                best_prev = -1
                
                for prev in range(n):
                    if prev_mask & (1 << prev):
                        current_dist = dp.get((prev_mask, prev), float('inf')) + self.distances[prev][end]
                        if current_dist < min_dist:
                            min_dist = current_dist
                            best_prev = prev
                
                dp[(mask, end)] = min_dist
                
                # Create the path for this subproblem
                if best_prev != -1:
                    # Get the path to best_prev
                    prev_path = next((item['path'] for item in self.animation_data 
                                    if item['type'] == 'subproblem' and 
                                    item['mask'] == prev_mask and 
                                    item['end'] == best_prev), [0])
                    
                    current_path = prev_path.copy()
                    current_path.append(end)
                    
                    self.animation_data.append({
                        'type': 'subproblem',
                        'mask': mask,
                        'end': end,
                        'cost': min_dist,
                        'path': current_path
                    })
        
        # Find the optimal solution
        min_cost = float('inf')
        best_last = -1
        all_cities_except_0 = (1 << n) - 2  # All cities except 0
        
        for end in range(1, n):
            cost = dp.get((all_cities_except_0, end), float('inf')) + self.distances[end][0]
            if cost < min_cost:
                min_cost = cost
                best_last = end
        
        # Reconstruct the path
        if best_last != -1:
            # Get the path to best_last
            final_path = next((item['path'] for item in self.animation_data 
                             if item['type'] == 'subproblem' and 
                             item['mask'] == all_cities_except_0 and 
                             item['end'] == best_last), [0])
            
            final_path.append(0)  # Return to the start
            self.solution = final_path
            self.solution_cost = min_cost
            
            self.animation_data.append({
                'type': 'final_solution',
                'path': final_path,
                'cost': min_cost
            })
        else:
            self.solution = []
            self.solution_cost = float('inf')
        
        self.execution_time = time.time() - start_time
        return self.solution, self.solution_cost
    
    def generate_random_cities(self, n, max_coord=100, seed=None):
        """Generate n random city coordinates within [0, max_coord]."""
        if seed is not None:
            random.seed(seed)
            
        cities = []
        for _ in range(n):
            x = random.uniform(0, max_coord)
            y = random.uniform(0, max_coord)
            cities.append((x, y))
            
        return cities
    
    def get_mst_based_lower_bound(self):
        """Calculate a lower bound using minimum spanning tree."""
        if len(self.cities) <= 1:
            return 0
            
        # Create a graph
        G = nx.Graph()
        for i in range(len(self.cities)):
            for j in range(i+1, len(self.cities)):
                G.add_edge(i, j, weight=self.distances[i][j])
                
        # Find MST
        mst = nx.minimum_spanning_tree(G)
        mst_weight = sum(mst[u][v]['weight'] for u, v in mst.edges())
        
        # Find two minimum edges for each vertex
        min_edges = []
        for v in G.nodes():
            edges = [(v, u, G[v][u]['weight']) for u in G.neighbors(v)]
            edges.sort(key=lambda x: x[2])
            if len(edges) >= 2:
                min_edges.extend(edges[:2])
            else:
                min_edges.extend(edges)
                
        min_edges.sort(key=lambda x: x[2])
        
        # Add the weight of the two cheapest edges
        mst_weight += min_edges[0][2]
        
        return mst_weight
    
    def get_subproblems_at_level(self, level):
        """Get all subproblems with a specific number of cities in the mask."""
        results = []
        for item in self.animation_data:
            if item['type'] == 'subproblem':
                cities_count = bin(item['mask']).count('1')
                if cities_count == level:
                    results.append(item)
        return results
    
    def get_animation_data(self):
        """Return the animation data for visualization."""
        return self.animation_data
        
    def get_statistics(self):
        """Return solver statistics."""
        n = len(self.cities)
        subproblems_solved = len([item for item in self.animation_data if item['type'] == 'subproblem'])
        
        total_subproblems = 0
        for i in range(1, n):
            total_subproblems += i * (2 ** (n-1))
        
        return {
            'cities': n,
            'execution_time': self.execution_time,
            'optimal_cost': self.solution_cost,
            'subproblems_solved': subproblems_solved,
            'theoretical_complexity': f"O(n²·2ⁿ) = {n*2 * 2*n}",
            'lower_bound': self.get_mst_based_lower_bound() if n > 1 else 0
        }

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define layout
app.layout = html.Div([
    html.Div([
        html.H1("Traveling Salesman Problem Solver", 
                style={'textAlign': 'center', 'color': COLOR_PALETTE['primary'], 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                html.H3("Configuration", style={'color': COLOR_PALETTE['text']}),
                html.Div([
                    html.Label("Number of Cities"),
                    dcc.Slider(
                        id='num-cities-slider',
                        min=4,
                        max=15,
                        value=6,
                        marks={i: str(i) for i in range(4, 16)},
                        step=1
                    ),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Button("Generate Random Cities", id='generate-button', 
                             style={'backgroundColor': COLOR_PALETTE['primary'], 'color': 'white', 
                                   'border': 'none', 'padding': '10px', 'borderRadius': '5px',
                                   'marginRight': '10px'}),
                    html.Button("Solve TSP", id='solve-button', 
                             style={'backgroundColor': COLOR_PALETTE['accent'], 'color': 'white', 
                                   'border': 'none', 'padding': '10px', 'borderRadius': '5px'}),
                ], style={'display': 'flex', 'marginBottom': '15px'}),
                
                html.Div([
                    html.Label("Animation Speed"),
                    dcc.Slider(
                        id='animation-speed-slider',
                        min=1,
                        max=10,
                        value=5,
                        marks={1: 'Slow', 5: 'Medium', 10: 'Fast'},
                        step=1
                    ),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Label("Animation Progress"),
                    dcc.Slider(
                        id='animation-progress-slider',
                        min=0,
                        max=100,
                        value=0,
                        step=1,
                        updatemode='drag'
                    ),
                ], style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Button("Play Animation", id='play-button', 
                             style={'backgroundColor': COLOR_PALETTE['secondary'], 'color': 'white', 
                                   'border': 'none', 'padding': '10px', 'borderRadius': '5px',
                                   'marginRight': '10px'}),
                    html.Button("Pause Animation", id='pause-button', 
                             style={'backgroundColor': COLOR_PALETTE['text'], 'color': 'white', 
                                   'border': 'none', 'padding': '10px', 'borderRadius': '5px'}),
                ], style={'display': 'flex', 'marginBottom': '15px'}),
                
                html.Div([
                    html.H4("Statistics", style={'color': COLOR_PALETTE['text']}),
                    html.Div(id='statistics-container', style={
                        'backgroundColor': COLOR_PALETTE['panel'],
                        'padding': '15px',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
                    })
                ]),
                
                # Store for cities data
                dcc.Store(id='cities-store'),
                # Store for solver data
                dcc.Store(id='solver-store'),
                # Store for animation state
                dcc.Store(id='animation-state', data={'playing': False, 'step': 0, 'total': 0}),
                # Interval component for animation
                dcc.Interval(
                    id='animation-interval',
                    interval=200,  # in milliseconds
                    n_intervals=0,
                    disabled=True
                ),
                
            ], style={'width': '30%', 'padding': '20px', 'backgroundColor': COLOR_PALETTE['lightgray'], 
                     'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='tsp-graph',
                        figure={},
                        style={'height': '500px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    dcc.Tabs([
                        dcc.Tab(label='Current Step', children=[
                            html.Div(id='current-step-info', style={
                                'padding': '15px',
                                'backgroundColor': COLOR_PALETTE['panel'],
                                'borderRadius': '5px',
                                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                                'height': '150px',
                                'overflowY': 'auto'
                            })
                        ], style={'padding': '10px'}),
                        dcc.Tab(label='State Space Tree', children=[
                            dcc.Graph(
                                id='state-space-graph',
                                figure={},
                                style={'height': '150px'}
                            )
                        ], style={'padding': '10px'}),
                        dcc.Tab(label='Progress', children=[
                            dcc.Graph(
                                id='progress-chart',
                                figure={},
                                style={'height': '150px'}
                            )
                        ], style={'padding': '10px'}),
                    ])
                ])
                
            ], style={'width': '70%', 'padding': '20px'})
            
        ], style={'display': 'flex', 'gap': '20px'})
        
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 
              'backgroundColor': COLOR_PALETTE['background'], 'fontFamily': 'Arial'})
])

# Callback to generate random cities
@callback(
    Output('cities-store', 'data'),
    Output('tsp-graph', 'figure'),
    Input('generate-button', 'n_clicks'),
    State('num-cities-slider', 'value'),
    prevent_initial_call=True
)
def generate_cities(n_clicks, num_cities):
    solver = TSPSolver()
    cities = solver.generate_random_cities(num_cities)
    
    # Create the plotly figure
    fig = go.Figure()
    
    # Add the cities as scatter points
    x_coords, y_coords = zip(*cities)
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(
            size=15,
            color=COLOR_PALETTE['primary'],
            symbol='circle',
            line=dict(width=2, color='white')
        ),
        text=list(range(len(cities))),
        textposition="top center",
        name='Cities'
    ))
    
    # Update layout
    fig.update_layout(
        title='TSP Visualization',
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        template='plotly_white',
        showlegend=False,
        hovermode='closest',
    )
    
    # Make axes equal to preserve distance perception
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1
    )
    
    return {'cities': cities}, fig

# Callback to solve TSP
@callback(
    Output('solver-store', 'data'),
    Output('animation-state', 'data'),
    Output('animation-progress-slider', 'max'),
    Output('statistics-container', 'children'),
    Input('solve-button', 'n_clicks'),
    State('cities-store', 'data'),
    prevent_initial_call=True
)
def solve_tsp(n_clicks, cities_data):
    if not cities_data or 'cities' not in cities_data:
        raise PreventUpdate
    
    solver = TSPSolver()
    solver.set_cities(cities_data['cities'])
    solution, cost = solver.solve_tsp_dp()
    animation_data = solver.get_animation_data()
    stats = solver.get_statistics()
    
    # Update animation state
    animation_state = {
        'playing': False,
        'step': 0,
        'total': len(animation_data)
    }
    
    # Format statistics for display
    stats_children = [
        html.Table([
            html.Tr([html.Td("Number of Cities:"), html.Td(f"{stats['cities']}")]),
            html.Tr([html.Td("Execution Time:"), html.Td(f"{stats['execution_time']:.4f} seconds")]),
            html.Tr([html.Td("Optimal Tour Cost:"), html.Td(f"{stats['optimal_cost']:.2f}")]),
            html.Tr([html.Td("Subproblems Solved:"), html.Td(f"{stats['subproblems_solved']}")]),
            html.Tr([html.Td("Theoretical Complexity:"), html.Td(stats['theoretical_complexity'])]),
            html.Tr([html.Td("MST Lower Bound:"), html.Td(f"{stats['lower_bound']:.2f}")])
        ], style={'width': '100%', 'borderCollapse': 'collapse'})
    ]
    
    return {'animation_data': animation_data, 'solution': solution, 'cost': cost}, animation_state, len(animation_data), stats_children

# Callback to update the animation interval
@callback(
    Output('animation-interval', 'interval'),
    Output('animation-interval', 'disabled'),
    Input('animation-speed-slider', 'value'),
    Input('play-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    State('animation-state', 'data'),
    prevent_initial_call=True
)
def update_animation_interval(speed, play_clicks, pause_clicks, animation_state):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return 1000 // speed, True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'play-button':
        return 1000 // speed, False
    elif button_id == 'pause-button':
        return 1000 // speed, True
    else:  # Speed slider changed
        return 1000 // speed, not animation_state.get('playing', False)

# Callback to update animation state when playing
@callback(
    Output('animation-state', 'data', allow_duplicate=True),
    Output('animation-progress-slider', 'value'),
    Input('animation-interval', 'n_intervals'),
    State('animation-state', 'data'),
    State('solver-store', 'data'),
    prevent_initial_call=True
)
def update_animation_step(n_intervals, animation_state, solver_data):
    if not solver_data or 'animation_data' not in solver_data:
        raise PreventUpdate
    
    total_steps = len(solver_data['animation_data'])
    
    if animation_state['step'] < total_steps - 1:
        animation_state['step'] += 1
        animation_state['playing'] = True
    else:
        animation_state['playing'] = False
    
    progress_value = (animation_state['step'] / (total_steps - 1)) * 100 if total_steps > 1 else 0
    
    return animation_state, progress_value

# Callback to update animation state when slider is moved
@callback(
    Output('animation-state', 'data', allow_duplicate=True),
    Input('animation-progress-slider', 'value'),
    State('animation-state', 'data'),
    State('solver-store', 'data'),
    prevent_initial_call=True
)
def update_animation_from_slider(progress_value, animation_state, solver_data):
    if not solver_data or 'animation_data' not in solver_data:
        raise PreventUpdate
    
    total_steps = len(solver_data['animation_data'])
    
    if total_steps <= 1:
        return animation_state
    
    step = int((progress_value / 100) * (total_steps - 1))
    animation_state['step'] = step
    
    return animation_state

# Callback to update the TSP graph based on animation state
@callback(
    Output('tsp-graph', 'figure', allow_duplicate=True),
    Output('current-step-info', 'children'),
    Output('state-space-graph', 'figure'),
    Output('progress-chart', 'figure'),
    Input('animation-state', 'data'),
    State('cities-store', 'data'),
    State('solver-store', 'data'),
    prevent_initial_call=True
)
def update_tsp_visualization(animation_state, cities_data, solver_data):
    if not cities_data or not solver_data or 'cities' not in cities_data or 'animation_data' not in solver_data:
        raise PreventUpdate
    
    cities = cities_data['cities']
    animation_data = solver_data['animation_data']
    step = animation_state['step']
    
    if step >= len(animation_data):
        step = len(animation_data) - 1
    
    current_data = animation_data[step] if animation_data else None
    
    # Create the TSP graph
    fig = go.Figure()
    
    # Add the cities as scatter points
    x_coords, y_coords = zip(*cities)
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(
            size=15,
            color=COLOR_PALETTE['primary'],
            symbol='circle',
            line=dict(width=2, color='white')
        ),
        text=list(range(len(cities))),
        textposition="top center",
        name='Cities'
    ))
    
    # Add the path if available
    if current_data and 'path' in current_data:
        path = current_data['path']
        path_x = [cities[i][0] for i in path]
        path_y = [cities[i][1] for i in path]
        
        # Different styling based on type
        line_color = COLOR_PALETTE['visited']
        if current_data['type'] == 'final_solution':
            line_color = COLOR_PALETTE['optimal']
        
        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='lines',
            line=dict(
                width=3,
                color=line_color,
                dash='solid'
            ),
            name='Current Path'
        ))
        
        # Add arrows to show direction
        for i in range(len(path) - 1):
            x1, y1 = cities[path[i]]
            x2, y2 = cities[path[i + 1]]
            
            # Calculate the position for arrow (80% of the way from start to end)
            arrow_x = x1 + 0.8 * (x2 - x1)
            arrow_y = y1 + 0.8 * (y2 - y1)
            
            # Calculate the angle of the line
            angle = math.atan2(y2 - y1, x2 - x1)
            
            # Add arrow annotation
            fig.add_annotation(
                x=arrow_x,
                y=arrow_y,
                ax=x1,
                ay=y1,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=line_color
            )
    
    # Update layout
    fig.update_layout(
        title=f'TSP Visualization - {step + 1}/{len(animation_data)}',
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        template='plotly_white',
        showlegend=False,
        hovermode='closest',
    )
    
    # Make axes equal to preserve distance perception
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1
    )
    
    # Create step info display
    step_info = []
    if current_data:
        if current_data['type'] == 'subproblem':
            step_info = [
                html.H4(f"Subproblem - Step {step + 1}"),
                html.P(f"Cities visited: {', '.join(map(str, current_data['path']))}"),
                html.P(f"Current cost: {current_data['cost']:.2f}"),
                html.P(f"Number of cities in subproblem: {bin(current_data['mask']).count('1')}"),
            ]
        elif current_data['type'] == 'final_solution':
            step_info = [
                html.H4("Final Solution"),
                html.P(f"Optimal tour: {' → '.join(map(str, current_data['path']))}"),
                html.P(f"Total cost: {current_data['cost']:.2f}"),
            ]
    
    # Create state space visualization
    state_space_fig = go.Figure()
    
    if animation_data:
        # Collect nodes and edges for state space tree
        state_levels = {}
        for i, item in enumerate(animation_data[:step+1]):
            if item['type'] == 'subproblem':
                level = bin(item['mask']).count('1')
                if level not in state_levels:
                    state_levels[level] = []
                state_levels[level].append({
                    'mask': item['mask'],
                    'end': item['end'],
                    'cost': item['cost'],
                    'index': i
                })
        
        # Create node positions
        node_positions = {}
        node_costs = {}
        node_colors = []
        node_texts = []
        node_sizes = []
        
        max_level = max(state_levels.keys()) if state_levels else 0
        x_coords = []
        y_coords = []
        
        for level in sorted(state_levels.keys()):
            nodes = state_levels[level]
            width = len(nodes)
            
            for i, node in enumerate(nodes):
                node_id = f"{node['mask']}_{node['end']}"
                x = (i + 0.5) / (width + 0.1) if width > 0 else 0.5
                y = 1 - (level / (max_level + 1))
                
                node_positions[node_id] = (x, y)
                node_costs[node_id] = node['cost']
                
                x_coords.append(x)
                y_coords.append(y)
                
                # Color the current node differently
                if node['index'] == step:
                    node_colors.append(COLOR_PALETTE['secondary'])
                    node_sizes.append(15)
                else:
                    node_colors.append(COLOR_PALETTE['primary'])
                    node_sizes.append(10)
                
                node_texts.append(f"End: {node['end']}<br>Cost: {node['cost']:.2f}")
        
        # Add nodes
        state_space_fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors
            ),
            text=node_texts,
            hoverinfo='text'
        ))
        
        # Create edges (lines) between nodes
        for i, item in enumerate(animation_data[:step+1]):
            if item['type'] == 'subproblem' and i > 0:
                # Find potential previous nodes
                current_mask = item['mask']
                current_end = item['end']
                current_id = f"{current_mask}_{current_end}"
                
                # Skip if we don't have enough info to draw an edge
                if current_id not in node_positions:
                    continue
                
                # For simplicity, connect to any valid previous node
                prev_mask = current_mask & ~(1 << current_end)
                
                for prev_end in range(len(cities)):
                    if prev_mask & (1 << prev_end):
                        prev_id = f"{prev_mask}_{prev_end}"
                        
                        if prev_id in node_positions:
                            start_x, start_y = node_positions[prev_id]
                            end_x, end_y = node_positions[current_id]
                            
                            state_space_fig.add_shape(
                                type="line",
                                x0=start_x,
                                y0=start_y,
                                x1=end_x,
                                y1=end_y,
                                line=dict(
                                    color=COLOR_PALETTE['lightgray'],
                                    width=1
                                )
                            )
                            break
    
    state_space_fig.update_layout(
        title="State Space Exploration",
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Create progress chart
    progress_fig = go.Figure()
    
    if animation_data:
        # Count subproblems by level
        subproblems_by_level = {}
        for item in animation_data[:step+1]:
            if item['type'] == 'subproblem':
                level = bin(item['mask']).count('1')
                if level not in subproblems_by_level:
                    subproblems_by_level[level] = 0
                subproblems_by_level[level] += 1
        
        levels = list(sorted(subproblems_by_level.keys()))
        counts = [subproblems_by_level[level] for level in levels]
        
        progress_fig.add_trace(go.Bar(
            x=[f"Level {level}" for level in levels],
            y=counts,
            marker_color=COLOR_PALETTE['primary']
        ))
        
        progress_fig.update_layout(
            title="Subproblems by Level",
            xaxis_title="Number of Cities in Subproblem",
            yaxis_title="Count",
            margin=dict(l=20, r=20, t=40, b=20)
        )
    
    return fig, step_info, state_space_fig, progress_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)