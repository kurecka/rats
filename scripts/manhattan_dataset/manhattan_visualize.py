
import folium
import networkx as nx

def plot_manhattan_map(mapfile, filename, init_state, targets):
    G = nx.MultiDiGraph(nx.read_graphml(mapfile))

    nodes_all = {}
    for node in G.nodes.data():
        name = str(node[0])
        point = [node[1]['lat'], node[1]['lon']]
        nodes_all[name] = point
    global_lat = []; global_lon = []
    for name, point in nodes_all.items():
        global_lat.append(point[0])
        global_lon.append(point[1])
    min_point = [min(global_lat), min(global_lon)]
    max_point =[max(global_lat), max(global_lon)]
    m = folium.Map(zoom_start=1, tiles='cartodbpositron')
    m.fit_bounds([min_point, max_point])

    # add initial state, reload states and target states
    folium.CircleMarker(location=[G.nodes[init_state]['lat'], G.nodes[init_state]['lon']],
                    radius= 3,
                    popup = 'initial state',
                    color='green',
                    fill_color = 'green',
                    fill_opacity=1,
                    fill=True).add_to(m)

    for node in targets:
        folium.CircleMarker(location=[G.nodes[node]['lat'], G.nodes[node]['lon']],
                    radius= 3,
                    popup = 'target state',
                    color="red",
                    fill_color = "red",
                    fill_opacity=1,
                    fill=True).add_to(m)
    m.save(filename)
