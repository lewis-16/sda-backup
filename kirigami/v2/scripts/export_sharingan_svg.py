import numpy as np
import phidl as ph
import phidl.geometry as pg
from phidl import Path
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

circle_port_start = {}
circle_port_end = {}
start_port_line = {}
end_port_line = {}
start_port_wave = {}
end_port_wave = {}
plot_layer = {}
net_thick_list = [10, 15, 30, 60, 120, 240]
outlier_shift_angle_list = [np.pi / 64 *1.5, np.pi / 64 *1.5, np.pi / 64 , np.pi / 128 , np.pi / 64 , np.pi / 64]

def my_arc(name, radius=10, width=0.5, theta=45, start_angle=0, angle_resolution=2.5, layer=0):
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    angle1 = (start_angle) * np.pi / 180
    angle2 = (start_angle + theta) * np.pi / 180
    t = np.linspace(angle1, angle2, int(np.ceil(abs(theta) / angle_resolution)))
    inner_points_x = (inner_radius * np.cos(t)).tolist()
    inner_points_y = (inner_radius * np.sin(t)).tolist()
    outer_points_x = (outer_radius * np.cos(t)).tolist()
    outer_points_y = (outer_radius * np.sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]

    D = ph.Device("arc")
    D.add_polygon(points=(xpts, ypts), layer=layer)
    circle_port_start[f'{name}'] = D.add_port(
        name=f'{name}_start',
        midpoint=(radius * np.cos(angle1), radius * np.sin(angle1)),
        width=width,
        orientation=start_angle - 90 + 180 * (theta < 0),
    )
    circle_port_end[f'{name}'] = D.add_port(
        name=f'{name}_end',
        midpoint=(radius * np.cos(angle2), radius * np.sin(angle2)),
        width=width,
        orientation=start_angle + theta + 90 - 180 * (theta < 0),
    )
    D.info["length"] = (abs(theta) * np.pi / 180) * radius
    return D

def plot_wave_circle(rad, thickness, n_ondulation, n_loop, amplitude):
    r = rad + amplitude * rad * np.sin(np.linspace(0, n_ondulation * 2 * np.pi, n_loop))
    t = np.linspace(0, 2 * np.pi, n_loop) + 30
    x, y = r * np.cos(t), r * np.sin(t)

    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.array([dy, -dx])
    norm = norm / np.sqrt(np.sum(norm**2, axis=0))

    x_out = x + thickness / 2 * norm[0]
    y_out = y + thickness / 2 * norm[1]
    x_in = x - thickness / 2 * norm[0]
    y_in = y - thickness / 2 * norm[1]

    position_in = np.vstack((x_in[::-1], y_in[::-1]))
    position_out = np.vstack((x_out, y_out))
    input_poly = np.hstack((position_in, position_out))
    return input_poly

def get_circle_radius(rad, thickness, n_loop):
    t = np.linspace(0, 2 * np.pi, n_loop) + 30
    x, y = rad* np.cos(t), rad* np.sin(t)

    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.array([dy, -dx])
    norm = norm / np.sqrt(np.sum(norm**2, axis=0))

    x_out = x + thickness / 2 * norm[0]
    y_out = y + thickness / 2 * norm[1]
    x_in = x - thickness / 2 * norm[0]
    y_in = y - thickness / 2 * norm[1]

    radii_in = np.sqrt(x_in**2 + y_in**2)  
    radius_in = np.mean(radii_in) 

    radii_out = np.sqrt(x_out**2 + y_out**2)  
    radius_out = np.mean(radii_out)  
    return radius_in, radius_out

def get_coords_net(rad_1, rad_2, thickness_1, thickness_2, n_loop, n_ondulation, shift = True, scale = 0.99, outlier_shift_angle = np.pi / 64 *1.5):
    if shift:
        shift_angle = np.pi / 32
    else:
        shift_angle = 0
    angles = np.pi / n_ondulation * np.arange(1, n_ondulation*2 + 1) - shift_angle
    
    rad_1 = rad_1 * scale
    x_coords = rad_1 * np.cos(angles)  
    y_coords = rad_1 * np.sin(angles)  

    points_1 = np.column_stack((x_coords, y_coords))

    angles = angles - np.pi/4  - outlier_shift_angle
    x_coords = rad_2 * np.cos(angles)  
    y_coords = rad_2 * np.sin(angles) 
    points_2 = np.column_stack((x_coords, y_coords))
    return points_1, points_2

def plot_net_circle(points_1, points_2, circle_polygon_in, circle_polygon_out, thick, net, n_ondulation):
    import itertools
    for i in range(n_ondulation):
        if i in range(0, n_ondulation, 2):
            thick_final = 2* thick
        else:
            thick_final = thick
        start_point = points_1[i, :]
        end_point = points_2[i, :]
        
        mid_point = (start_point + end_point) / 2
        vector = end_point - start_point
        vector_length = np.linalg.norm(vector)

        height = vector_length / (2 * np.tan(np.radians(30) / 2))
        perpendicular_vector = np.array([-vector[1], vector[0]]) / vector_length
        candidate_1 = mid_point + height * perpendicular_vector
        candidate_2 = mid_point - height * perpendicular_vector

        distance_1 = np.linalg.norm(candidate_1)
        distance_2 = np.linalg.norm(candidate_2)
        apex = candidate_1 if distance_1 < distance_2 else candidate_2

        radius = np.linalg.norm(apex - start_point)

        start_angle = np.arctan2(start_point[1] - apex[1], start_point[0] - apex[0])
        end_angle = np.arctan2(end_point[1] - apex[1], end_point[0] - apex[0])

        if end_angle < start_angle:
            end_angle += 2 * np.pi
        if end_angle - start_angle > np.pi:
            start_angle, end_angle = end_angle, start_angle + 2 * np.pi

        angles = np.linspace(start_angle, end_angle, 120)
        arc_points = np.column_stack((apex[0] + radius * np.cos(angles),
                                    apex[1] + radius * np.sin(angles)))

        path = Path(arc_points)
        line = path.extrude(width=thick_final)
        wave_line_poly_coords = line.get_polygons()[0]  
        wave_line_poly = Polygon(wave_line_poly_coords)  
        wave_line_poly = wave_line_poly.buffer(0)  
        if not wave_line_poly.exterior.is_ccw:
            wave_line_poly = Polygon(wave_line_poly.exterior.coords[::-1])

        wave_line_diff = wave_line_poly.difference(circle_polygon_in)

        if isinstance(wave_line_diff, Polygon):
            pass
        elif isinstance(wave_line_diff, MultiPolygon):
            wave_line_diff = max(wave_line_diff.geoms, key=lambda poly: poly.area)
        else:
            print("wave_line_diff is empty or not a valid geometry.")

        wave_line_diff = wave_line_diff.difference(circle_polygon_out)

        if isinstance(wave_line_diff, Polygon):
            plot_layer[f'net_{net + 1}'][i] = D.add_polygon(np.array(wave_line_diff.exterior.coords), layer=0)
        elif isinstance(wave_line_diff, MultiPolygon):
            largest_poly = max(wave_line_diff.geoms, key=lambda poly: poly.area)
            plot_layer[f'net_{net + 1}'][i] = D.add_polygon(np.array(largest_poly.exterior.coords), layer=0)
        else:
            print("wave_line_diff is empty or not a valid geometry.")

def plot_Au_line(points_1, points_2, thick, n_ondulation, net, radius_shift = 0):
    from phidl.geometry import circle
    import math
    from itertools import combinations
    plot_layer[f'Au_line_{net}'] = {}
    start_port_line[f'Au_line_{net}'] = {}
    end_port_line[f'Au_line_{net}'] = {}

    def cal_distance(point_1, point_2):
        return np.sqrt((point_1[0] - point_2[0]) **2 + (point_1[1] - point_2[1]) ** 2)

    def find_farthest_points(points):
        max_distance = 0
        farthest_points = None
        for p1, p2 in combinations(points, 2):
            distance = cal_distance(p1, p2)
            if distance > max_distance:
                max_distance = distance
                farthest_points = np.vstack((p1, p2))
        return farthest_points

    for i in range(n_ondulation):
        if i in range(0, n_ondulation, 2):
            thick_final = thick
            
            start_point = points_1[i, :]
            end_point = points_2[i, :]
            
            mid_point = (start_point + end_point) / 2
            vector = end_point - start_point
            vector_length = np.linalg.norm(vector)

            height = vector_length / (2 * np.tan(np.radians(30) / 2))  
            perpendicular_vector = np.array([-vector[1], vector[0]]) / vector_length  
            candidate_1 = mid_point + height * perpendicular_vector  
            candidate_2 = mid_point - height * perpendicular_vector  

            distance_1 = np.linalg.norm(candidate_1)
            distance_2 = np.linalg.norm(candidate_2)
            apex = candidate_1 if distance_1 < distance_2 else candidate_2

            radius = np.linalg.norm(apex - start_point)
            radius = radius + radius_shift
            
            start_angle = np.arctan2(start_point[1] - apex[1], start_point[0] - apex[0])
            end_angle = np.arctan2(end_point[1] - apex[1], end_point[0] - apex[0])

            if end_angle < start_angle:
                end_angle += 2 * np.pi
            if end_angle - start_angle > np.pi:
                start_angle, end_angle = end_angle, start_angle + 2 * np.pi

            angles = np.linspace(start_angle, end_angle, 120) 
            arc_points = np.column_stack((apex[0] + radius * np.cos(angles),
                                        apex[1] + radius * np.sin(angles)))

            path = Path(arc_points)
            line = path.extrude(width=thick_final) 
            wave_line_poly_coords = line.get_polygons()[0]  
            wave_line_poly = Polygon(wave_line_poly_coords)  
            wave_line_poly = wave_line_poly.buffer(0)  
            if not wave_line_poly.exterior.is_ccw:
                wave_line_poly = Polygon(wave_line_poly.exterior.coords[::-1])

            wave_line_diff = wave_line_poly.difference(circle_polygon_in)

            if isinstance(wave_line_diff, Polygon):
                pass
            elif isinstance(wave_line_diff, MultiPolygon):
                wave_line_diff = max(wave_line_diff.geoms, key=lambda poly: poly.area)
            else:
                print("wave_line_diff is empty or not a valid geometry.")

            wave_line_diff = wave_line_diff.difference(circle_polygon_out)

            if isinstance(wave_line_diff, Polygon):
                largest_poly = wave_line_diff
                plot_layer[f'Au_line_{net}'][i] = D.add_polygon(np.array(largest_poly.exterior.coords), layer=1)
            elif isinstance(wave_line_diff, MultiPolygon):
                largest_poly = max(wave_line_diff.geoms, key=lambda poly: poly.area)
                plot_layer[f'Au_line_{net}'][i] = D.add_polygon(np.array(largest_poly.exterior.coords), layer=1)
            else:
                print("wave_line_diff is empty or not a valid geometry.")

            wave_line_coord = np.array(largest_poly.exterior.coords)
            
            mask = ~np.any(np.all(wave_line_coord[:, None] == wave_line_poly_coords, axis=-1), axis=1)
            wave_line_coord = wave_line_coord[mask]
            
            short_edge_points_group1 = [0]
            short_edge_points_group2 = []
            mean_distance = []
            for id in range(1, len(wave_line_coord) - 1):
                mean_distance.append(cal_distance(wave_line_coord[0, :], wave_line_coord[id, :]))
            mean_distance = np.array(mean_distance).mean()

            for id in range(1, len(wave_line_coord) - 1):
                if cal_distance(wave_line_coord[0, :], wave_line_coord[id, :]) < mean_distance:
                    short_edge_points_group1.append(id)
                else:
                    short_edge_points_group2.append(id)

            short_edge_points_group1 = wave_line_coord[short_edge_points_group1]
            short_edge_points_group2 = wave_line_coord[short_edge_points_group2]
            start_coord = find_farthest_points(short_edge_points_group1)
            end_coord = find_farthest_points(short_edge_points_group2)

            if math.dist(np.mean(start_coord, axis=0), (0,0)) > math.dist(np.mean(end_coord, axis=0), (0,0)):
                end_coord, start_coord = start_coord, end_coord
            
            vector_start = np.diff(start_coord, axis=0)[0]
            angle_start = np.arctan2(vector_start[1], vector_start[0])
            angle_start = np.degrees(angle_start) + 90

            vector_end = np.diff(end_coord, axis=0)[0]
            angle_end = np.arctan2(vector_end[1], vector_end[0])
            angle_end = np.degrees(angle_end) + 90
            
            start_port_line[f'Au_line_{net}'][i] = D.add_port(name=f'Au_line_{net}_{i}_start', midpoint = np.mean(start_coord, axis=0), width=np.sqrt(vector_start[0] **2 + vector_start[1]**2), orientation=angle_start)
            end_port_line[f'Au_line_{net}'][i] = D.add_port(name=f'Au_line_{net}_{i}_end', midpoint = np.mean(end_coord, axis=0), width=np.sqrt(vector_end[0] **2 + vector_end[1]**2), orientation=angle_end)

def plot_Au_wave(rad, thickness, n_ondulation, n_loop, amplitude, shift_distance, divide, net, shift_angle = 0):
    from phidl.geometry import circle
    r = rad + amplitude * rad * np.sin(np.linspace(0, n_ondulation * 2 * np.pi, n_loop))
    t = np.linspace(0, 2 * np.pi, n_loop) - shift_angle

    plot_layer[f'Au_wave_{net}'] = {}
    start_port_wave[f'Au_wave_{net}'] = {}
    end_port_wave[f'Au_wave_{net}'] = {}
    t_len = len(t) // divide
    r_len = len(r) // divide
    for i in range(1, divide, int(divide/8)):
        t_seg = t[i * t_len: (i + 1) * t_len]
        x, y = r[i * r_len: (i + 1) * r_len] * np.cos(t_seg), r[i * r_len: (i + 1) * r_len] * np.sin(t_seg)

        dx = np.gradient(x)
        dy = np.gradient(y)
        norm = np.array([dy, -dx])
        norm = norm / np.sqrt(np.sum(norm**2, axis=0))

        x_out = x + thickness / 2 * norm[0]
        y_out = y + thickness / 2 * norm[1]
        x_in = x - thickness / 2 * norm[0]
        y_in = y - thickness / 2 * norm[1]

        shift_x = -shift_distance * x / np.sqrt(x**2 + y**2)
        shift_y = -shift_distance * y / np.sqrt(x**2 + y**2)

        x_in += shift_x
        y_in += shift_x
        x_out += shift_x
        y_out += shift_y

        position_in = np.vstack((x_in[::-1], y_in[::-1]))
        position_out = np.vstack((x_out, y_out))
        input_poly = np.hstack((position_in, position_out))
        
        vector_start = position_in[:, 0] - position_out[:, -1]
        angle_start = np.arctan2(vector_start[1], vector_start[0])
        angle_start = np.degrees(angle_start) + 90

        vector_end = position_in[:, -1] - position_out[:, 0]
        angle_end = np.arctan2(vector_end[1], vector_end[0])
        angle_end = np.degrees(angle_end) + 90

        plot_layer[f'Au_wave_{net}'][i] = D.add_polygon(input_poly, layer=1)
        start_port_wave[f'Au_wave_{net}'][i] = D.add_port(name=f'Au_wave_{net}_{i}_start', midpoint=(position_in[:, 0] + position_out[:, -1]) / 2, width=3, orientation=angle_start)
        end_port_wave[f'Au_wave_{net}'][i] = D.add_port(name=f'Au_wave_{net}_{i}_end', midpoint=(position_in[:, -1] + position_out[:, 0])/2, width=3, orientation=angle_end)

# Build Device D
D = ph.Device()
rad_list = [300, 700, 1000, 1500, 3000, 6000, 10000]
thickness_list = [30, 30, 50, 50, 200, 400, 1200]
ondulation = [16, 16, 16, 16, 32, 32]
amplitude_list = [0.03, 0.03, 0.03, 0.03, 0.02, 0.02]
for i in range(6):
    plot_layer[f'circle_{i + 1}'] = D.add_polygon(plot_wave_circle(rad=rad_list[i], thickness=thickness_list[i], n_loop=4096 * (i + 1) * (i + 1), n_ondulation=ondulation[i], amplitude=0.03))
    plot_layer[f'circle_{i + 1}_rotated'] = plot_layer[f'circle_{i + 1}'].rotate(30)

from phidl.geometry import circle
plot_layer['circel_7'] = D.add_polygon(pg.arc(radius=rad_list[6], width=thickness_list[6], theta=360).get_polygons()[0])
plot_layer[f'circle_7_rotated'] = plot_layer['circel_7'].rotate(30)

inner_circle_radius = 305

num_line = 16
spacing = 2 * inner_circle_radius / (num_line + 1)
wave_amplitude = 3  
wave_frequency = 15   

circle_polygon = Polygon(plot_layer['circle_1_rotated'].polygons[0])
circle_polygon = circle_polygon.buffer(0)  

plot_layer['mesh_1'] = {}
for i in range(1, num_line + 1):
    y = -inner_circle_radius + i * spacing
    if abs(y) <= inner_circle_radius: 
        x = np.linspace(-np.sqrt(inner_circle_radius**2 - y**2), 
                        np.sqrt(inner_circle_radius**2 - y**2), 
                        500)  
        wave = wave_amplitude * np.sin(wave_frequency * np.pi * (x - x[0]) / (x[-1] - x[0])) 
        points = np.column_stack((x, y + wave))  
        path = Path(points) 
        wave_line = path.extrude(width=7.5)  
        wave_line_poly_coords = wave_line.get_polygons()[0]  
        wave_line_poly = Polygon(wave_line_poly_coords)  
        wave_line_poly = wave_line_poly.buffer(0)  
        if not wave_line_poly.exterior.is_ccw:
            wave_line_poly = Polygon(wave_line_poly.exterior.coords[::-1])

        wave_line_diff = wave_line_poly.difference(circle_polygon)

        if isinstance(wave_line_diff, Polygon):
            plot_layer['mesh_1'][f'y_{i}'] = D.add_polygon(np.array(wave_line_diff.exterior.coords), layer=0)
        elif isinstance(wave_line_diff, MultiPolygon):
            largest_poly = max(wave_line_diff.geoms, key=lambda poly: poly.area)
            plot_layer['mesh_1'][f'y_{i}'] = D.add_polygon(np.array(largest_poly.exterior.coords), layer=0)
        else:
            print("wave_line_diff is empty or not a valid geometry.")

for i in range(1, num_line + 1):
    x = -inner_circle_radius + i * spacing
    if abs(x) <= inner_circle_radius:  
        y = np.linspace(-np.sqrt(inner_circle_radius**2 - x**2), 
                        np.sqrt(inner_circle_radius**2 - x**2), 
                        500)  
        wave = wave_amplitude * np.sin(wave_frequency * np.pi * (y - y[0]) / (y[-1] - y[0]))  
        points = np.column_stack((x + wave, y)) 
        path = Path(points)  
        wave_line = path.extrude(width=7.5) 

        wave_line_poly_coords = wave_line.get_polygons()[0] 
        wave_line_poly = Polygon(wave_line_poly_coords)  
        if not wave_line_poly.exterior.is_ccw:
            wave_line_poly = Polygon(wave_line_poly.exterior.coords[::-1])
        wave_line_diff = wave_line_poly.difference(circle_polygon)

        if isinstance(wave_line_diff, Polygon):
            plot_layer['mesh_1'][f'x_{i}'] = D.add_polygon(np.array(wave_line_diff.exterior.coords), layer=0)
        elif isinstance(wave_line_diff, MultiPolygon):
            largest_poly = max(wave_line_diff.geoms, key=lambda poly: poly.area)
            plot_layer['mesh_1'][f'x_{i}'] = D.add_polygon(np.array(largest_poly.exterior.coords), layer=0)
        else:
            print("wave_line_diff is empty or not a valid geometry.")

net_thick_list = [10, 15, 30, 60, 120, 240]
for i in range(6):
    if i < 4:
        n_ondulation = 8
    else:
        n_ondulation = 16

    points_1, points_2 = get_coords_net(rad_1=rad_list[i], rad_2=rad_list[i+1], thickness_1=thickness_list[i], thickness_2=thickness_list[i+1], n_loop=4096, n_ondulation=n_ondulation, shift= True, outlier_shift_angle=outlier_shift_angle_list[i])
    
    circle_polygon_in = Polygon(plot_layer[f'circle_{i + 1}_rotated'].polygons[0])
    circle_polygon_in = circle_polygon_in.buffer(0)  

    circle_polygon_out = Polygon(plot_layer[f'circle_{i + 2}_rotated'].polygons[0])
    circle_polygon_out = circle_polygon_out.buffer(0) 

    plot_layer[f'net_{i + 1}'] = {}
    plot_net_circle(points_1=points_1, points_2=points_2, circle_polygon_in=circle_polygon_in, circle_polygon_out=circle_polygon_out, thick = net_thick_list[i], net = i, n_ondulation= n_ondulation * 2)

net_rad_list = [400, 500, 600, 850, 1250, 2000, 2500, 4000, 5000, 7000, 8000, 9000]
net_thickness_list = [15, 15, 25, 25, 40, 80,  80, 100, 100, 120, 120, 120]
ondulation_list = [16, 16, 16, 16,  16,  16, 16, 32, 32, 32, 32, 32]
amplitude_list = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02]

plot_layer['arm'] = {}
plot_layer['arm_rotated'] = {}
for i in range(len(net_rad_list)):
    plot_layer['arm'][i+1] = D.add_polygon(plot_wave_circle(rad=net_rad_list[i], thickness=net_thickness_list[i], n_loop=4096 * 5 * (i + 1), n_ondulation=ondulation_list[i], amplitude=amplitude_list[i]))

def plot_probe_point(points_1, radius, n_ondulation, net, orentation = 0):
    from phidl.geometry import circle
    plot_layer[f'Au_point_{net}'] = {}
    for i in range(n_ondulation):
        if i in range(orentation, n_ondulation, 2):
            radius_final = radius
        
            start_point = points_1[i, :]

            circ = circle(radius = radius_final)
            circ.move(start_point)
            plot_layer[f'Au_point_{net}'][i] = D.add_polygon(circ.get_polygons(), layer=1)    

net_thick_list = [10, 15, 30, 60, 120, 240]
for i in range(3):
    if i == 0:
        orentation = 0
    else: 
        orentation = 1

    n_ondulation = 8

    points_1, points_2 = get_coords_net(rad_1=rad_list[i], rad_2=rad_list[i+1], thickness_1=0, thickness_2=0, n_loop=4096, n_ondulation=n_ondulation, shift=True, scale = 1.0, outlier_shift_angle=outlier_shift_angle_list[i])

    plot_probe_point(points_1=points_1, radius = 25, n_ondulation= n_ondulation * 2, orentation=orentation, net = i + 1)

net_thick_list = [3, 3, 3, 6, 9, 15]
radius_shift_list = [0, -3, -6, -12, -18, -30]
start_port_line = {}
end_port_line = {}
for i in range(6):
    n_ondulation = 8

    points_1, points_2 = get_coords_net(rad_1=rad_list[i], rad_2=rad_list[i+1], thickness_1=thickness_list[i], thickness_2=thickness_list[i+1], n_loop=4096, n_ondulation=n_ondulation, shift= True, outlier_shift_angle=outlier_shift_angle_list[i])

    circle_polygon_in = Polygon(plot_layer[f'circle_{i + 1}_rotated'].polygons[0])
    circle_polygon_in = circle_polygon_in.buffer(0)  

    circle_polygon_out = Polygon(plot_layer[f'circle_{i + 2}_rotated'].polygons[0])
    circle_polygon_out = circle_polygon_out.buffer(0) 

    plot_Au_line(points_1=points_1, points_2=points_2, thick = net_thick_list[i], n_ondulation= n_ondulation * 2, net = f'{str(i + 1)}_1', radius_shift = radius_shift_list[i])

radius_shift_list_2 = [0, 3, 0, 0, 0, 0]
n_ondulation = 8
for i in [1,2,3,4, 5]:
    points_1, points_2 = get_coords_net(rad_1=rad_list[i], rad_2=rad_list[i+1], thickness_1=thickness_list[i], thickness_2=thickness_list[i+1], n_loop=4096, n_ondulation=n_ondulation, shift= True, outlier_shift_angle=outlier_shift_angle_list[i])

    circle_polygon_in = Polygon(plot_layer[f'circle_{i + 1}_rotated'].polygons[0])
    circle_polygon_in = circle_polygon_in.buffer(0)  

    circle_polygon_out = Polygon(plot_layer[f'circle_{i + 2}_rotated'].polygons[0])
    circle_polygon_out = circle_polygon_out.buffer(0) 

    plot_Au_line(points_1=points_1, points_2=points_2, thick = net_thick_list[i], n_ondulation= n_ondulation * 2, net = f'{str(i + 1)}_2', radius_shift = radius_shift_list_2[i])

radius_shift_list_3 = [0, 3, 6, 12, 18, 30]
n_ondulation = 8
for i in [2, 3, 4, 5]:
    points_1, points_2 = get_coords_net(rad_1=rad_list[i], rad_2=rad_list[i+1], thickness_1=thickness_list[i], thickness_2=thickness_list[i+1], n_loop=4096, n_ondulation=n_ondulation, shift= True, outlier_shift_angle=outlier_shift_angle_list[i])

    circle_polygon_in = Polygon(plot_layer[f'circle_{i + 1}_rotated'].polygons[0])
    circle_polygon_in = circle_polygon_in.buffer(0)  

    circle_polygon_out = Polygon(plot_layer[f'circle_{i + 2}_rotated'].polygons[0])
    circle_polygon_out = circle_polygon_out.buffer(0) 

    plot_Au_line(points_1=points_1, points_2=points_2, thick = net_thick_list[i], n_ondulation= n_ondulation * 2, net = f'{str(i + 1)}_3', radius_shift = radius_shift_list_3[i])

start_port_wave = {}
end_port_wave = {}

for i in [1, 2]:
    plot_Au_wave(rad=rad_list[i], thickness=3, n_loop=4096 * 5 * (i + 1), n_ondulation=ondulation[i], amplitude=0.03, shift_distance=-5, divide = 16, shift_angle= np.pi / 32, net = i)

port_layer = {}
import phidl.routing as pr

port_layer['layer1'] = {}
port_layer['layer1']['Au_line_1_line_1'] = {}
for i in range(0, 16, 2):
    port_layer['layer1']['Au_line_1_line_1'][i] = ph.routing.route_quad(end_port_line['Au_line_1_1'][(i + 2) % 16], start_port_line['Au_line_2_1'][i])
    D.add_polygon(port_layer['layer1']['Au_line_1_line_1'][i].get_polygons()[0], layer=1)

port_layer['layer1']['Au_line_2_wave_1'] = {}
for i in range(0, 16, 2):
    port_layer['layer1']['Au_line_2_wave_1'][i] = ph.routing.route_quad(end_port_wave['Au_wave_1'][i + 1], start_port_line['Au_line_2_2'][i])
    D.add_polygon(port_layer['layer1']['Au_line_2_wave_1'][i].get_polygons()[0], layer=1)

port_layer['layer2'] = {}
port_layer['layer2']['Au_line_1_line_1'] = {}
port_layer['layer2']['Au_line_2_line_2'] = {}

for i in range(0, 16, 2):
    port_layer['layer2']['Au_line_1_line_1'][i] = ph.routing.route_quad(end_port_line['Au_line_2_1'][(i + 2) % 16], start_port_line['Au_line_3_1'][i])
    D.add_polygon(port_layer['layer2']['Au_line_1_line_1'][i].get_polygons()[0], layer=1)

    port_layer['layer2']['Au_line_2_line_2'][i] = ph.routing.route_quad(end_port_line['Au_line_2_2'][(i + 2) % 16], start_port_line['Au_line_3_2'][i])
    D.add_polygon(port_layer['layer2']['Au_line_2_line_2'][i].get_polygons()[0], layer=1)

port_layer['layer2']['Au_line_3_wave_2'] = {}
for i in range(0, 16, 2):
    port_layer['layer2']['Au_line_3_wave_2'][i] = ph.routing.route_quad(end_port_wave['Au_wave_2'][i + 1], start_port_line['Au_line_3_3'][i])
    D.add_polygon(port_layer['layer2']['Au_line_3_wave_2'][i].get_polygons()[0], layer=1)

for layer in [3, 4, 5]:
    port_layer[f'layer_{layer}'] = {}
    port_layer[f'layer_{layer}']['Au_line_1_line_1'] = {}
    port_layer[f'layer_{layer}']['Au_line_2_line_2'] = {}
    port_layer[f'layer_{layer}']['Au_line_3_line_3'] = {}

    for i in range(0, 16, 2):
        port_layer[f'layer_{layer}']['Au_line_1_line_1'][i] = ph.routing.route_quad(end_port_line[f'Au_line_{layer}_1'][(i + 2) % 16], start_port_line[f'Au_line_{layer + 1}_1'][i])
        D.add_polygon(port_layer[f'layer_{layer}']['Au_line_1_line_1'][i].get_polygons()[0], layer=1)

        port_layer[f'layer_{layer}']['Au_line_2_line_2'][i] = ph.routing.route_quad(end_port_line[f'Au_line_{layer}_2'][(i + 2) % 16], start_port_line[f'Au_line_{layer + 1}_2'][i])
        D.add_polygon(port_layer[f'layer_{layer}']['Au_line_2_line_2'][i].get_polygons()[0], layer=1)

        port_layer[f'layer_{layer}']['Au_line_3_line_3'][i] = ph.routing.route_quad(end_port_line[f'Au_line_{layer}_3'][(i + 2) % 16], start_port_line[f'Au_line_{layer + 1}_3'][i])
        D.add_polygon(port_layer[f'layer_{layer}']['Au_line_3_line_3'][i].get_polygons()[0], layer=1)

circle_radius = np.array(range(10000-560, 10000+ 560, 15))
angles = np.degrees(np.pi / 4 * np.arange(1, 4*2 + 1))[::-1]
circle_port_start = {}
circle_port_end = {}

count = 0
for i in range(1, len(circle_radius), 3):
    radius_circle = circle_radius[i]
    angle_shift = count // 3
    D.add_ref(my_arc(name= f'circle_{count}_',radius=radius_circle, width=15, theta=(360 - angles[angle_shift]) + 15 + 0.4 * (4- angle_shift), layer = 1))
    count += 1
    if count == 24:
        break

port_layer['layer_out'] = {}
for n in range(2, 16, 2):
    for j in [1, 2, 3]:
        id = int(3 * n /2 - 4 + j)
        port_layer['layer_out'][id] = ph.routing.route_quad(end_port_line[f'Au_line_6_{j}'][n], circle_port_end[f'circle_{id}_'])
        D.add_polygon(port_layer['layer_out'][id].get_polygons()[0], layer=1)

for id in [21, 22, 23]:
    port_layer['layer_out'][id] = ph.routing.route_quad(end_port_line[f'Au_line_6_{id-20}'][0], circle_port_end[f'circle_{id}_'])
    D.add_polygon(port_layer['layer_out'][id].get_polygons()[0], layer=1)

# Export to SVG with 20x scale
scale_factor = 20.0

fig, ax = plt.subplots()
ax.set_aspect('equal')

polys = D.get_polygons(by_spec=True)
for (layer, datatype), polygon_list in polys.items():
    color = 'blue' if layer == 0 else 'red'
    for poly in polygon_list:
        coords = np.array(poly.points) if hasattr(poly, 'points') else np.array(poly)
        if len(coords) >= 3:
            xs = [p[0] * scale_factor for p in coords]
            ys = [p[1] * scale_factor for p in coords]
            ax.fill(xs, ys, alpha=0.3, edgecolor=color, facecolor=color, linewidth=1)

ax.autoscale()
ax.invert_yaxis()
plt.savefig('Sharingan.svg', dpi=300)
plt.close()
print("SVG saved to Sharingan.svg")
