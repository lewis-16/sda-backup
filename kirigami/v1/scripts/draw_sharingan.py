import os
import warnings
import numpy as np
import phidl as ph
from phidl import Device, Path
import phidl.geometry as pg
import ezdxf
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

N_LOOP_BASE = 64
N_ARC_PTS = 48

plot_layer = {}
D = None


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


def get_coords_net(rad_1, rad_2, thickness_1, thickness_2, n_loop, n_ondulation, shift=True, scale=0.99, outlier_shift_angle=np.pi / 64 * 1.5):
    if shift:
        shift_angle = np.pi / 32
    else:
        shift_angle = 0
    angles = np.pi / n_ondulation * np.arange(1, n_ondulation * 2 + 1) - shift_angle

    rad_1 = rad_1 * scale
    x_coords = rad_1 * np.cos(angles)
    y_coords = rad_1 * np.sin(angles)
    points_1 = np.column_stack((x_coords, y_coords))

    angles = angles - np.pi / 4 - outlier_shift_angle
    x_coords = rad_2 * np.cos(angles)
    y_coords = rad_2 * np.sin(angles)
    points_2 = np.column_stack((x_coords, y_coords))
    return points_1, points_2


def plot_net_circle(points_1, points_2, circle_polygon_in, circle_polygon_out, thick, net, n_ondulation):
    for i in range(n_ondulation):
        if i in range(0, n_ondulation, 2):
            thick_final = 2 * thick
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

        angles = np.linspace(start_angle, end_angle, N_ARC_PTS)
        arc_points = np.column_stack((apex[0] + radius * np.cos(angles), apex[1] + radius * np.sin(angles)))

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
            plot_layer[f"net_{net + 1}"][i] = D.add_polygon(np.array(wave_line_diff.exterior.coords), layer=0)
        elif isinstance(wave_line_diff, MultiPolygon):
            largest_poly = max(wave_line_diff.geoms, key=lambda poly: poly.area)
            plot_layer[f"net_{net + 1}"][i] = D.add_polygon(np.array(largest_poly.exterior.coords), layer=0)
        else:
            print("wave_line_diff is empty or not a valid geometry.")


def _poly_to_shapely(poly, buffer_zero=False):
    pts = np.array(poly) if not hasattr(poly, "points") else np.array(poly.points)
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    if buffer_zero:
        p = p.buffer(0)
    if p.is_empty:
        return None
    if hasattr(p, "make_valid"):
        p = p.make_valid()
    return p


def _geom_to_polygon_coords(geom):
    if geom.is_empty:
        return []
    out = []
    if geom.geom_type == "Polygon":
        if not geom.exterior.is_empty:
            out.append(np.array(geom.exterior.coords[:-1]))
        for interior in geom.interiors:
            if not interior.is_empty:
                out.append(np.array(interior.coords[:-1]))
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            if not g.exterior.is_empty:
                out.append(np.array(g.exterior.coords[:-1]))
            for interior in g.interiors:
                if not interior.is_empty:
                    out.append(np.array(interior.coords[:-1]))
    return out


def _simplify_polygon_coords(coords, tolerance=0.5):
    if len(coords) < 4:
        return coords
    p = Polygon(coords)
    if p.is_empty:
        return coords
    simplified = p.simplify(tolerance, preserve_topology=True)
    if simplified.is_empty:
        return coords
    out = np.array(simplified.exterior.coords[:-1])
    return out if len(out) >= 3 else coords


def sharingan_device_to_dxf(D, output_path, simplify_tolerance=0.5):
    polys = D.get_polygons(by_spec=True)
    substrate_polys = polys.get((0, 0), [])
    geoms = []
    for poly in substrate_polys:
        g = _poly_to_shapely(poly, buffer_zero=True)
        if g is not None and not g.is_empty:
            geoms.append(g)
    if not geoms:
        doc = ezdxf.new("R2010")
        doc.modelspace()
        doc.layers.add("substrate", color=1)
        doc.saveas(output_path)
        return
    material = unary_union(geoms)
    if hasattr(material, "make_valid"):
        material = material.make_valid()
    if material.is_empty:
        doc = ezdxf.new("R2010")
        doc.modelspace()
        doc.layers.add("substrate", color=1)
        doc.saveas(output_path)
        return
    coords_list = _geom_to_polygon_coords(material)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("substrate", color=1)
    for coords in coords_list:
        if len(coords) < 3:
            continue
        if simplify_tolerance > 0:
            coords = _simplify_polygon_coords(coords, simplify_tolerance)
        points = [(float(p[0]), float(p[1])) for p in coords]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "substrate"})
    doc.saveas(output_path)


def main():
    global D, plot_layer

    rad_list = [300, 700, 1000, 1500, 3000, 6000, 10000]
    thickness_list = [30, 30, 50, 50, 200, 400, 1200]
    ondulation = [16, 16, 16, 16, 32, 32]
    amplitude_list = [0.03, 0.03, 0.03, 0.03, 0.02, 0.02]

    D = ph.Device("sharingan")
    plot_layer = {}

    for i in range(6):
        n_loop = max(N_LOOP_BASE * ondulation[i], 512)
        plot_layer[f"circle_{i + 1}"] = D.add_polygon(
            plot_wave_circle(
                rad=rad_list[i],
                thickness=thickness_list[i],
                n_loop=int(n_loop),
                n_ondulation=ondulation[i],
                amplitude=0.03,
            )
        )
        plot_layer[f"circle_{i + 1}_rotated"] = plot_layer[f"circle_{i + 1}"].rotate(30)

    plot_layer["circel_7"] = D.add_polygon(
        pg.arc(radius=rad_list[6], width=thickness_list[6], theta=360).get_polygons()[0]
    )
    plot_layer["circle_7_rotated"] = plot_layer["circel_7"].rotate(30)

    net_thick_list = [10, 15, 30, 60, 120, 240]
    outlier_shift_angle_list = [np.pi / 64 * 1.5, np.pi / 64 * 1.5, np.pi / 64, np.pi / 128, np.pi / 64, np.pi / 64]
    for i in range(6):
        if i < 4:
            n_ond = 8
        else:
            n_ond = 16
        points_1, points_2 = get_coords_net(
            rad_1=rad_list[i],
            rad_2=rad_list[i + 1],
            thickness_1=thickness_list[i],
            thickness_2=thickness_list[i + 1],
            n_loop=4096,
            n_ondulation=n_ond,
            shift=True,
            outlier_shift_angle=outlier_shift_angle_list[i],
        )
        circle_polygon_in = Polygon(plot_layer[f"circle_{i + 1}_rotated"].polygons[0])
        circle_polygon_in = circle_polygon_in.buffer(0)
        circle_polygon_out = Polygon(plot_layer[f"circle_{i + 2}_rotated"].polygons[0])
        circle_polygon_out = circle_polygon_out.buffer(0)
        plot_layer[f"net_{i + 1}"] = {}
        plot_net_circle(
            points_1=points_1,
            points_2=points_2,
            circle_polygon_in=circle_polygon_in,
            circle_polygon_out=circle_polygon_out,
            thick=net_thick_list[i],
            net=i,
            n_ondulation=n_ond * 2,
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dxf_path = os.path.join(RESULTS_DIR, "sharingan.dxf")
    sharingan_device_to_dxf(D, dxf_path)
    print(f"已保存至 {dxf_path}")


if __name__ == "__main__":
    main()
