import os
import multiprocessing
import numpy as np
import phidl as ph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.collections import PatchCollection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
import phidl.geometry as pg
from phidl import Path
from concurrent.futures import ProcessPoolExecutor
import ezdxf
from shapely.geometry import Point, Polygon, LineString, box
from shapely.affinity import rotate, translate, scale
from shapely.ops import unary_union

GRID_CELL_SIZE_FINE = 15.0
GRID_CELL_SIZE_COARSE = 500.0
REFINE_BUFFER = 5.0


def generate_arc_connectors(
    layer_radii,
    radius_offset,
    arc_cols,
    arc_cell_width,
    arc_row_arc_length,
    from_layer_idx,
    to_layer_idx,
    col_min,
    col_max,
    include_col_max,
):
    connectors = []
    for layer_idx in [from_layer_idx, to_layer_idx]:
        r_layer_center = layer_radii[layer_idx - 1] + radius_offset
        theta_start = -np.pi / 2 - 2 * np.pi * layer_idx / 16 + np.pi - 1 * np.pi / 96
        theta_end = -np.pi / 2 - 2 * np.pi * (layer_idx + 1) / 16 + np.pi + 1 * np.pi / 96
        arc_total_angle = abs(theta_end - theta_start)
        r_arc = r_layer_center
        arc_rows = int(arc_total_angle * r_arc / arc_row_arc_length)
        if layer_idx == from_layer_idx:
            theta_from = theta_end
        else:
            theta_to = theta_start
    for col in range(arc_cols):
        if col < col_min:
            continue
        if include_col_max:
            if col > col_max:
                continue
        else:
            if col >= col_max:
                continue
        if col % 2 != 0:
            continue
        r_from_center = layer_radii[from_layer_idx - 1] + radius_offset
        r_to_center = layer_radii[to_layer_idx - 1] + radius_offset
        r_from = r_from_center - arc_cell_width * arc_cols / 2 + (col + 0.5) * arc_cell_width
        r_to = r_to_center - arc_cell_width * arc_cols / 2 + (col + 0.5) * arc_cell_width
        p1 = (r_from * np.cos(theta_from), r_from * np.sin(theta_from))
        p2 = (r_to * np.cos(theta_to), r_to * np.sin(theta_to))
        line = LineString([p1, p2])
        conn = line.buffer(arc_cell_width / 2.0, cap_style=2)
        if not conn.is_empty:
            connectors.append(conn)
    return connectors


def generate_arc_connectors_zigzag(
    layer_radii,
    radius_offset,
    arc_cols,
    arc_cell_width,
    arc_row_arc_length,
    from_layer_idx,
    to_layer_idx,
    col_min,
    col_max,
    include_col_max,
    line_width=2.0,
    delta_a=68.0,
):
    theta_from = None
    theta_to = None
    for layer_idx in [from_layer_idx, to_layer_idx]:
        theta_start = -np.pi / 2 - 2 * np.pi * layer_idx / 16 + np.pi - 1 * np.pi / 96
        theta_end = -np.pi / 2 - 2 * np.pi * (layer_idx + 1) / 16 + np.pi + 1 * np.pi / 96
        if layer_idx == from_layer_idx:
            theta_from = theta_end
        else:
            theta_to = theta_start
    r_from_center = layer_radii[from_layer_idx - 1] + radius_offset
    r_to_center = layer_radii[to_layer_idx - 1] + radius_offset
    theta_boundary = (theta_from + theta_to) / 2.0
    u_axis = np.array([np.cos(theta_boundary), np.sin(theta_boundary)])
    v_axis = np.array([-np.sin(theta_boundary), np.cos(theta_boundary)])
    half_lw = line_width / 2.0
    connectors = []
    x = 0
    for col in range(arc_cols):
        if col < col_min:
            continue
        if include_col_max:
            if col > col_max:
                continue
        else:
            if col >= col_max:
                continue
        if col % 2 != 0:
            continue
        x += 1
        r_from = r_from_center - arc_cell_width * arc_cols / 2 + (col + 0.5) * arc_cell_width
        r_to = r_to_center - arc_cell_width * arc_cols / 2 + (col + 0.5) * arc_cell_width
        d_b = (r_from + r_to) / 2.0 * (theta_to - theta_from)
        p1_global = np.array([r_from * np.cos(theta_from), r_from * np.sin(theta_from)])
        p2_global = p1_global + (delta_a - 4 * x) * u_axis
        p3_global = p2_global + d_b * v_axis
        p4_global = np.array([r_to * np.cos(theta_to), r_to * np.sin(theta_to)])
        line = LineString([
            (float(p1_global[0]), float(p1_global[1])),
            (float(p2_global[0]), float(p2_global[1])),
            (float(p3_global[0]), float(p3_global[1])),
            (float(p4_global[0]), float(p4_global[1])),
        ])
        conn = line.buffer(half_lw, cap_style=2, join_style=2)
        if not conn.is_empty:
            connectors.append(conn)
    return connectors


def build_adaptive_cells(shape, x0, y0, x1, y1, size, kong_buffer):
    cells = []
    cell = box(x0, y0, x1, y1)
    if not shape.intersects(cell):
        return cells
    if size <= GRID_CELL_SIZE_FINE:
        return [cell]
    near = kong_buffer.intersects(cell)
    if near:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        half = size / 2
        for (a0, b0, a1, b1) in [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]:
            cells.extend(build_adaptive_cells(shape, a0, b0, a1, b1, half, kong_buffer))
    else:
        cells.append(cell)
    return cells


def _build_cells_for_block(args):
    shape, kong_buffer, x0, y0, x1, y1 = args
    return build_adaptive_cells(shape, x0, y0, x1, y1, GRID_CELL_SIZE_COARSE, kong_buffer)


def _intersect_batch(args):
    cells_batch, merged = args
    out = []
    for c in cells_batch:
        p = c.intersection(merged)
        if not p.is_empty:
            out.append(p)
    return out


def fillet_polygon(points, radius, quad_segs=16):
    if len(points) < 3 or radius <= 0:
        return np.array(points)
    poly = Polygon(points).buffer(0)
    if poly.is_empty:
        return np.array(points)
    filleted = poly.buffer(-radius, join_style=1, quad_segs=quad_segs).buffer(
        radius, join_style=1, quad_segs=quad_segs
    )
    if filleted.is_empty:
        return np.array(points)
    if hasattr(filleted, 'geoms') and len(filleted.geoms) > 0:
        geom = max(filleted.geoms, key=lambda g: g.area)
    else:
        geom = filleted
    coords = np.array(geom.exterior.coords[:-1])
    return coords


def generate_radial_positions(r_i, r_o, dr1, dr2, n, dr_min, slit_width):
    positions = []
    half_sw = slit_width / 2
    r = r_i + dr1 + half_sw
    if r >= r_o:
        return positions
    positions.append(r)

    r_next = r + dr2 + slit_width
    if r_next >= r_o:
        return positions
    positions.append(r_next)

    dr_prev = dr2
    r_curr = r_next
    while True:
        dr_next = r_o * ((1 + dr_prev / r_o) ** n - 1)
        if dr_next < dr_min:
            dr_next = dr_min
        r_next = r_curr + dr_next + slit_width
        if r_next + half_sw >= r_o - dr_min:
            break
        positions.append(r_next)
        dr_prev = dr_next
        r_curr = r_next
    return positions


def add_cross_bridge(D, r_inner, width_outer, center_hole_diameter, bridge_length=100.0, inner_length=200.0, width_inner=80.0, layer=(2, 0)):
    center_circle_radius = center_hole_diameter / 2
    side = 80.0
    r_start = side / 2
    outer_length = bridge_length - inner_length
    half_w_inner = width_inner / 2
    half_w_outer = width_outer / 2

    arm_0 = ph.Device()
    arm_0.add_ref(pg.rectangle(size=(inner_length, width_inner), layer=layer)).move((r_start, -half_w_inner))
    arm_0.add_ref(pg.rectangle(size=(outer_length, width_outer), layer=layer)).move((r_start + inner_length, -half_w_outer))
    arm_90 = ph.Device()
    arm_90.add_ref(pg.rectangle(size=(width_inner, inner_length), layer=layer)).move((-half_w_inner, r_start))
    arm_90.add_ref(pg.rectangle(size=(width_outer, outer_length), layer=layer)).move((-half_w_outer, r_start + inner_length))
    arm_180 = ph.Device()
    arm_180.add_ref(pg.rectangle(size=(inner_length, width_inner), layer=layer)).move((-r_start - inner_length, -half_w_inner))
    arm_180.add_ref(pg.rectangle(size=(outer_length, width_outer), layer=layer)).move((-r_start - bridge_length, -half_w_outer))
    arm_270 = ph.Device()
    arm_270.add_ref(pg.rectangle(size=(width_inner, inner_length), layer=layer)).move((-half_w_inner, -r_start - inner_length))
    arm_270.add_ref(pg.rectangle(size=(width_outer, outer_length), layer=layer)).move((-half_w_outer, -r_start - bridge_length))
    cross_01 = pg.boolean(arm_0, arm_90, operation="A+B", layer=layer)
    cross_012 = pg.boolean(cross_01, arm_180, operation="A+B", layer=layer)
    cross = pg.boolean(cross_012, arm_270, operation="A+B", layer=layer)
    center_hole = pg.circle(radius=center_circle_radius, layer=(0, 0))
    bridge = pg.boolean(cross, center_hole, operation="A-B", layer=layer)
    D.add_ref(bridge)


def _build_spiral_grid_chunk(args):
    theta_start, theta_end, r_start, r_end, grid_rows, grid_cols, grid_cell_width, row_start, row_end = args
    n_fine = 4
    row_indices = np.concatenate([
        np.linspace(0, n_fine, n_fine * n_fine + 1)[:-1],
        np.arange(n_fine, grid_rows + 1),
    ])
    t = row_indices / grid_rows
    theta = theta_start + (theta_end - theta_start) * t
    if np.abs(theta_end - theta_start) < 1e-10:
        r = r_start + (r_end - r_start) * np.linspace(0, 1, len(t))
    else:
        r = r_start + (r_end - r_start) * t
    center_pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    n_pts = len(center_pts)
    perps = []
    for i in range(n_pts):
        if i == 0:
            dx = center_pts[1][0] - center_pts[0][0]
            dy = center_pts[1][1] - center_pts[0][1]
        elif i == n_pts - 1:
            dx = center_pts[i][0] - center_pts[i - 1][0]
            dy = center_pts[i][1] - center_pts[i - 1][1]
        else:
            dx = center_pts[i + 1][0] - center_pts[i - 1][0]
            dy = center_pts[i + 1][1] - center_pts[i - 1][1]
        seg_len = np.hypot(dx, dy)
        if seg_len < 1e-8:
            perps.append((0.0, 0.0))
        else:
            perps.append((-dy / seg_len, dx / seg_len))
    seg_start = row_start * n_fine if row_start < n_fine else n_fine * n_fine + (row_start - n_fine)
    seg_end = row_end * n_fine if row_end <= n_fine else n_fine * n_fine + (row_end - n_fine)
    out = []
    for seg in range(seg_start, seg_end):
        if seg + 1 >= n_pts:
            break
        p0 = center_pts[seg]
        p1 = center_pts[seg + 1]
        perp0 = perps[seg]
        perp1 = perps[seg + 1]
        for col in range(grid_cols):
            off0 = (col - (grid_cols - 1) / 2.0) * grid_cell_width
            off1 = off0 + grid_cell_width
            c00 = (p0[0] + off0 * perp0[0], p0[1] + off0 * perp0[1])
            c01 = (p0[0] + off1 * perp0[0], p0[1] + off1 * perp0[1])
            c10 = (p1[0] + off0 * perp1[0], p1[1] + off0 * perp1[1])
            c11 = (p1[0] + off1 * perp1[0], p1[1] + off1 * perp1[1])
            rect_pts = [c00, c01, c11, c10, c00]
            rect = Polygon(rect_pts).buffer(0)
            if not rect.is_empty:
                out.append((col, seg, rect))
    return out


def archimedean_spiral_points(theta_start, theta_end, r_start, r_end, n_pts=128):
    theta = np.linspace(theta_start, theta_end, n_pts)
    if np.abs(theta_end - theta_start) < 1e-10:
        r = np.linspace(r_start, r_end, n_pts)
    else:
        r = r_start + (r_end - r_start) * (theta - theta_start) / (theta_end - theta_start)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def get_one_arm_device(arm_index, r_hole, bridge_length, inner_length, width_inner, width_outer, layer=(2, 0)):
    outer_length = bridge_length - inner_length
    half_w_inner = width_inner / 2
    half_w_outer = width_outer / 2
    if arm_index == 0:
        arm = ph.Device()
        arm.add_ref(pg.rectangle(size=(inner_length, width_inner), layer=layer)).move((r_hole, -half_w_inner))
        arm.add_ref(pg.rectangle(size=(outer_length, width_outer), layer=layer)).move((r_hole + inner_length, -half_w_outer))
    elif arm_index == 1:
        arm = ph.Device()
        arm.add_ref(pg.rectangle(size=(width_inner, inner_length), layer=layer)).move((-half_w_inner, r_hole))
        arm.add_ref(pg.rectangle(size=(width_outer, outer_length), layer=layer)).move((-half_w_outer, r_hole + inner_length))
    elif arm_index == 2:
        arm = ph.Device()
        arm.add_ref(pg.rectangle(size=(inner_length, width_inner), layer=layer)).move((-r_hole - inner_length, -half_w_inner))
        arm.add_ref(pg.rectangle(size=(outer_length, width_outer), layer=layer)).move((-r_hole - bridge_length, -half_w_outer))
    else:
        arm = ph.Device()
        arm.add_ref(pg.rectangle(size=(width_inner, inner_length), layer=layer)).move((-half_w_inner, -r_hole - inner_length))
        arm.add_ref(pg.rectangle(size=(width_outer, outer_length), layer=layer)).move((-half_w_outer, -r_hole - bridge_length))
    return arm


def get_one_spiral_polygons(arm_index, r_i, r_hole, bridge_length, spiral_width=80.0):
    r_start = r_hole + bridge_length
    r_end = r_i + spiral_width / 2.0
    span = 2 * np.pi * (1.0 + 1.0 / 16.0)
    arms_params = [
        (0, -span),
        (np.pi / 2, np.pi / 2 - span),
        (np.pi, np.pi - span),
        (3 * np.pi / 2, 3 * np.pi / 2 - span),
    ]
    theta_start, theta_end = arms_params[arm_index]
    pts = archimedean_spiral_points(theta_start, theta_end, r_start, r_end, n_pts=128)
    path = Path(pts)
    spiral_dev = path.extrude(width=spiral_width)
    polys = spiral_dev.get_polygons()
    out = []
    for poly in (polys or []):
        coords = np.array(poly.points) if hasattr(poly, 'points') else np.array(poly)
        if len(coords) >= 3:
            p = Polygon(coords).buffer(0)
            if p and not p.is_empty:
                out.append(p)
    return out


def add_spirals_to_180(D, r_i, r_hole, bridge_length, spiral_width=80.0, layer=(2, 0)):
    r_start = r_hole + bridge_length
    r_end = r_i + spiral_width / 2.0

    span = 2 * np.pi * (1.0 + 1.0 / 16.0)
    arms = [
        (0, -span),
        (np.pi / 2, np.pi / 2 - span),
        (np.pi, np.pi - span),
        (3 * np.pi / 2, 3 * np.pi / 2 - span),
    ]
    for theta_start, theta_end in arms:
        pts = archimedean_spiral_points(theta_start, theta_end, r_start, r_end, n_pts=128)
        path = Path(pts)
        spiral_dev = path.extrude(width=spiral_width)
        polys = spiral_dev.get_polygons()
        if polys:
            for poly in polys:
                coords = np.array(poly.points) if hasattr(poly, 'points') else np.array(poly)
                if len(coords) >= 3:
                    D.add_polygon(coords, layer=layer)


def add_kirigami_to_device(D, r_i, r_o, positions, N_theta, theta, offset=0, slit_width=0.05, fillet_radius=0, slit_base_angle=0, layer=1):
    theta_i = 2 * np.pi / (N_theta * (1 + theta))
    theta_a = theta * theta_i
    theta_i_deg = np.degrees(theta_i)
    base_rad = np.radians(slit_base_angle)

    for j, r in enumerate(positions):
        phase = offset * j
        for k in range(N_theta):
            slit_center = k * (theta_i + theta_a) + phase + base_rad
            start_angle = slit_center - theta_i / 2
            start_deg = np.degrees(start_angle)
            arc_dev = pg.arc(radius=r, width=slit_width, theta=theta_i_deg, start_angle=start_deg, layer=layer)
            raw = arc_dev.get_polygons(by_spec=False)
            p0 = raw[0]
            poly = np.array(p0.points) if hasattr(p0, 'points') else np.array(p0)
            if fillet_radius > 0:
                poly = fillet_polygon(poly, fillet_radius)
            D.add_polygon(poly, layer=layer)


def _poly_to_shapely(poly, buffer_zero=False):
    pts = np.array(poly) if not hasattr(poly, "points") else np.array(poly.points)
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    if buffer_zero:
        p = p.buffer(0)
    if p.is_empty:
        return None
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


def phidl_device_to_dxf(D, output_path, include_io_pad=True):
    """导出 DXF，包含 Kirigami_platte 图层，可选包含 io_pad"""
    polys = D.get_polygons(by_spec=True)
    outline_geoms = [_poly_to_shapely(p, buffer_zero=True) for p in polys.get((0, 0), [])]
    bridge_geoms = [_poly_to_shapely(p, buffer_zero=True) for p in polys.get((2, 0), [])]
    kirigami_geoms = [_poly_to_shapely(p, buffer_zero=False) for p in polys.get((1, 0), [])]
    outline_geoms = [g for g in outline_geoms if g is not None]
    bridge_geoms = [g for g in bridge_geoms if g is not None]
    kirigami_geoms = [g for g in kirigami_geoms if g is not None]
    outline_plus_bridge = unary_union(outline_geoms + bridge_geoms) if (outline_geoms or bridge_geoms) else Polygon()
    if hasattr(outline_plus_bridge, "make_valid"):
        outline_plus_bridge = outline_plus_bridge.make_valid()
    tol = 0.1
    kirigami_buffed = [g.buffer(tol) for g in kirigami_geoms if not g.is_empty]
    kirigami_union = unary_union(kirigami_buffed) if kirigami_buffed else Polygon()
    if hasattr(kirigami_union, "make_valid"):
        kirigami_union = kirigami_union.make_valid()
    material = outline_plus_bridge.difference(kirigami_union)
    if hasattr(material, "make_valid"):
        material = material.make_valid()
    coords_list = _geom_to_polygon_coords(material)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("Kirigami_platte", color=1)
    for coords in coords_list:
        if len(coords) < 3:
            continue
        points = [(float(p[0]), float(p[1])) for p in coords]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "Kirigami_platte"})
    if include_io_pad:
        doc.layers.add("io_pad", color=2)
        for poly in polys.get((3, 0), []):
            pts = np.array(poly) if not hasattr(poly, "points") else np.array(poly.points)
            if len(pts) < 3:
                continue
            points = [(float(p[0]), float(p[1])) for p in pts]
            msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "io_pad"})
    doc.saveas(output_path)


def write_woio_dxf_5layers(
    D,
    output_path,
    r_i,
    r_o,
    center_hole_diameter,
    bridge_length,
    inner_length,
    width_inner,
    width_outer,
    spiral_width,
):
    polys = D.get_polygons(by_spec=True)
    outline_geoms = [_poly_to_shapely(p, buffer_zero=True) for p in polys.get((0, 0), [])]
    kirigami_geoms = [_poly_to_shapely(p, buffer_zero=False) for p in polys.get((1, 0), [])]
    outline_geoms = [g for g in outline_geoms if g is not None]
    kirigami_geoms = [g for g in kirigami_geoms if g is not None]
    outline_union = unary_union(outline_geoms) if outline_geoms else Polygon()
    if hasattr(outline_union, "make_valid"):
        outline_union = outline_union.make_valid()
    tol = 0.1
    kirigami_buffed = [g.buffer(tol) for g in kirigami_geoms if not g.is_empty]
    kirigami_union = unary_union(kirigami_buffed) if kirigami_buffed else Polygon()
    if hasattr(kirigami_union, "make_valid"):
        kirigami_union = kirigami_union.make_valid()
    ring = outline_union.difference(kirigami_union)
    if hasattr(ring, "make_valid"):
        ring = ring.make_valid()

    r_hole = center_hole_diameter / 2
    center_hole = Point(0, 0).buffer(r_hole)

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.layers.add("Ring", color=1)
    for coords in _geom_to_polygon_coords(ring):
        if len(coords) < 3:
            continue
        points = [(float(p[0]), float(p[1])) for p in coords]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "Ring"})

    side = 80.0
    half_side = side / 2.0
    square_coords = [
        (-half_side, -half_side),
        (half_side, -half_side),
        (half_side, half_side),
        (-half_side, half_side),
        (-half_side, -half_side),
    ]
    square_poly = Polygon(square_coords).buffer(0)
    inner_circle = Point(0, 0).buffer(30.0)
    center_shape = square_poly.difference(inner_circle)
    if hasattr(center_shape, "make_valid"):
        center_shape = center_shape.make_valid()
    doc.layers.add("Center_square_circle", color=6)
    for coords in _geom_to_polygon_coords(center_shape):
        if len(coords) < 3:
            continue
        points = [(float(p[0]), float(p[1])) for p in coords]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "Center_square_circle"})

    trapezoid_base = Polygon([(940.0, 40.0), (1030.0, 40.0), (1030.0, 120.0), (1000.0, 120.0)]).buffer(0)
    trapezoid_inner_base = Polygon([
        (960.0, 50.0),
        (1020.0, 50.0),
        (1020.0, 110.0),
        (1008.0, 110.0),
    ]).buffer(0)
    trapezoid_base = trapezoid_base.difference(trapezoid_inner_base)
    trapezoids_980_40 = [rotate(trapezoid_base, k * 90.0, origin=(0.0, 0.0)) for k in range(4)]
    doc.layers.add("Square_980_40", color=7)
    for t in trapezoids_980_40:
        for coords in _geom_to_polygon_coords(t):
            if len(coords) < 3:
                continue
            points = [(float(p[0]), float(p[1])) for p in coords]
            msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "Square_980_40"})

    base_x = 450.0
    step = 50.0
    n_circles = 8
    group1 = []
    holes_g1 = []
    for base_y in (60.0, -60.0):
        for i in range(n_circles):
            cx = base_x - i * step
            cy = base_y
            c = Point(cx, cy).buffer(20.0).difference(Point(cx, cy).buffer(10.0))
            if hasattr(c, "make_valid"):
                c = c.make_valid()
            group1.append(c)
            holes_g1.append(Point(cx, cy).buffer(10.0))
            dx_outer = 25.0
            dy_outer = 25.0 * np.sqrt(3.0) if base_y > 0 else -25.0 * np.sqrt(3.0)
            ox = cx + dx_outer
            oy = cy + dy_outer
            c2 = Point(ox, oy).buffer(20.0).difference(Point(ox, oy).buffer(10.0))
            if hasattr(c2, "make_valid"):
                c2 = c2.make_valid()
            group1.append(c2)
            holes_g1.append(Point(ox, oy).buffer(10.0))
    group2 = [scale(g, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for g in group1]
    holes_g2 = [scale(h, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for h in holes_g1]

    line_width_vert = 40.0
    half_lw_vert = line_width_vert / 2.0
    # 垂直向中心的线（与 bridge 相连）
    vert_lines_g1 = []
    vert_lines_g2 = []
    for sign_x in (1.0, -1.0):
        for base_y in (60.0, -60.0):
            for i in range(n_circles):
                cx = sign_x * (base_x - i * step)
                if base_y > 0:
                    y1 = base_y + 2
                    y2 = 40.0
                else:
                    y1 = base_y - 2
                    y2 = -40.0
                x0 = cx
                rect_coords = [
                    (x0 - half_lw_vert, y1),
                    (x0 + half_lw_vert, y1),
                    (x0 + half_lw_vert, y2),
                    (x0 - half_lw_vert, y2),
                    (x0 - half_lw_vert, y1),
                ]
                lp = Polygon(rect_coords).buffer(0)
                if lp.is_empty:
                    continue
                if cx > 0:
                    vert_lines_g1.append(lp)
                else:
                    vert_lines_g2.append(lp)

    # 圆心连线（两 circle 相连）
    line_width_conn = 40.0
    half_lw_conn = line_width_conn / 2.0
    conn_lines_g1 = []
    for base_y in (60.0, -60.0):
        for i in range(n_circles):
            cx = base_x - i * step
            cy = base_y
            ox = cx + 25.0
            oy = cy + (25.0 * np.sqrt(3.0) if base_y > 0 else -25.0 * np.sqrt(3.0))
            mx = 0.5 * (cx + ox)
            my = 0.5 * (cy + oy)
            dx = ox - cx
            dy = oy - cy
            length = np.hypot(dx, dy)
            if length == 0:
                continue
            angle_deg = np.degrees(np.arctan2(dy, dx))
            rect_local = Polygon(
                [
                    (-length / 2.0, -half_lw_conn),
                    (length / 2.0, -half_lw_conn),
                    (length / 2.0, half_lw_conn),
                    (-length / 2.0, half_lw_conn),
                    (-length / 2.0, -half_lw_conn),
                ]
            )
            lp = rotate(rect_local, angle_deg, origin=(0.0, 0.0))
            lp = translate(lp, xoff=mx, yoff=my)
            lp = lp.buffer(0)
            if not lp.is_empty:
                conn_lines_g1.append(lp)
    conn_lines_g2 = [scale(lp, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for lp in conn_lines_g1]

    group3 = [rotate(translate(g, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for g in group1]
    group4 = [scale(h, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for h in group3]
    holes_g3 = [rotate(translate(h, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for h in holes_g1]
    holes_g4 = [scale(h, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for h in holes_g3]

    lines1 = vert_lines_g1 + conn_lines_g1
    lines2 = vert_lines_g2 + conn_lines_g2
    lines_3 = [rotate(translate(lp, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for lp in lines1]
    lines_4 = [scale(h, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for h in lines_3]

    bridge_90 = group3
    bridge_270 = group4

    # 将每组的圆和线合并为一个图形
    shape_g1 = unary_union(group1 + lines1)
    shape_g2 = unary_union(group2 + lines2)
    shape_g3 = unary_union(group3 + lines_3)
    shape_g4 = unary_union(group4 + lines_4)
    if holes_g1:
        shape_g1 = shape_g1.difference(unary_union(holes_g1))
    if holes_g2:
        shape_g2 = shape_g2.difference(unary_union(holes_g2))
    if holes_g3:
        shape_g3 = shape_g3.difference(unary_union(holes_g3))
    if holes_g4:
        shape_g4 = shape_g4.difference(unary_union(holes_g4))

    def _write_shape(geom, layer_name, color):
        if geom.is_empty:
            return
        doc.layers.add(layer_name, color=color)
        for coords in _geom_to_polygon_coords(geom):
            if len(coords) < 3:
                continue
            pts = [(float(p[0]), float(p[1])) for p in coords]
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer_name})

    _write_shape(shape_g1, "Circle_line_g1", 7)
    _write_shape(shape_g2, "Circle_line_g2", 8)
    _write_shape(shape_g3, "Circle_line_g3", 9)
    _write_shape(shape_g4, "Circle_line_g4", 10)

    layer_names = ["Bridge_spiral_0", "Bridge_spiral_90", "Bridge_spiral_180", "Bridge_spiral_270"]
    for k in range(4):
        doc.layers.add(layer_names[k], color=k + 2)
        arm_dev = get_one_arm_device(k, r_hole, bridge_length, inner_length, width_inner, width_outer)
        arm_geoms = [_poly_to_shapely(p, buffer_zero=True) for p in arm_dev.get_polygons()]
        arm_geoms = [g for g in arm_geoms if g is not None]
        spiral_geoms = get_one_spiral_polygons(k, r_i, r_hole, bridge_length, spiral_width)
        combined = unary_union(arm_geoms + spiral_geoms)
        if hasattr(combined, "make_valid"):
            combined = combined.make_valid()
        combined = combined.buffer(100.0).buffer(-100.0)
        bridge_spiral = combined.difference(center_hole)
        if hasattr(bridge_spiral, "make_valid"):
            bridge_spiral = bridge_spiral.make_valid()
        for coords in _geom_to_polygon_coords(bridge_spiral):
            if len(coords) < 3:
                continue
            points = [(float(p[0]), float(p[1])) for p in coords]
            msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer_names[k]})

    r_start = r_hole + bridge_length
    r_end = r_i + spiral_width / 2.0
    conn_line_width = 10.0
    half_lw = conn_line_width / 2.0
    conn_line_length = r_end - r_start
    doc.layers.add("Spiral_connector", color=11)
    for theta in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]:
        rect_local = Polygon([
            (0.0, -half_lw),
            (conn_line_length, -half_lw),
            (conn_line_length, half_lw),
            (0.0, half_lw),
            (0.0, -half_lw),
        ]).buffer(0)
        lp = rotate(rect_local, np.degrees(theta), origin=(0.0, 0.0))
        lp = translate(lp, xoff=r_start * np.cos(theta), yoff=r_start * np.sin(theta))
        if not lp.is_empty:
            for coords in _geom_to_polygon_coords(lp):
                if len(coords) >= 3:
                    pts = [(float(p[0]), float(p[1])) for p in coords]
                    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Spiral_connector"})
    doc.saveas(output_path)


def generate_au1_polys(
    r_hole,
    bridge_length,
    r_i,
    spiral_width,
    spiral_n,
    spiral_spacing,
    spiral_radial_shift,
    r_start_grid,
    r_end_grid,
    grid_cols,
    grid_rows,
    n_workers,
    slit_width_au1,
    r_i_au1,
    dr1_au1,
    dr2_au1,
    n_layers,
    arc_cols,
    arc_cell_width,
    arc_row_arc_length,
    radius_offset,
    layer_radii,
):
    from collections import defaultdict

    au1_spiral_polys = []
    grid_cell_width = spiral_width / grid_cols
    span_grid = 2 * np.pi * (1.0 + 1.0 / 16.0)
    arms_params_grid = [
        (0, -span_grid),
        (np.pi / 2, np.pi / 2 - span_grid),
        (np.pi, np.pi - span_grid),
        (3 * np.pi / 2, 3 * np.pi / 2 - span_grid),
    ]
    rows_per_chunk = max(1, grid_rows // n_workers)
    grid_args = []
    for theta_start, theta_end in arms_params_grid:
        for row_start in range(0, grid_rows, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, grid_rows)
            if row_start < row_end:
                grid_args.append((
                    theta_start, theta_end, r_start_grid, r_end_grid,
                    grid_rows, grid_cols, grid_cell_width, row_start, row_end
                ))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_build_spiral_grid_chunk, grid_args))
    from collections import defaultdict
    grid_by_col = defaultdict(list)
    for chunk_result in results:
        for col, seg, rect in chunk_result:
            grid_by_col[(col, seg)].append(rect)
    n_fine = 4
    start_skip_micro_max = n_fine * n_fine
    for col_seg_key, polys in grid_by_col.items():
        col_idx, segment = col_seg_key
        i = col_idx + 1
        if col_idx >= 2 and col_idx <= 34 and col_idx % 2 == 0 and col_idx // 2 < 18:
            k = (i - 3) // 2
            start_skip_micro = start_skip_micro_max - k
            if segment >= start_skip_micro:
                merged = unary_union(polys)
                if hasattr(merged, "geoms"):
                    for g in merged.geoms:
                        au1_spiral_polys.append(g)
                else:
                    au1_spiral_polys.append(merged)

    au1_arc_polys = []
    arc_grid_by_layer_col = defaultdict(list)

    span_grid = 2 * np.pi * (1.0 + 1.0 / 16.0)
    arms_params_grid = [
        (0, -span_grid),
        (np.pi / 2, np.pi / 2 - span_grid),
        (np.pi, np.pi - span_grid),
        (3 * np.pi / 2, 3 * np.pi / 2 - span_grid),
    ]
    spiral_n = 17
    spiral_spacing = 4.0
    spiral_radial_shift = 3.0
    shorten_angle = 2.0 * np.pi / 128.0
    line8_ref = 10
    start_per_line_1024 = 2.0 * np.pi / 1024.0

    for layer_idx in range(1, n_layers + 1):
        r_layer_center = layer_radii[layer_idx - 1] + radius_offset
        theta_start = -np.pi / 2 - 2 * np.pi * layer_idx / 16 + np.pi - 1 * np.pi / 96
        theta_end = -np.pi / 2 - 2 * np.pi * (layer_idx + 1) / 16 + np.pi + 1 * np.pi / 96
        arc_total_angle = abs(theta_end - theta_start)
        r_arc = r_layer_center
        arc_rows = int(arc_total_angle * r_arc / arc_row_arc_length)
        for col in range(arc_cols):
            if col >= 2 and col <= 34 and col % 2 == 0 and col // 2 < 18:
                r_col = r_layer_center - arc_cell_width * arc_cols / 2 + (col + 0.5) * arc_cell_width
                for row in range(arc_rows):
                    theta_frac1 = row / arc_rows
                    theta_frac2 = (row + 1) / arc_rows
                    theta1 = theta_start + (theta_end - theta_start) * theta_frac1
                    theta2 = theta_start + (theta_end - theta_start) * theta_frac2
                    pts = [
                        (r_col * np.cos(theta1), r_col * np.sin(theta1)),
                        ((r_col + arc_cell_width) * np.cos(theta1), (r_col + arc_cell_width) * np.sin(theta1)),
                        ((r_col + arc_cell_width) * np.cos(theta2), (r_col + arc_cell_width) * np.sin(theta2)),
                        (r_col * np.cos(theta2), r_col * np.sin(theta2)),
                    ]
                    rect = Polygon(pts).buffer(0)
                    if not rect.is_empty:
                        arc_grid_by_layer_col[(layer_idx, col)].append(rect)
    for (layer_idx, col_idx), polys in arc_grid_by_layer_col.items():
        merged = unary_union(polys)
        if hasattr(merged, "geoms"):
            for g in merged.geoms:
                au1_arc_polys.append(g)
        else:
            au1_arc_polys.append(merged)

    for n in range(1, 10):
        zigzag = generate_arc_connectors_zigzag(
            layer_radii=layer_radii,
            radius_offset=radius_offset,
            arc_cols=arc_cols,
            arc_cell_width=arc_cell_width,
            arc_row_arc_length=arc_row_arc_length,
            from_layer_idx=n,
            to_layer_idx=n + 1,
            col_min=2,
            col_max=34,
            include_col_max=True,
        )
        au1_arc_polys.extend(zigzag)

    au1_arc_conn_rotated = []
    for rot_deg in [0, 90, 180, 270]:
        for p in au1_arc_polys:
            rp = rotate(p, rot_deg, origin=(0, 0))
            if not rp.is_empty:
                au1_arc_conn_rotated.append(rp)
    au1_arc_polys = au1_arc_conn_rotated

    layer_idx = 11
    r_layer_center = layer_radii[layer_idx - 1] + radius_offset + 40

    au1_arc_layer11_group1_start = np.pi - np.pi / 8 + np.pi /32 + np.pi /64
    au1_arc_layer11_group1_end = np.pi + np.pi / 16 + np.pi /32 + np.pi /64
    au1_arc_layer11_group2_start = np.pi /2 - np.pi/8
    au1_arc_layer11_group2_end = np.pi /8 + np.pi / 64
    au1_arc_layer11_group3_start = - 3 * np.pi/8
    au1_arc_layer11_group3_end = np.pi /8 - np.pi / 64
    arc_grid_layer11_by_col = defaultdict(list)

    for group_start, group_end in [(au1_arc_layer11_group1_start, au1_arc_layer11_group1_end), (au1_arc_layer11_group2_start, au1_arc_layer11_group2_end), (au1_arc_layer11_group3_start, au1_arc_layer11_group3_end)]:
        if group_start == group_end:
            continue
        arc_total_angle = abs(group_end - group_start)
        arc_rows = int(arc_total_angle * r_layer_center / arc_row_arc_length)
        for col in range(arc_cols):
            if col >= 2 and col <= 34 and col % 2 == 0 and col // 2 < 18:
                r_col = r_layer_center - arc_cell_width * arc_cols / 2 + (col + 0.5) * arc_cell_width
                for row in range(arc_rows):
                    theta_frac1 = row / arc_rows
                    theta_frac2 = (row + 1) / arc_rows
                    theta1 = group_start + (group_end - group_start) * theta_frac1
                    theta2 = group_start + (group_end - group_start) * theta_frac2
                    pts = [
                        (r_col * np.cos(theta1), r_col * np.sin(theta1)),
                        ((r_col + arc_cell_width) * np.cos(theta1), (r_col + arc_cell_width) * np.sin(theta1)),
                        ((r_col + arc_cell_width) * np.cos(theta2), (r_col + arc_cell_width) * np.sin(theta2)),
                        (r_col * np.cos(theta2), r_col * np.sin(theta2)),
                    ]
                    rect = Polygon(pts).buffer(0)
                    if not rect.is_empty:
                        arc_grid_layer11_by_col[col].append(rect)

    for col_idx, polys in arc_grid_layer11_by_col.items():
        merged = unary_union(polys)
        if hasattr(merged, "geoms"):
            for g in merged.geoms:
                au1_arc_polys.append(g)
        else:
            au1_arc_polys.append(merged)

    au1_arc_connectors = generate_arc_connectors(
        layer_radii=layer_radii,
        radius_offset=radius_offset,
        arc_cols=arc_cols,
        arc_cell_width=arc_cell_width,
        arc_row_arc_length=arc_row_arc_length,
        from_layer_idx=10,
        to_layer_idx=11,
        col_min=2,
        col_max=34,
        include_col_max=True,
    )
    au1_arc_connectors_rotated = []
    for rot_deg in [0, 90, 180, 270]:
        for p in au1_arc_connectors:
            rp = rotate(p, rot_deg, origin=(0, 0))
            if not rp.is_empty:
                au1_arc_connectors_rotated.append(rp)
    au1_arc_polys.extend(au1_arc_connectors_rotated)

    au1_conn_to_scale_circle_polys = []

    return au1_spiral_polys, au1_arc_polys, au1_conn_to_scale_circle_polys



def generate_au2_polys(
    r_hole,
    bridge_length,
    r_i,
    spiral_width,
    spiral_n,
    spiral_spacing,
    spiral_radial_shift,
    r_start_grid,
    r_end_grid,
    grid_cols,
    grid_rows,
    n_workers,
    slit_width_au1,
    r_i_au1,
    dr1_au1,
    dr2_au1,
    n_layers,
    arc_cols,
    arc_cell_width,
    arc_row_arc_length,
    radius_offset,
    layer_radii,
):
    from collections import defaultdict

    au2_spiral_polys = []
    grid_cell_width = spiral_width / grid_cols
    span_grid = 2 * np.pi * (1.0 + 1.0 / 16.0)
    arms_params_grid = [
        (0, -span_grid),
        (np.pi / 2, np.pi / 2 - span_grid),
        (np.pi, np.pi - span_grid),
        (3 * np.pi / 2, 3 * np.pi / 2 - span_grid),
    ]
    rows_per_chunk = max(1, grid_rows // n_workers)
    grid_args = []
    for theta_start, theta_end in arms_params_grid:
        for row_start in range(0, grid_rows, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, grid_rows)
            if row_start < row_end:
                grid_args.append((
                    theta_start, theta_end, r_start_grid, r_end_grid,
                    grid_rows, grid_cols, grid_cell_width, row_start, row_end
                ))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_build_spiral_grid_chunk, grid_args))
    from collections import defaultdict
    grid_by_col = defaultdict(list)
    for chunk_result in results:
        for col, seg, rect in chunk_result:
            grid_by_col[(col, seg)].append(rect)
    n_fine = 4
    start_skip_micro_max = n_fine * n_fine
    for col_seg_key, polys in grid_by_col.items():
        col_idx, segment = col_seg_key
        i = col_idx + 1
        if col_idx >= 2 and col_idx < 34 and col_idx % 2 == 0 and col_idx // 2 < 18:
            k = (i - 3) // 2
            start_skip_micro = start_skip_micro_max - k
            if segment >= start_skip_micro:
                merged = unary_union(polys)
                if hasattr(merged, "geoms"):
                    for g in merged.geoms:
                        au2_spiral_polys.append(g)
                else:
                    au2_spiral_polys.append(merged)

    au2_arc_polys = []
    arc_grid_by_layer_col = defaultdict(list)

    span_grid = 2 * np.pi * (1.0 + 1.0 / 16.0)
    arms_params_grid = [
        (0, -span_grid),
        (np.pi / 2, np.pi / 2 - span_grid),
        (np.pi, np.pi - span_grid),
        (3 * np.pi / 2, 3 * np.pi / 2 - span_grid),
    ]
    spiral_n = 17
    spiral_spacing = 4.0
    spiral_radial_shift = 3.0
    shorten_angle = 2.0 * np.pi / 128.0
    line8_ref = 10
    start_per_line_1024 = 2.0 * np.pi / 1024.0

    for layer_idx in range(1, n_layers + 1):
        r_layer_center = layer_radii[layer_idx - 1] + radius_offset
        theta_start = -np.pi / 2 - 2 * np.pi * layer_idx / 16 + np.pi - 1 * np.pi / 96
        theta_end = -np.pi / 2 - 2 * np.pi * (layer_idx + 1) / 16 + np.pi + 1 * np.pi / 96
        arc_total_angle = abs(theta_end - theta_start)
        r_arc = r_layer_center
        arc_rows = int(arc_total_angle * r_arc / arc_row_arc_length)
        for col in range(arc_cols):
            if col >= 2 and col < 34 and col % 2 == 0 and col // 2 < 18:
                r_col = r_layer_center - arc_cell_width * arc_cols / 2 + (col + 0.5) * arc_cell_width
                for row in range(arc_rows):
                    theta_frac1 = row / arc_rows
                    theta_frac2 = (row + 1) / arc_rows
                    theta1 = theta_start + (theta_end - theta_start) * theta_frac1
                    theta2 = theta_start + (theta_end - theta_start) * theta_frac2
                    pts = [
                        (r_col * np.cos(theta1), r_col * np.sin(theta1)),
                        ((r_col + arc_cell_width) * np.cos(theta1), (r_col + arc_cell_width) * np.sin(theta1)),
                        ((r_col + arc_cell_width) * np.cos(theta2), (r_col + arc_cell_width) * np.sin(theta2)),
                        (r_col * np.cos(theta2), r_col * np.sin(theta2)),
                    ]
                    rect = Polygon(pts).buffer(0)
                    if not rect.is_empty:
                        arc_grid_by_layer_col[(layer_idx, col)].append(rect)
    for (layer_idx, col_idx), polys in arc_grid_by_layer_col.items():
        merged = unary_union(polys)
        if hasattr(merged, "geoms"):
            for g in merged.geoms:
                au2_arc_polys.append(g)
        else:
            au2_arc_polys.append(merged)

    for n in range(1, 10):
        zigzag = generate_arc_connectors_zigzag(
            layer_radii=layer_radii,
            radius_offset=radius_offset,
            arc_cols=arc_cols,
            arc_cell_width=arc_cell_width,
            arc_row_arc_length=arc_row_arc_length,
            from_layer_idx=n,
            to_layer_idx=n + 1,
            col_min=2,
            col_max=34,
            include_col_max=False,
        )
        au2_arc_polys.extend(zigzag)

    au2_arc_conn_rotated = []
    for rot_deg in [0, 90, 180, 270]:
        for p in au2_arc_polys:
            rp = rotate(p, rot_deg, origin=(0, 0))
            if not rp.is_empty:
                au2_arc_conn_rotated.append(rp)
    au2_arc_polys = au2_arc_conn_rotated

    layer_idx = 11
    r_layer_center = layer_radii[layer_idx - 1] + radius_offset + 40

    au2_arc_layer11_group1_start = np.pi - np.pi / 8 + np.pi /32 + np.pi /64
    au2_arc_layer11_group1_end = np.pi + np.pi / 16 + np.pi /32 + np.pi /64
    au2_arc_layer11_group2_start = np.pi /2 - np.pi/8
    au2_arc_layer11_group2_end = np.pi /8 + np.pi / 64
    au2_arc_layer11_group3_start = - 3 * np.pi/8
    au2_arc_layer11_group3_end = np.pi /8 - np.pi / 64

    arc_grid_layer11_by_col = defaultdict(list)

    for group_start, group_end in [(au2_arc_layer11_group1_start, au2_arc_layer11_group1_end), (au2_arc_layer11_group2_start, au2_arc_layer11_group2_end), (au2_arc_layer11_group3_start, au2_arc_layer11_group3_end)]:
        if group_start == group_end:
            continue
        arc_total_angle = abs(group_end - group_start)
        arc_rows = int(arc_total_angle * r_layer_center / arc_row_arc_length)
        for col in range(arc_cols):
            if col >= 2 and col < 34 and col % 2 == 0 and col // 2 < 18:
                r_col = r_layer_center - arc_cell_width * arc_cols / 2 + (col + 0.5) * arc_cell_width
                for row in range(arc_rows):
                    theta_frac1 = row / arc_rows
                    theta_frac2 = (row + 1) / arc_rows
                    theta1 = group_start + (group_end - group_start) * theta_frac1
                    theta2 = group_start + (group_end - group_start) * theta_frac2
                    pts = [
                        (r_col * np.cos(theta1), r_col * np.sin(theta1)),
                        ((r_col + arc_cell_width) * np.cos(theta1), (r_col + arc_cell_width) * np.sin(theta1)),
                        ((r_col + arc_cell_width) * np.cos(theta2), (r_col + arc_cell_width) * np.sin(theta2)),
                        (r_col * np.cos(theta2), r_col * np.sin(theta2)),
                    ]
                    rect = Polygon(pts).buffer(0)
                    if not rect.is_empty:
                        arc_grid_layer11_by_col[col].append(rect)

    for col_idx, polys in arc_grid_layer11_by_col.items():
        merged = unary_union(polys)
        if hasattr(merged, "geoms"):
            for g in merged.geoms:
                au2_arc_polys.append(g)
        else:
            au2_arc_polys.append(merged)

    au2_arc_connectors = generate_arc_connectors(
        layer_radii=layer_radii,
        radius_offset=radius_offset,
        arc_cols=arc_cols,
        arc_cell_width=arc_cell_width,
        arc_row_arc_length=arc_row_arc_length,
        from_layer_idx=10,
        to_layer_idx=11,
        col_min=2,
        col_max=34,
        include_col_max=False,
    )
    au2_arc_connectors_rotated = []
    for rot_deg in [0, 90, 180, 270]:
        for p in au2_arc_connectors:
            rp = rotate(p, rot_deg, origin=(0, 0))
            if not rp.is_empty:
                au2_arc_connectors_rotated.append(rp)
    au2_arc_polys.extend(au2_arc_connectors_rotated)

    return au2_spiral_polys, au2_arc_polys




def write_woio_dxf_triangle_merged(
    D,
    output_path,
    merged_output_path,
    r_i,
    r_o,
    center_hole_diameter,
    bridge_length,
    inner_length,
    width_inner,
    width_outer,
    spiral_width,
):
    polys = D.get_polygons(by_spec=True)
    outline_geoms = [_poly_to_shapely(p, buffer_zero=True) for p in polys.get((0, 0), [])]
    kirigami_geoms = [_poly_to_shapely(p, buffer_zero=False) for p in polys.get((1, 0), [])]
    outline_geoms = [g for g in outline_geoms if g is not None]
    kirigami_geoms = [g for g in kirigami_geoms if g is not None]
    outline_union = unary_union(outline_geoms) if outline_geoms else Polygon()
    if hasattr(outline_union, "make_valid"):
        outline_union = outline_union.make_valid()
    tol = 0.1
    kirigami_buffed = [g.buffer(tol) for g in kirigami_geoms if not g.is_empty]
    kirigami_union = unary_union(kirigami_buffed) if kirigami_buffed else Polygon()
    if hasattr(kirigami_union, "make_valid"):
        kirigami_union = kirigami_union.make_valid()
    ring = outline_union.difference(kirigami_union)
    if hasattr(ring, "make_valid"):
        ring = ring.make_valid()

    r_hole = center_hole_diameter / 2
    center_hole = Point(0, 0).buffer(r_hole)

    side = 80.0
    half_side = side / 2.0
    square_coords = [
        (-half_side, -half_side),
        (half_side, -half_side),
        (half_side, half_side),
        (-half_side, half_side),
        (-half_side, -half_side),
    ]
    square_poly = Polygon(square_coords).buffer(0)
    inner_circle = Point(0, 0).buffer(30.0)
    center_shape = square_poly.difference(inner_circle)
    if hasattr(center_shape, "make_valid"):
        center_shape = center_shape.make_valid()

    trapezoid_base = Polygon([(940.0, 40.0), (1030.0, 40.0), (1030.0, 120.0), (1000.0, 120.0)]).buffer(0)
    trapezoid_inner_base = Polygon([
        (960.0, 40.0),
        (1020.0, 40.0),
        (1020.0, 110.0),
        (1008.0, 110.0),
    ]).buffer(0)
    trapezoid_base = trapezoid_base.difference(trapezoid_inner_base)
    trapezoids_980_40 = [rotate(trapezoid_base, k * 90.0, origin=(0.0, 0.0)) for k in range(4)]
    square_980_40 = unary_union(trapezoids_980_40)

    base_x = 450.0
    step = 50.0
    n_circles = 8
    group1 = []
    holes_g1 = []
    for base_y in (60.0, -60.0):
        for i in range(n_circles):
            cx = base_x - i * step
            cy = base_y
            c = Point(cx, cy).buffer(20.0).difference(Point(cx, cy).buffer(10.0))
            if hasattr(c, "make_valid"):
                c = c.make_valid()
            group1.append(c)
            holes_g1.append(Point(cx, cy).buffer(10.0))
            dx_outer = 25.0
            dy_outer = 25.0 * np.sqrt(3.0) if base_y > 0 else -25.0 * np.sqrt(3.0)
            ox = cx + dx_outer
            oy = cy + dy_outer
            c2 = Point(ox, oy).buffer(20.0).difference(Point(ox, oy).buffer(10.0))
            if hasattr(c2, "make_valid"):
                c2 = c2.make_valid()
            group1.append(c2)
            holes_g1.append(Point(ox, oy).buffer(10.0))
    group2 = [scale(g, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for g in group1]
    holes_g2 = [scale(h, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for h in holes_g1]

    line_width_vert = 40.0
    half_lw_vert = line_width_vert / 2.0
    vert_lines_g1 = []
    vert_lines_g2 = []
    for sign_x in (1.0, -1.0):
        for base_y in (60.0, -60.0):
            for i in range(n_circles):
                cx = sign_x * (base_x - i * step)
                if base_y > 0:
                    y1 = base_y + 2
                    y2 = 40.0
                else:
                    y1 = base_y - 2
                    y2 = -40.0
                x0 = cx
                rect_coords = [
                    (x0 - half_lw_vert, y1),
                    (x0 + half_lw_vert, y1),
                    (x0 + half_lw_vert, y2),
                    (x0 - half_lw_vert, y2),
                    (x0 - half_lw_vert, y1),
                ]
                lp = Polygon(rect_coords).buffer(0)
                if lp.is_empty:
                    continue
                if cx > 0:
                    vert_lines_g1.append(lp)
                else:
                    vert_lines_g2.append(lp)

    line_width_conn = 40.0
    half_lw_conn = line_width_conn / 2.0
    conn_lines_g1 = []
    for base_y in (60.0, -60.0):
        for i in range(n_circles):
            cx = base_x - i * step
            cy = base_y
            ox = cx + 25.0
            oy = cy + (25.0 * np.sqrt(3.0) if base_y > 0 else -25.0 * np.sqrt(3.0))
            mx = 0.5 * (cx + ox)
            my = 0.5 * (cy + oy)
            dx = ox - cx
            dy = oy - cy
            length = np.hypot(dx, dy)
            if length == 0:
                continue
            angle_deg = np.degrees(np.arctan2(dy, dx))
            rect_local = Polygon([
                (-length / 2.0, -half_lw_conn),
                (length / 2.0, -half_lw_conn),
                (length / 2.0, half_lw_conn),
                (-length / 2.0, half_lw_conn),
                (-length / 2.0, -half_lw_conn),
            ])
            lp = rotate(rect_local, angle_deg, origin=(0.0, 0.0))
            lp = translate(lp, xoff=mx, yoff=my)
            lp = lp.buffer(0)
            if not lp.is_empty:
                conn_lines_g1.append(lp)
    conn_lines_g2 = [scale(lp, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for lp in conn_lines_g1]

    group3 = [rotate(translate(g, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for g in group1]
    group4 = [scale(h, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for h in group3]
    holes_g3 = [rotate(translate(h, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for h in holes_g1]
    holes_g4 = [scale(h, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for h in holes_g3]

    outer_centers_g1 = []
    for base_y in (60.0, -60.0):
        for i in range(n_circles):
            cx = base_x - i * step
            cy = base_y
            dx_outer = 25.0
            dy_outer = 25.0 * np.sqrt(3.0) if base_y > 0 else -25.0 * np.sqrt(3.0)
            ox = cx + dx_outer
            oy = cy + dy_outer
            outer_centers_g1.append((ox, oy))
    outer_centers_g2 = [(-ox, oy) for ox, oy in outer_centers_g1]
    outer_centers_g3 = [(oy, -(ox + 420.0)) for ox, oy in outer_centers_g1]
    outer_centers_g4 = [(ox, -oy) for ox, oy in outer_centers_g3]
    au1_circle_radius = 15.0
    au1_circles = []
    for xc, yc in outer_centers_g1 + outer_centers_g2 + outer_centers_g3 + outer_centers_g4:
        circ = Point(xc, yc).buffer(au1_circle_radius)
        if not circ.is_empty:
            au1_circles.append(circ)

    au1_line_width = 2.0
    au1_half_lw = au1_line_width / 2.0
    au1_conn_two_circles_polys = []
    for base_y in (60.0, -60.0):
        for i in range(n_circles):
            cx = base_x - i * step
            cy = base_y
            ox = cx + 25.0
            oy = cy + (25.0 * np.sqrt(3.0) if base_y > 0 else -25.0 * np.sqrt(3.0))
            mx = 0.5 * (cx + ox)
            my = 0.5 * (cy + oy)
            dx = ox - cx
            dy = oy - cy
            length = np.hypot(dx, dy)
            if length == 0:
                continue
            angle_deg = np.degrees(np.arctan2(dy, dx))
            rect_local = Polygon([
                (-length / 2.0, -au1_half_lw),
                (length / 2.0, -au1_half_lw),
                (length / 2.0, au1_half_lw),
                (-length / 2.0, au1_half_lw),
                (-length / 2.0, -au1_half_lw),
            ])
            lp = rotate(rect_local, angle_deg, origin=(0.0, 0.0))
            lp = translate(lp, xoff=mx, yoff=my)
            lp = lp.buffer(0)
            if not lp.is_empty:
                au1_conn_two_circles_polys.append(lp)
    au1_conn_two_circles_g2 = [scale(g, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for g in au1_conn_two_circles_polys]
    au1_conn_two_circles_g3 = [rotate(translate(g, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for g in au1_conn_two_circles_polys]
    au1_conn_two_circles_g4 = [scale(g, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for g in au1_conn_two_circles_g3]
    au1_conn_two_circles_polys.extend(au1_conn_two_circles_g2)
    au1_conn_two_circles_polys.extend(au1_conn_two_circles_g3)
    au1_conn_two_circles_polys.extend(au1_conn_two_circles_g4)

    inner_base_y = 60.0
    scale_y_vals = [4.0 * (j + 1) for j in range(8)]
    au1_conn_to_scale_polys = []
    for base_y in (inner_base_y, -inner_base_y):
        for i in range(n_circles):
            x1 = base_x - i * step
            cx, cy = x1, base_y
            scale_y = scale_y_vals[7 - i] if base_y > 0 else -scale_y_vals[7 - i]
            ln = box(cx - au1_half_lw, min(cy, scale_y), cx + au1_half_lw, max(cy, scale_y))
            if not ln.is_empty:
                au1_conn_to_scale_polys.append(ln)
    au1_conn_to_scale_g2 = [scale(g, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for g in au1_conn_to_scale_polys]
    au1_conn_to_scale_g3 = [rotate(translate(g, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for g in au1_conn_to_scale_polys]
    au1_conn_to_scale_g4 = [scale(g, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for g in au1_conn_to_scale_g3]
    au1_conn_to_scale_polys.extend(au1_conn_to_scale_g2)
    au1_conn_to_scale_polys.extend(au1_conn_to_scale_g3)
    au1_conn_to_scale_polys.extend(au1_conn_to_scale_g4)

    circle_radius = 2.0
    au1_conn_to_scale_circle_polys = []
    for base_y in (inner_base_y, -inner_base_y):
        for i in range(n_circles):
            x1 = base_x - i * step
            cx, cy = x1, base_y
            scale_y = scale_y_vals[7 - i] if base_y > 0 else -scale_y_vals[7 - i]
            c1 = Point(cx, cy).buffer(circle_radius)
            c2 = Point(cx, scale_y).buffer(circle_radius)
            if not c1.is_empty:
                au1_conn_to_scale_circle_polys.append(c1)
            if not c2.is_empty:
                au1_conn_to_scale_circle_polys.append(c2)
    au1_conn_to_scale_circle_g2 = [scale(g, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for g in au1_conn_to_scale_circle_polys]
    au1_conn_to_scale_circle_g3 = [rotate(translate(g, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for g in au1_conn_to_scale_circle_polys]
    au1_conn_to_scale_circle_g4 = [scale(g, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for g in au1_conn_to_scale_circle_g3]
    au1_conn_to_scale_circle_polys.extend(au1_conn_to_scale_circle_g2)
    au1_conn_to_scale_circle_polys.extend(au1_conn_to_scale_circle_g3)
    au1_conn_to_scale_circle_polys.extend(au1_conn_to_scale_circle_g4)

    au2_circle_radius_small = 2.0
    au2_circle_radius_large = 15.0
    au2_conn_to_scale_circle_polys = []
    for base_y in (inner_base_y, -inner_base_y):
        for i in range(n_circles):
            x1 = base_x - i * step
            cx, cy = x1, base_y
            scale_y = scale_y_vals[7 - i] if base_y > 0 else -scale_y_vals[7 - i]
            c1 = Point(cx, cy).buffer(au2_circle_radius_large)
            c2 = Point(cx, scale_y).buffer(au2_circle_radius_small)
            if not c1.is_empty:
                au2_conn_to_scale_circle_polys.append(c1)
            if not c2.is_empty:
                au2_conn_to_scale_circle_polys.append(c2)
    au2_conn_to_scale_circle_g2 = [scale(g, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)) for g in au2_conn_to_scale_circle_polys]
    au2_conn_to_scale_circle_g3 = [rotate(translate(g, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for g in au2_conn_to_scale_circle_polys]
    au2_conn_to_scale_circle_g4 = [scale(g, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for g in au2_conn_to_scale_circle_g3]
    au2_conn_to_scale_circle_polys.extend(au2_conn_to_scale_circle_g2)
    au2_conn_to_scale_circle_polys.extend(au2_conn_to_scale_circle_g3)
    au2_conn_to_scale_circle_polys.extend(au2_conn_to_scale_circle_g4)

    lines1 = vert_lines_g1 + conn_lines_g1
    lines2 = vert_lines_g2 + conn_lines_g2
    lines_3 = [rotate(translate(lp, xoff=420.0, yoff=0.0), -90.0, origin=(0.0, 0.0)) for lp in lines1]
    lines_4 = [scale(h, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0)) for h in lines_3]

    shape_g1 = unary_union(group1 + lines1)
    shape_g2 = unary_union(group2 + lines2)
    shape_g3 = unary_union(group3 + lines_3)
    shape_g4 = unary_union(group4 + lines_4)
    if holes_g1:
        shape_g1 = shape_g1.difference(unary_union(holes_g1))
    if holes_g2:
        shape_g2 = shape_g2.difference(unary_union(holes_g2))
    if holes_g3:
        shape_g3 = shape_g3.difference(unary_union(holes_g3))
    if holes_g4:
        shape_g4 = shape_g4.difference(unary_union(holes_g4))

    bridge_spirals = []
    for k in range(4):
        arm_dev = get_one_arm_device(k, r_hole, bridge_length, inner_length, width_inner, width_outer)
        arm_geoms = [_poly_to_shapely(p, buffer_zero=True) for p in arm_dev.get_polygons()]
        arm_geoms = [g for g in arm_geoms if g is not None]
        spiral_geoms = get_one_spiral_polygons(k, r_i, r_hole, bridge_length, spiral_width)
        combined = unary_union(arm_geoms + spiral_geoms)
        if hasattr(combined, "make_valid"):
            combined = combined.make_valid()
        combined = combined.buffer(100.0).buffer(-100.0)
        bridge_spiral = combined.difference(center_hole)
        if hasattr(bridge_spiral, "make_valid"):
            bridge_spiral = bridge_spiral.make_valid()
        bridge_spirals.append(bridge_spiral)

    au1_conn_to_scale_circle_polys_local = au1_conn_to_scale_circle_polys

    au1_spiral_polys, au1_arc_polys, au1_conn_to_scale_circle_polys = generate_au1_polys(
        r_hole=r_hole,
        bridge_length=bridge_length,
        r_i=r_i,
        spiral_width=spiral_width,
        spiral_n=17,
        spiral_spacing=4.0,
        spiral_radial_shift=3.0,
        r_start_grid=r_hole + bridge_length,
        r_end_grid=r_i + spiral_width / 2.0,
        grid_cols=40,
        grid_rows=450,
        n_workers=30,
        slit_width_au1=5.0,
        r_i_au1=1500.0,
        dr1_au1=90.0,
        dr2_au1=90.0,
        n_layers=10,
        arc_cols=40,
        arc_cell_width=90/40,
        arc_row_arc_length=90/40,
        radius_offset=-40,
        layer_radii=[1500.0 + 90.0 + i * (90.0 + 5.0) for i in range(11)],
    )

    au1_conn_to_scale_circle_polys = au1_conn_to_scale_circle_polys_local

    au2_spiral_polys, au2_arc_polys = generate_au2_polys(
        r_hole=r_hole,
        bridge_length=bridge_length,
        r_i=r_i,
        spiral_width=spiral_width,
        spiral_n=17,
        spiral_spacing=4.0,
        spiral_radial_shift=3.0,
        r_start_grid=r_hole + bridge_length,
        r_end_grid=r_i + spiral_width / 2.0,
        grid_cols=40,
        grid_rows=450,
        n_workers=30,
        slit_width_au1=5.0,
        r_i_au1=1500.0,
        dr1_au1=90.0,
        dr2_au1=90.0,
        n_layers=10,
        arc_cols=40,
        arc_cell_width=90/40,
        arc_row_arc_length=90/40,
        radius_offset=-40,
        layer_radii=[1500.0 + 90.0 + i * (90.0 + 5.0) for i in range(11)],
    )

    r_start = r_hole + bridge_length
    r_end = r_i + spiral_width / 2.0

    conn_line_width = 10.0
    half_lw = conn_line_width / 2.0
    conn_line_length = r_end - r_start
    conn_lines = []
    for theta in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]:
        rect_local = Polygon([
            (0.0, -half_lw),
            (conn_line_length, -half_lw),
            (conn_line_length, half_lw),
            (0.0, half_lw),
            (0.0, -half_lw),
        ]).buffer(0)
        lp = rotate(rect_local, np.degrees(theta), origin=(0.0, 0.0))
        lp = translate(lp, xoff=r_start * np.cos(theta), yoff=r_start * np.sin(theta))
        if not lp.is_empty:
            conn_lines.append(lp)

    r_outer = 2500.0
    r_bc_inward = 20.0
    r_bc = r_outer - r_bc_inward
    angle_start = np.pi / 2.0
    a_angles = [
        angle_start + 3.0 / 16.0 * 2.0 * np.pi,
        angle_start + 1.0 / 16.0 * 2.0 * np.pi,
        angle_start - 3.0 / 16.0 * 2.0 * np.pi,
        angle_start - 1.0 / 16.0 * 2.0 * np.pi,
    ]
    arc_64 = 2.0 * np.pi / 64.0
    bridge_extend = 150.0
    n_arc_pts = 17
    bridge_polys = []
    b_prime_list = []
    c_prime_list = []
    for a_angle in a_angles:
        b_angle = a_angle + arc_64
        c_angle = a_angle - arc_64
        b_pt = (r_bc * np.cos(b_angle), r_bc * np.sin(b_angle))
        c_pt = (r_bc * np.cos(c_angle), r_bc * np.sin(c_angle))
        arc_angles = np.linspace(b_angle, c_angle, num=n_arc_pts)
        arc_pts = [(r_bc * np.cos(t), r_bc * np.sin(t)) for t in arc_angles]
        b_out = ((r_outer + bridge_extend) * np.cos(b_angle), (r_outer + bridge_extend) * np.sin(b_angle))
        c_out = ((r_outer + bridge_extend) * np.cos(c_angle), (r_outer + bridge_extend) * np.sin(c_angle))
        b_prime_list.append(b_out)
        c_prime_list.append(c_out)
        poly_pts = [b_pt] + arc_pts[1:-1] + [c_pt, c_out, b_out]
        p = Polygon(poly_pts).buffer(0)
        if not p.is_empty:
            bridge_polys.append(p)

    保留 = unary_union([ring, center_shape, square_980_40, shape_g1, shape_g2, shape_g3, shape_g4] + bridge_spirals + conn_lines)
    if hasattr(保留, "make_valid"):
        保留 = 保留.make_valid()

    outer_radius = 2615.0
    outer = Point(0, 0).buffer(outer_radius)
    if hasattr(outer, "make_valid"):
        outer = outer.make_valid()

    镂空 = outer.difference(保留)
    if hasattr(镂空, "make_valid"):
        镂空 = 镂空.make_valid()
    merged_circle = outer.difference(镂空)
    if hasattr(merged_circle, "make_valid"):
        merged_circle = merged_circle.make_valid()

    xmin, ymin, xmax, ymax = -4000.0, -2500.0, 4000.0, 6000.0
    clip_box = box(xmin, ymin, xmax, ymax)
    circle_clip = box(xmin, ymin, xmax, 2615.0)
    merged_circle = merged_circle.intersection(circle_clip)
    if hasattr(merged_circle, "make_valid"):
        merged_circle = merged_circle.make_valid()

    rect_base = [(-3600.0, 6000.0), (-3400.0, 6000.0), (-3400.0, 3000.0), (-3600.0, 3000.0)]
    rect_spacing = 1000.0
    n_rects = 8
    rect_buffer = 75.0
    rect_polys = []
    for i in range(n_rects):
        offset = i * rect_spacing
        pts = [(p[0] + offset, p[1]) for p in rect_base]
        r = Polygon(pts).buffer(0).buffer(rect_buffer, join_style=2)
        if not r.is_empty:
            r = r.intersection(clip_box)
            if not r.is_empty:
                rect_polys.append(r)

    au1_rect_polys = []
    au2_rect_polys = []
    grid_cols = 40
    grid_rows = 1
    for rect_idx, rect in enumerate(rect_polys):
        bounds = rect.bounds
        x_min, y_min, x_max, y_max = bounds
        cell_width = (x_max - x_min) / grid_cols
        cell_height = (y_max - y_min) / grid_rows
        for col in range(grid_cols):
            if col >= 4 and col % 2 == 0 and col <= 34:
                x1 = x_min + col * cell_width
                x2 = x1 + cell_width
                y1 = y_min
                y2 = y_min + cell_height
                grid_cell = box(x1, y1, x2, y2)
                grid_cell = grid_cell.intersection(rect)
                if not grid_cell.is_empty:
                    if rect_idx % 2 == 0:
                        au1_rect_polys.append(grid_cell)
                    elif col < 34:
                        au2_rect_polys.append(grid_cell)

    def bottom_vertices(poly):
        if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
            return None, None
        if poly.geom_type == "MultiPolygon":
            geom = max(poly.geoms, key=lambda g: g.area)
        else:
            geom = poly
        coords = list(geom.exterior.coords)[:-1]
        if len(coords) < 2:
            return None, None
        y_min = min(c[1] for c in coords)
        tol = 1e-6
        bottom_pts = [c for c in coords if abs(c[1] - y_min) <= tol]
        bottom_pts.sort(key=lambda p: p[0])
        return tuple(bottom_pts[0]), tuple(bottom_pts[-1])

    bridge_idx_for_rect = [0, 0, 1, 1, 3, 3, 2, 2]
    rect_bridge_connector_polys = []
    for i in range(len(rect_polys)):
        v1, v2 = bottom_vertices(rect_polys[i])
        if v1 is None or v2 is None:
            continue
        bridge_idx = bridge_idx_for_rect[i]
        b_prime = b_prime_list[bridge_idx]
        c_prime = c_prime_list[bridge_idx]
        quad_pts = [v1, b_prime, c_prime, v2]
        q = Polygon(quad_pts).buffer(0)
        if not q.is_empty:
            rect_bridge_connector_polys.append(q)

    def _rect_geom(poly):
        if poly.geom_type == "MultiPolygon":
            return max(poly.geoms, key=lambda g: g.area)
        return poly

    rect_horizontal_bridge_width = 200.0
    n_y_levels = 8
    n_gaps = 7
    rect_left_x = []
    rect_right_x = []
    rect_y_min = float("inf")
    rect_y_max = float("-inf")
    for i in range(len(rect_polys)):
        g = _rect_geom(rect_polys[i])
        coords = list(g.exterior.coords)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        rect_left_x.append(min(xs))
        rect_right_x.append(max(xs))
        rect_y_min = min(rect_y_min, min(ys))
        rect_y_max = max(rect_y_max, max(ys))
    half_w = rect_horizontal_bridge_width / 2.0
    y_levels = np.linspace(rect_y_min + half_w, rect_y_max - half_w, n_y_levels)
    rect_horizontal_bridge_polys = []
    for y_center in y_levels:
        y_lo = y_center - half_w
        y_hi = y_center + half_w
        for i in range(n_gaps):
            if i + 1 >= len(rect_polys):
                break
            right_x = rect_right_x[i]
            left_x = rect_left_x[i + 1]
            if right_x >= left_x:
                continue
            br = box(right_x, y_lo, left_x, y_hi)
            if not br.is_empty:
                rect_horizontal_bridge_polys.append(br)

    scale_line_width = 2.0
    scale_line_half = scale_line_width / 2.0
    scale_n = 8
    scale_line_start_offset_vertical = 420.0
    scale_x_end_base = 950.0
    scale_lines_polys = []
    for a in range(1, scale_n + 1):
        x2_neg = scale_x_end_base - 4.0 * a
        x2_pos = scale_x_end_base + 4.0 * a
        x1_neg = 50.0 + 50.0 * a
        y_neg = 1.0 - 4.0 * a
        ln_neg = box(x1_neg, y_neg - scale_line_half, x2_neg, y_neg + scale_line_half)
        scale_lines_polys.append(ln_neg)
        x1_pos = 50.0 + 50.0 * a
        y_pos = -1.0 + 4.0 * a
        ln_pos = box(x1_pos, y_pos - scale_line_half, x2_pos, y_pos + scale_line_half)
        scale_lines_polys.append(ln_pos)
    for g in scale_lines_polys[: 2 * scale_n]:
        rot = rotate(g, 180.0, origin=(0.0, 0.0))
        if not rot.is_empty:
            scale_lines_polys.append(rot)
    for a in range(1, scale_n + 1):
        x2_neg = scale_x_end_base - 4.0 * a
        x2_pos = scale_x_end_base + 4.0 * a
        x1_vert = 50.0 + 50.0 * a + scale_line_start_offset_vertical
        y_neg = 1.0 - 4.0 * a
        ln_neg = box(x1_vert, y_neg - scale_line_half, x2_neg, y_neg + scale_line_half)
        rot = rotate(ln_neg, -90.0, origin=(0.0, 0.0))
        if not rot.is_empty:
            scale_lines_polys.append(rot)
        y_pos = -1.0 + 4.0 * a
        ln_pos = box(x1_vert, y_pos - scale_line_half, x2_pos, y_pos + scale_line_half)
        rot = rotate(ln_pos, -90.0, origin=(0.0, 0.0))
        if not rot.is_empty:
            scale_lines_polys.append(rot)
    for a in range(1, scale_n + 1):
        x2_neg = scale_x_end_base - 4.0 * a
        x2_pos = scale_x_end_base + 4.0 * a
        x1_vert = 50.0 + 50.0 * a + scale_line_start_offset_vertical
        y_neg = 1.0 - 4.0 * a
        ln_neg = box(x1_vert, y_neg - scale_line_half, x2_neg, y_neg + scale_line_half)
        rot = rotate(ln_neg, 90.0, origin=(0.0, 0.0))
        if not rot.is_empty:
            scale_lines_polys.append(rot)
        y_pos = -1.0 + 4.0 * a
        ln_pos = box(x1_vert, y_pos - scale_line_half, x2_pos, y_pos + scale_line_half)
        rot = rotate(ln_pos, 90.0, origin=(0.0, 0.0))
        if not rot.is_empty:
            scale_lines_polys.append(rot)

    merged_tessellate = merged_circle.intersection(clip_box)
    if hasattr(merged_tessellate, "make_valid"):
        merged_tessellate = merged_tessellate.make_valid()

    kong_buffer = 镂空.buffer(REFINE_BUFFER)
    blocks = []
    x, y = xmin, ymin
    while x < xmax:
        while y < ymax:
            x1 = min(x + GRID_CELL_SIZE_COARSE, xmax)
            y1 = min(y + GRID_CELL_SIZE_COARSE, ymax)
            blocks.append((merged_tessellate, kong_buffer, x, y, x1, y1))
            y = y1
        y = ymin
        x += GRID_CELL_SIZE_COARSE

    n_workers = min(multiprocessing.cpu_count(), 8)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_build_cells_for_block, blocks))
    adaptive_cells = []
    for r in results:
        adaptive_cells.extend(r)

    chunk_size = max(1, len(adaptive_cells) // (n_workers * 4))
    chunks = [adaptive_cells[i:i + chunk_size] for i in range(0, len(adaptive_cells), chunk_size)]
    batch_args = [(chunk, merged_tessellate) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        pieces_nested = list(ex.map(_intersect_batch, batch_args))

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    def write_geom(geom, layer, color=7):
        if geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            ext = list(geom.exterior.coords)[:-1]
            if len(ext) >= 3:
                msp.add_lwpolyline([(float(x), float(y)) for x, y in ext], close=True, dxfattribs={"layer": layer})
            for interior in geom.interiors:
                pts = list(interior.coords)[:-1]
                if len(pts) >= 3:
                    msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": layer})
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                write_geom(g, layer, color)

    doc.layers.add("Ring", color=2)
    for coords in _geom_to_polygon_coords(ring):
        if len(coords) >= 3:
            pts = [(float(p[0]), float(p[1])) for p in coords]
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Ring"})
    doc.layers.add("Center_square_circle", color=6)
    for coords in _geom_to_polygon_coords(center_shape):
        if len(coords) >= 3:
            pts = [(float(p[0]), float(p[1])) for p in coords]
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Center_square_circle"})
    doc.layers.add("Square_980_40", color=7)
    for t in trapezoids_980_40:
        for coords in _geom_to_polygon_coords(t):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Square_980_40"})
    for name, g, c in [("Circle_line_g1", shape_g1, 7), ("Circle_line_g2", shape_g2, 8),
                       ("Circle_line_g3", shape_g3, 9), ("Circle_line_g4", shape_g4, 10)]:
        doc.layers.add(name, color=c)
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": name})
    doc.layers.add("Au_1", color=16)
    # 梯形内圈 - 位于中心附近的梯形结构
    trapezoid_inner_coords = [(960.0, 40.0), (1020.0, 40.0), (1020.0, 110.0), (1008.0, 110.0)]
    trapezoid_inner_polys = [rotate(Polygon(trapezoid_inner_coords).buffer(5.0), k * 90.0, origin=(0.0, 0.0)) for k in range(4)]
    for tp in trapezoid_inner_polys:
        for coords in _geom_to_polygon_coords(tp):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 外圈圆点 - 位于四个象限的外侧圆点阵列
    for g in au1_circles:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 刻度线 - 放射状刻度标记线
    for g in scale_lines_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 连接两圆的线 - 连接外圈圆点的线段
    for g in au1_conn_two_circles_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 连接刻度的线 - 从圆点连接到刻度的线段
    for g in au1_conn_to_scale_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 连接刻度的圆 - 刻度端点的小圆点
    for g in au1_conn_to_scale_circle_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 螺旋结构 - 螺旋形网格结构
    for g in au1_spiral_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 弧形结构 - 弧形网格层
    for g in au1_arc_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})

    doc.layers.add("Au_2", color=17)
    # 刻度线 - 放射状刻度标记线
    for g in scale_lines_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 连接刻度的线 - 从圆点连接到刻度的线段
    for g in au1_conn_to_scale_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 连接刻度的圆 - 刻度端点的小圆点（一端直径2微米，另一端直径30微米）
    for g in au2_conn_to_scale_circle_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 螺旋结构 - 螺旋形网格结构
    for g in au2_spiral_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 弧形结构 - 弧形网格层
    for g in au2_arc_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})

    layer_names = ["Bridge_spiral_0", "Bridge_spiral_90", "Bridge_spiral_180", "Bridge_spiral_270"]
    for k, (name, g) in enumerate(zip(layer_names, bridge_spirals)):
        doc.layers.add(name, color=k + 2)
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": name})
    doc.layers.add("Spiral_connector", color=11)
    for g in conn_lines:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Spiral_connector"})

    doc.layers.add("Bridge", color=5)
    for g in bridge_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Bridge"})

    doc.layers.add("Rect_bridge_connector", color=14)
    for g in rect_bridge_connector_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Rect_bridge_connector"})

    doc.layers.add("Rect_horizontal_bridge", color=15)
    for g in rect_horizontal_bridge_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Rect_horizontal_bridge"})

    bridge_polys_clip = []
    for p in bridge_polys:
        b = p.intersection(clip_box)
        if not b.is_empty:
            bridge_polys_clip.append(b)
    rect_bridge_connector_clip = []
    for q in rect_bridge_connector_polys:
        qc = q.intersection(clip_box)
        if not qc.is_empty:
            rect_bridge_connector_clip.append(qc)
    rect_horizontal_bridge_clip = []
    for h in rect_horizontal_bridge_polys:
        hc = h.intersection(clip_box)
        if not hc.is_empty:
            rect_horizontal_bridge_clip.append(hc)

    doc.layers.add("merged", color=12)
    for pieces in pieces_nested:
        for piece in pieces:
            write_geom(piece, "merged", 12)
    for r in rect_polys:
        write_geom(r, "merged", 12)
    for b in bridge_polys_clip:
        write_geom(b, "merged", 12)
    for q in rect_bridge_connector_clip:
        write_geom(q, "merged", 12)
    for h in rect_horizontal_bridge_clip:
        write_geom(h, "merged", 12)
    doc.saveas(output_path)

    merged_doc = ezdxf.new("R2010")
    merged_msp = merged_doc.modelspace()
    merged_doc.layers.add("merged", color=12)

    def write_merged_geom(geom):
        if geom.is_empty:
            return
        if geom.geom_type == "Polygon":
            ext = list(geom.exterior.coords)[:-1]
            if len(ext) >= 3:
                merged_msp.add_lwpolyline([(float(x), float(y)) for x, y in ext], close=True, dxfattribs={"layer": "merged"})
            for interior in geom.interiors:
                pts = list(interior.coords)[:-1]
                if len(pts) >= 3:
                    merged_msp.add_lwpolyline([(float(x), float(y)) for x, y in pts], close=True, dxfattribs={"layer": "merged"})
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                write_merged_geom(g)

    for pieces in pieces_nested:
        for piece in pieces:
            write_merged_geom(piece)
    for r in rect_polys:
        write_merged_geom(r)
    for b in bridge_polys_clip:
        write_merged_geom(b)
    for q in rect_bridge_connector_clip:
        write_merged_geom(q)
    for h in rect_horizontal_bridge_clip:
        write_merged_geom(h)

    merged_doc.layers.add("Au_1", color=16)
    # 梯形内圈 - 位于中心附近的梯形结构
    for tp in trapezoid_inner_polys:
        for coords in _geom_to_polygon_coords(tp):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 外圈圆点 - 位于四个象限的外侧圆点阵列
    for g in au1_circles:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 刻度线 - 放射状刻度标记线
    for g in scale_lines_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 连接两圆的线 - 连接外圈圆点的线段
    for g in au1_conn_two_circles_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 连接刻度的线 - 从圆点连接到刻度的线段
    for g in au1_conn_to_scale_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 连接刻度的圆 - 刻度端点的小圆点
    for g in au1_conn_to_scale_circle_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 螺旋结构 - 螺旋形网格结构
    for g in au1_spiral_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 弧形结构 - 弧形网格层
    for g in au1_arc_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})
    # 矩形网格 - Au_1 (第1/3/5/7个矩形的第3/5/7...个网格)
    for g in au1_rect_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_1"})

    merged_doc.layers.add("Au_2", color=17)
    # 刻度线 - 放射状刻度标记线
    for g in scale_lines_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 连接刻度的线 - 从圆点连接到刻度的线段
    for g in au1_conn_to_scale_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 连接刻度的圆 - 刻度端点的小圆点（一端直径2微米，另一端直径30微米）
    for g in au2_conn_to_scale_circle_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 螺旋结构 - 螺旋形网格结构
    for g in au2_spiral_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 弧形结构 - 弧形网格层
    for g in au2_arc_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})
    # 矩形网格 - Au_2 (第2/4/6/8个矩形的第3/5/7...个网格)
    for g in au2_rect_polys:
        for coords in _geom_to_polygon_coords(g):
            if len(coords) >= 3:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                merged_msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "Au_2"})


    io_az_path = os.path.join(RESULTS_DIR, "io_AZ_merged.dxf")
    io_offset = (6495.0, 6000.0)

    def add_io_from_dxf(target_doc, target_msp, layer_name="Top-Pi", layer_color=13):
        if not os.path.exists(io_az_path):
            return
        try:
            io_doc = ezdxf.readfile(io_az_path)
            io_msp = io_doc.modelspace()
            ents = list(io_msp.query('*[layer=="merged"]'))
            if not ents:
                return
            if layer_name not in [lyr.dxf.name for lyr in target_doc.layers]:
                target_doc.layers.add(layer_name, color=layer_color)
            for e in ents:
                if e.dxftype() == "LWPOLYLINE":
                    pts = list(e.get_points("xy"))
                    if len(pts) >= 2:
                        shifted = [(-float(p[0]) + io_offset[0], float(p[1]) + io_offset[1]) for p in pts]
                        target_msp.add_lwpolyline(shifted, close=e.closed, dxfattribs={"layer": layer_name})
        except Exception:
            pass

    add_io_from_dxf(doc, msp, layer_name="Top-Pi", layer_color=13)
    doc.saveas(output_path)

    add_io_from_dxf(merged_doc, merged_msp, layer_name="merged", layer_color=12)
    merged_doc.saveas(merged_output_path)

    merged_svg_path = merged_output_path.replace(".dxf", ".svg")
    merged_obj_path = merged_output_path.replace(".dxf", ".obj")

    scale_factor = 20.0
    thickness = 50.0 * scale_factor

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for layer_name in ["merged", "Au_1", "Au_2"]:
        layer_color = {"merged": "blue", "Au_1": "red", "Au_2": "green"}.get(layer_name, "black")
        try:
            ents = list(merged_msp.query(f'*[layer=="{layer_name}"]'))
            for e in ents:
                if e.dxftype() == "LWPOLYLINE":
                    pts = list(e.get_points("xy"))
                    if len(pts) >= 2:
                        xs = [p[0] * scale_factor for p in pts]
                        ys = [p[1] * scale_factor for p in pts]
                        ax.fill(xs, ys, alpha=0.3, edgecolor=layer_color, facecolor=layer_color, linewidth=1)
        except Exception:
            pass

    ax.autoscale()
    ax.invert_yaxis()
    plt.savefig(merged_svg_path, dpi=300)
    plt.close()
    print(f"SVG saved to {merged_svg_path}")

    vertices = []
    faces = []
    vertex_idx = 0

    for layer_name in ["merged", "Au_1", "Au_2"]:
        try:
            ents = list(merged_msp.query(f'*[layer=="{layer_name}"]'))
            for e in ents:
                if e.dxftype() == "LWPOLYLINE":
                    pts = list(e.get_points("xy"))
                    if len(pts) >= 3:
                        n = len(pts)
                        top_face_indices = []
                        bottom_face_indices = []

                        for p in pts:
                            x = float(p[0]) * scale_factor
                            y = float(p[1]) * scale_factor
                            vertices.append((x, y, 0.0))
                            top_face_indices.append(vertex_idx)
                            vertex_idx += 1

                            vertices.append((x, y, thickness))
                            bottom_face_indices.append(vertex_idx)
                            vertex_idx += 1

                        faces.append(tuple(top_face_indices))
                        faces.append(tuple(reversed(bottom_face_indices)))

                        for i in range(n):
                            i_next = (i + 1) % n
                            side_face = [
                                top_face_indices[i],
                                top_face_indices[i_next],
                                bottom_face_indices[i_next],
                                bottom_face_indices[i]
                            ]
                            faces.append(tuple(side_face))
        except Exception:
            pass

    with open(merged_obj_path, 'w') as f:
        f.write("# OBJ file\n")
        f.write(f"# scale: {scale_factor}x, thickness: {thickness}\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {' '.join(str(i+1) for i in face)}\n")

    print(f"OBJ saved to {merged_obj_path}")


def main():
    scale = 0.1  # 除中心圆环外其余结构缩放比例
    r_o = 26150.0 * scale  # kirigami_plattle 外半径 (μm)，外侧圆盘
    r_i = 15000.0 * scale  # kirigami_plattle 中心孔半径 (μm)
    dr1 = 900.0 * scale  # 第一个切缝距中心孔的距离 (μm)
    dr2 = 900.0 * scale  # 第二个切缝与第一个切缝的间距 (μm)
    n = 1.0  # 径向分布指数 (≥1)，n=1 等距，n>1 外圈间距逐渐增大
    N_theta = 8  # 角向扇区数量
    theta_ratio = 0.3  # 角度比 θ = θ_a / θ_i，控制切缝宽度与间隔
    dr_min = 300.0 * scale  # 最小材料宽度 (μm)，制造约束，径向间距小于此值则停止
    theta_i = 2 * np.pi / (N_theta * (1 + theta_ratio))  # 单个切缝的角度宽度 (弧度)
    theta_a = theta_ratio * theta_i  # 相邻切缝间的间隔角度 (弧度)
    offset = (theta_i + theta_a) / 2  # 相邻圈切缝的角度偏移 (弧度)，1/2 周期使切缝交错
    slit_width = 50.0 * scale  # 切缝径向宽度 (μm)，即 pg.arc 的 width，模拟激光切割的切缝宽度
    fillet_radius = 25.0 * scale  # 切缝倒角半径 (μm)，0 表示不倒角
    slit_base_angle = 22.5  # 第一层切缝中心线的基准角度 (度)，0°=正X轴逆时针。N_theta=4 时 45 使中心分布于 45/135/225/315 度
    connector_width = 80.0
    center_hole_diameter = 60.0
    tab_length = 10385.0  # io_pad 长 (μm)，沿切线
    tab_width_above = 3940.0  # 圆盘上方露出的宽度 (μm)
    tab_width_below = 2000.0  # 向下延伸与圆盘重叠的宽度 (μm)，重叠部分将减去
    tab_width = tab_width_above + tab_width_below  # 总宽度 5940 (μm)

    D = ph.Device("kirigami")
    kirigami_plattle_outer = pg.circle(radius=r_o, layer=(0, 0))
    kirigami_plattle_inner = pg.circle(radius=r_i, layer=(0, 0))
    kirigami_plattle = pg.boolean(kirigami_plattle_outer, kirigami_plattle_inner, operation="A-B", layer=(0, 0))
    D.add_ref(kirigami_plattle)
    add_cross_bridge(D, r_i, connector_width, center_hole_diameter, bridge_length=1000.0, inner_length=200.0, width_inner=connector_width, layer=(2, 0))
    add_spirals_to_180(D, r_i, center_hole_diameter / 2, 1000.0, spiral_width=80.0, layer=(2, 0))

    tab_rect_dev = ph.Device()
    tab_rect_ref = tab_rect_dev.add_ref(pg.rectangle(size=(tab_length, tab_width), layer=(3, 0)))
    tab_rect_ref.move((-tab_length / 2, r_o - tab_width_below))
    tab_final = pg.boolean(tab_rect_dev, kirigami_plattle_outer, operation="A-B", layer=(3, 0))
    D.add_ref(tab_final)

    positions = generate_radial_positions(r_i, r_o, dr1, dr2, n, dr_min, slit_width)
    positions = positions[:-1]
    add_kirigami_to_device(D, r_i, r_o, positions, N_theta, theta_ratio, offset, slit_width, fillet_radius, slit_base_angle, layer=(1, 0))

    p1 = os.path.join(RESULTS_DIR, "kirigami_pattern.dxf")
    p2 = os.path.join(RESULTS_DIR, "kirigami_pattern_woio.dxf")
    p3 = os.path.join(RESULTS_DIR, "kirigami_pattern_triangle.dxf")
    p4 = os.path.join(RESULTS_DIR, "kirigami_pattern_triangle_merged.dxf")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    phidl_device_to_dxf(D, p1, include_io_pad=True)
    write_woio_dxf_5layers(
        D, p2,
        r_i=r_i, r_o=r_o,
        center_hole_diameter=center_hole_diameter,
        bridge_length=1000.0,
        inner_length=200.0,
        width_inner=connector_width,
        width_outer=connector_width,
        spiral_width=80.0,
    )
    write_woio_dxf_triangle_merged(
        D, p3, p4,
        r_i=r_i, r_o=r_o,
        center_hole_diameter=center_hole_diameter,
        bridge_length=1000.0,
        inner_length=200.0,
        width_inner=connector_width,
        width_outer=connector_width,
        spiral_width=80.0,
    )
    print(f"径向切缝位置 (μm): {[round(p, 2) for p in positions]}")
    print(f"已保存至 {p1}、{p2}、{p3}、{p4}")


if __name__ == "__main__":
    main()
