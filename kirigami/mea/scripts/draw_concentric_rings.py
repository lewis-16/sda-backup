import os
import numpy as np
import ezdxf
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

NUM_RINGS = 12
RADIUS_INNERMOST_UM = 100.0
RADIUS_OUTERMOST_UM = 1000.0
RING_WIDTH_UM = 8.0
OUTERMOST_RING_WIDTH_UM = 100.0
SEGMENTS_PER_RING = 360
SECTOR_DIVISIONS = 64
SECTOR_STRIP_EDGE_SHRINK_TURN_FRACTION = 0
CIRCLE_DIAMETER_UM = 20.0
PAD_ANNULUS_CENTER_RADIUS_UM = 11.0
PAD_ANNULUS_WIDTH_UM = 10.0
CIRCLE_OFFSET_UM = 15.0
PAIR_CENTER_DISTANCE_UM = 30.0
RING12_TRIPLE_CENTER_SPACING_UM = 45.0
RING6_11_PAIR_CENTER_SPACING_STEP_UM = 5.0
RING2_LINE_OFFSET_UM = 4.0
RING2_ARC_SEGMENTS = 48
RING2_START_DOT_DIAMETER_UM = 2.0
RING12_EXTRA_LINE_OFFSET_UM = 4.0
RING12_EXTRA_LINE_SIDE = "pos"
RING1_INNER_BRIDGE_RECT_LENGTH_UM = 10.0
RING1_INNER_BRIDGE_RECT_WIDTH_UM = 15.0


def build_ring_radii():
    return np.linspace(RADIUS_INNERMOST_UM, RADIUS_OUTERMOST_UM, NUM_RINGS)


def build_ring_cells_from_inner_outer(inner_radius, outer_radius, segments_per_ring):
    theta = np.linspace(0.0, 2.0 * np.pi, segments_per_ring, endpoint=False)
    inner_pts = np.column_stack((inner_radius * np.cos(theta), inner_radius * np.sin(theta)))
    outer_pts = np.column_stack((outer_radius * np.cos(theta), outer_radius * np.sin(theta)))
    cells = []
    for i in range(segments_per_ring):
        j = (i + 1) % segments_per_ring
        cells.append([
            (float(inner_pts[i][0]), float(inner_pts[i][1])),
            (float(inner_pts[j][0]), float(inner_pts[j][1])),
            (float(outer_pts[j][0]), float(outer_pts[j][1])),
            (float(outer_pts[i][0]), float(outer_pts[i][1])),
        ])
    return cells


def should_draw_segment(segment_index, segments_per_ring, sector_divisions):
    theta_fraction = segment_index / segments_per_ring
    sector_index = int(np.floor(theta_fraction * sector_divisions)) % sector_divisions
    return sector_index % 2 == 0


def build_annulus_sector_polygon(sector_index, sector_divisions, inner_radius, outer_radius, arc_points=32):
    edge = 2.0 * np.pi * SECTOR_STRIP_EDGE_SHRINK_TURN_FRACTION
    theta0 = 2.0 * np.pi * sector_index / sector_divisions + edge
    theta1 = 2.0 * np.pi * (sector_index + 1) / sector_divisions - edge
    theta_vals = np.linspace(theta0, theta1, arc_points)
    outer_arc = [(float(outer_radius * np.cos(t)), float(outer_radius * np.sin(t))) for t in theta_vals]
    inner_arc = [(float(inner_radius * np.cos(t)), float(inner_radius * np.sin(t))) for t in theta_vals[::-1]]
    return outer_arc + inner_arc


def line_circle_intersections_along_direction(base_x, base_y, ux, uy, radius):
    b_dot_u = base_x * ux + base_y * uy
    b_norm2 = base_x * base_x + base_y * base_y
    c = b_norm2 - radius * radius
    disc = b_dot_u * b_dot_u - c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = -b_dot_u - sqrt_disc
    t2 = -b_dot_u + sqrt_disc
    return (t1, t2)


def normalize_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def add_pad_annulus_at(msp, cx, cy, layer, segments_per_ring):
    ir = float(PAD_ANNULUS_CENTER_RADIUS_UM - PAD_ANNULUS_WIDTH_UM / 2.0)
    out_r = float(PAD_ANNULUS_CENTER_RADIUS_UM + PAD_ANNULUS_WIDTH_UM / 2.0)
    for rect_pts in build_ring_cells_from_inner_outer(
        inner_radius=ir,
        outer_radius=out_r,
        segments_per_ring=segments_per_ring,
    ):
        shifted = [(float(x + cx), float(y + cy)) for x, y in rect_pts]
        msp.add_lwpolyline(shifted, close=True, dxfattribs={"layer": layer})


def collect_d20_pad_centers(min_radius, ring_radii):
    out = []
    circle_count = 0
    for sector_index in range(SECTOR_DIVISIONS):
        if sector_index % 2 != 0:
            continue
        circle_count += 1
        radial_shift = -CIRCLE_OFFSET_UM if circle_count % 2 == 1 else CIRCLE_OFFSET_UM
        circle_radius_pos = min_radius + radial_shift
        theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
        cx = circle_radius_pos * np.cos(theta_center)
        cy = circle_radius_pos * np.sin(theta_center)
        out.append((float(cx), float(cy)))
    for ring_index in range(1, 5):
        ring_radius = float(ring_radii[ring_index])
        for sector_index in range(SECTOR_DIVISIONS):
            if sector_index % 2 == 0:
                continue
            theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
            cx = ring_radius * np.cos(theta_center)
            cy = ring_radius * np.sin(theta_center)
            out.append((float(cx), float(cy)))
    for ring_index in range(5, 11):
        ring_radius = float(ring_radii[ring_index])
        ring_id = ring_index + 1
        pair_center_distance_um = PAIR_CENTER_DISTANCE_UM + RING6_11_PAIR_CENTER_SPACING_STEP_UM * (
            ring_id - 6
        )
        half_pair_distance = pair_center_distance_um / 2.0
        for sector_index in range(SECTOR_DIVISIONS):
            if sector_index % 2 == 0:
                continue
            theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
            tx = -np.sin(theta_center)
            ty = np.cos(theta_center)
            base_x = ring_radius * np.cos(theta_center)
            base_y = ring_radius * np.sin(theta_center)
            c1x = base_x - half_pair_distance * tx
            c1y = base_y - half_pair_distance * ty
            c2x = base_x + half_pair_distance * tx
            c2y = base_y + half_pair_distance * ty
            out.append((float(c1x), float(c1y)))
            out.append((float(c2x), float(c2y)))
    ring12_radius = float(ring_radii[11])
    for sector_index in range(SECTOR_DIVISIONS):
        if sector_index % 2 == 0:
            continue
        theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
        tx = -np.sin(theta_center)
        ty = np.cos(theta_center)
        base_x = ring12_radius * np.cos(theta_center)
        base_y = ring12_radius * np.sin(theta_center)
        for offset in (-RING12_TRIPLE_CENTER_SPACING_UM, 0.0, RING12_TRIPLE_CENTER_SPACING_UM):
            cx = base_x + offset * tx
            cy = base_y + offset * ty
            out.append((float(cx), float(cy)))
    return out


def emit_polygon_regions_to_msp(msp, geom, layer):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        for sub in geom.geoms:
            emit_polygon_regions_to_msp(msp, sub, layer)
        return
    else:
        return
    for p in polys:
        if p.is_empty:
            continue
        coords = list(p.exterior.coords)
        if len(coords) >= 4:
            pts = [(float(x), float(y)) for x, y in coords[:-1]]
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
        for intr in p.interiors:
            ic = list(intr.coords)
            if len(ic) < 4:
                continue
            ipt = [(float(x), float(y)) for x, y in ic[:-1]]
            msp.add_lwpolyline(ipt, close=True, dxfattribs={"layer": layer})


def add_annulus_cells_minus_pad_holes(
    msp, inner_radius, outer_radius, layer, hole_tree, hole_disks, segments_per_ring
):
    for rect_pts in build_ring_cells_from_inner_outer(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        segments_per_ring=segments_per_ring,
    ):
        poly = Polygon(rect_pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        idxs = hole_tree.query(poly, predicate="intersects")
        if len(idxs) == 0:
            msp.add_lwpolyline(rect_pts, close=True, dxfattribs={"layer": layer})
            continue
        local_holes = unary_union([hole_disks[int(i)] for i in idxs])
        diff = poly.difference(local_holes)
        emit_polygon_regions_to_msp(msp, diff, layer)


def add_sector_strip_minus_pad_holes(msp, sector_pts, layer, hole_tree, hole_disks):
    poly = Polygon(sector_pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    idxs = hole_tree.query(poly, predicate="intersects")
    if len(idxs) == 0:
        msp.add_lwpolyline(sector_pts, close=True, dxfattribs={"layer": layer})
        return
    local_holes = unary_union([hole_disks[int(i)] for i in idxs])
    diff = poly.difference(local_holes)
    emit_polygon_regions_to_msp(msp, diff, layer)


def draw_concentric_rings_dxf(output_path, merged_two_layer=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    if merged_two_layer:
        doc.layers.add("Au", color=2)
        doc.layers.add("Az", color=1)
        layer_rings = "Az"
        layer_overlay = "Az"
        layer_inner_circles = "Au"
        layer_ring2_5 = "Au"
        layer_ring6_12_pairs = "Au"
        layer_au = "Au"
    else:
        doc.layers.add("rings", color=1)
        doc.layers.add("overlay_1_64", color=3)
        doc.layers.add("inner_circles", color=5)
        doc.layers.add("ring2_5_empty_sector_circles", color=6)
        doc.layers.add("ring6_12_empty_sector_circle_pairs", color=4)
        doc.layers.add("Au", color=2)
        layer_rings = "rings"
        layer_overlay = "overlay_1_64"
        layer_inner_circles = "inner_circles"
        layer_ring2_5 = "ring2_5_empty_sector_circles"
        layer_ring6_12_pairs = "ring6_12_empty_sector_circle_pairs"
        layer_au = "Au"
    ring_radii = build_ring_radii()
    min_radius = float(ring_radii[0] - RING_WIDTH_UM / 2.0)
    max_radius = float(ring_radii[-1] + OUTERMOST_RING_WIDTH_UM / 2.0)
    pad_centers = collect_d20_pad_centers(min_radius, ring_radii)
    hole_r_az_cut = float(CIRCLE_DIAMETER_UM / 2.0)
    hole_disks = [
        Point(float(cx), float(cy)).buffer(hole_r_az_cut, quad_segs=32) for cx, cy in pad_centers
    ]
    hole_tree = STRtree(hole_disks)
    for i, radius in enumerate(ring_radii):
        ring_width = OUTERMOST_RING_WIDTH_UM if i == len(ring_radii) - 1 else RING_WIDTH_UM
        if radius - ring_width / 2.0 <= 0:
            raise ValueError("圆环内半径必须大于 0")
        inner_radius = float(radius - ring_width / 2.0)
        outer_radius = float(radius + ring_width / 2.0)
        add_annulus_cells_minus_pad_holes(
            msp,
            inner_radius,
            outer_radius,
            layer_rings,
            hole_tree,
            hole_disks,
            SEGMENTS_PER_RING,
        )
    innermost_extra_ring_center_r = float(min_radius - CIRCLE_OFFSET_UM)
    innermost_extra_inner = float(innermost_extra_ring_center_r - RING_WIDTH_UM / 2.0)
    innermost_extra_outer = float(innermost_extra_ring_center_r + RING_WIDTH_UM / 2.0)
    if innermost_extra_inner <= 0:
        raise ValueError("圆环内半径必须大于 0")
    add_annulus_cells_minus_pad_holes(
        msp,
        innermost_extra_inner,
        innermost_extra_outer,
        layer_rings,
        hole_tree,
        hole_disks,
        SEGMENTS_PER_RING,
    )
    circle_count = 0
    line_start_by_ring = {}
    line_segment_by_ring = {}
    for sector_index in range(SECTOR_DIVISIONS):
        if sector_index % 2 != 0:
            continue
        circle_count += 1
        radial_shift = -CIRCLE_OFFSET_UM if circle_count % 2 == 1 else CIRCLE_OFFSET_UM
        circle_radius_pos = min_radius + radial_shift
        sector_inner_radius = circle_radius_pos if circle_count % 2 == 1 else min_radius
        sector_pts = build_annulus_sector_polygon(
            sector_index=sector_index,
            sector_divisions=SECTOR_DIVISIONS,
            inner_radius=sector_inner_radius,
            outer_radius=max_radius,
            arc_points=32,
        )
        add_sector_strip_minus_pad_holes(msp, sector_pts, layer_overlay, hole_tree, hole_disks)
        theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
        cx = circle_radius_pos * np.cos(theta_center)
        cy = circle_radius_pos * np.sin(theta_center)
        msp.add_circle(
            (float(cx), float(cy)),
            float(CIRCLE_DIAMETER_UM / 2.0),
            dxfattribs={"layer": layer_inner_circles},
        )
        add_pad_annulus_at(msp, float(cx), float(cy), layer_rings, SEGMENTS_PER_RING)
        if circle_count % 2 == 1:
            h_len = RING1_INNER_BRIDGE_RECT_LENGTH_UM / 2.0
            h_wid = RING1_INNER_BRIDGE_RECT_WIDTH_UM / 2.0
            r_mid = float(min_radius)
            ux = float(np.cos(theta_center))
            uy = float(np.sin(theta_center))
            tx = float(-np.sin(theta_center))
            ty = float(np.cos(theta_center))
            mx = r_mid * ux
            my = r_mid * uy
            rect_pts = [
                (mx + h_len * ux + h_wid * tx, my + h_len * uy + h_wid * ty),
                (mx + h_len * ux - h_wid * tx, my + h_len * uy - h_wid * ty),
                (mx - h_len * ux - h_wid * tx, my - h_len * uy - h_wid * ty),
                (mx - h_len * ux + h_wid * tx, my - h_len * uy + h_wid * ty),
            ]
            msp.add_lwpolyline(rect_pts, close=True, dxfattribs={"layer": layer_rings})
        ex = max_radius * np.cos(theta_center)
        ey = max_radius * np.sin(theta_center)
        msp.add_lwpolyline(
            [(float(cx), float(cy)), (float(ex), float(ey))],
            dxfattribs={"layer": layer_au, "const_width": 2.0},
        )
        ux = np.cos(theta_center)
        uy = np.sin(theta_center)
        nx = -np.sin(theta_center)
        ny = np.cos(theta_center)
        line_defs = [
            (-RING2_LINE_OFFSET_UM, float(ring_radii[1])),
            (+RING2_LINE_OFFSET_UM, float(ring_radii[2])),
            (-2.0 * RING2_LINE_OFFSET_UM, float(ring_radii[3])),
            (+2.0 * RING2_LINE_OFFSET_UM, float(ring_radii[4])),
        ]
        for ring_idx in range(5, len(ring_radii)):
            offset_mag = float((ring_idx - 2) * RING2_LINE_OFFSET_UM)
            start_radius = float(ring_radii[ring_idx])
            line_defs.append((-offset_mag, start_radius))
            line_defs.append((+offset_mag, start_radius))
        for offset_d, start_radius in line_defs:
            bx = cx + offset_d * nx
            by = cy + offset_d * ny
            t_start_pair = line_circle_intersections_along_direction(bx, by, ux, uy, start_radius)
            t_end_pair = line_circle_intersections_along_direction(bx, by, ux, uy, max_radius)
            if t_start_pair is None or t_end_pair is None:
                continue
            t_start = max(t_start_pair)
            t_end = max(t_end_pair)
            sx = bx + t_start * ux
            sy = by + t_start * uy
            ex2 = bx + t_end * ux
            ey2 = by + t_end * uy
            msp.add_lwpolyline(
                [(float(sx), float(sy)), (float(ex2), float(ey2))],
                dxfattribs={"layer": layer_au, "const_width": 2.0},
            )
            ring_id = None
            side_key = "neg" if offset_d < 0 else "pos"
            if np.isclose(offset_d, -RING2_LINE_OFFSET_UM) and np.isclose(start_radius, float(ring_radii[1])):
                ring_id = 2
            elif np.isclose(offset_d, +RING2_LINE_OFFSET_UM) and np.isclose(start_radius, float(ring_radii[2])):
                ring_id = 3
            elif np.isclose(offset_d, -2.0 * RING2_LINE_OFFSET_UM) and np.isclose(start_radius, float(ring_radii[3])):
                ring_id = 4
            elif np.isclose(offset_d, +2.0 * RING2_LINE_OFFSET_UM) and np.isclose(start_radius, float(ring_radii[4])):
                ring_id = 5
            if ring_id is None:
                ring_idx = int(np.argmin(np.abs(ring_radii - start_radius)))
                if np.isclose(start_radius, float(ring_radii[ring_idx])):
                    ring_id = ring_idx + 1
            if ring_id is not None:
                if ring_id not in line_start_by_ring:
                    line_start_by_ring[ring_id] = {"neg": {}, "pos": {}}
                if ring_id not in line_segment_by_ring:
                    line_segment_by_ring[ring_id] = {"neg": {}, "pos": {}}
                line_start_by_ring[ring_id][side_key][sector_index] = (float(sx), float(sy))
                line_segment_by_ring[ring_id][side_key][sector_index] = (float(sx), float(sy), float(ex2), float(ey2))
                msp.add_circle(
                    (float(sx), float(sy)),
                    float(RING2_START_DOT_DIAMETER_UM / 2.0),
                    dxfattribs={"layer": layer_au},
                )
    for ring_index in range(1, 5):
        ring_radius = float(ring_radii[ring_index])
        for sector_index in range(SECTOR_DIVISIONS):
            if sector_index % 2 == 0:
                continue
            theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
            cx = ring_radius * np.cos(theta_center)
            cy = ring_radius * np.sin(theta_center)
            msp.add_circle(
                (float(cx), float(cy)),
                float(CIRCLE_DIAMETER_UM / 2.0),
                dxfattribs={"layer": layer_ring2_5},
            )
            add_pad_annulus_at(msp, float(cx), float(cy), layer_rings, SEGMENTS_PER_RING)
            ring_id = ring_index + 1
            if ring_id in (2, 3, 4, 5):
                is_ccw = ring_id in (2, 4)
                target_sector = (sector_index + 1) % SECTOR_DIVISIONS if is_ccw else (sector_index - 1) % SECTOR_DIVISIONS
                side_key = "neg" if ring_id in (2, 4) else "pos"
                if ring_id in line_start_by_ring and target_sector in line_start_by_ring[ring_id][side_key]:
                    sx, sy = line_start_by_ring[ring_id][side_key][target_sector]
                    theta_circle = np.arctan2(cy, cx)
                    theta_line = np.arctan2(sy, sx)
                    if is_ccw:
                        if theta_line < theta_circle:
                            theta_line += 2.0 * np.pi
                    else:
                        if theta_line > theta_circle:
                            theta_line -= 2.0 * np.pi
                    theta_vals = np.linspace(theta_circle, theta_line, RING2_ARC_SEGMENTS)
                    arc_pts = [(float(ring_radius * np.cos(t)), float(ring_radius * np.sin(t))) for t in theta_vals]
                    msp.add_lwpolyline(
                        arc_pts,
                        dxfattribs={"layer": layer_au, "const_width": 2.0},
                    )
    for ring_index in range(5, 11):
        ring_radius = float(ring_radii[ring_index])
        ring_id = ring_index + 1
        pair_center_distance_um = PAIR_CENTER_DISTANCE_UM + RING6_11_PAIR_CENTER_SPACING_STEP_UM * (
            ring_id - 6
        )
        half_pair_distance = pair_center_distance_um / 2.0
        for sector_index in range(SECTOR_DIVISIONS):
            if sector_index % 2 == 0:
                continue
            theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
            tx = -np.sin(theta_center)
            ty = np.cos(theta_center)
            base_x = ring_radius * np.cos(theta_center)
            base_y = ring_radius * np.sin(theta_center)
            c1x = base_x - half_pair_distance * tx
            c1y = base_y - half_pair_distance * ty
            c2x = base_x + half_pair_distance * tx
            c2y = base_y + half_pair_distance * ty
            msp.add_circle(
                (float(c1x), float(c1y)),
                float(CIRCLE_DIAMETER_UM / 2.0),
                dxfattribs={"layer": layer_ring6_12_pairs},
            )
            add_pad_annulus_at(msp, float(c1x), float(c1y), layer_rings, SEGMENTS_PER_RING)
            msp.add_circle(
                (float(c2x), float(c2y)),
                float(CIRCLE_DIAMETER_UM / 2.0),
                dxfattribs={"layer": layer_ring6_12_pairs},
            )
            add_pad_annulus_at(msp, float(c2x), float(c2y), layer_rings, SEGMENTS_PER_RING)
            if ring_id in line_start_by_ring:
                pair_defs = ((c1x, c1y, "pos"), (c2x, c2y, "neg"))
                for cx_i, cy_i, side_key in pair_defs:
                    line_map = line_start_by_ring[ring_id][side_key]
                    if not line_map:
                        continue
                    theta_c = np.arctan2(cy_i, cx_i)
                    best = None
                    best_abs = None
                    for sx, sy in line_map.values():
                        theta_s = np.arctan2(sy, sx)
                        d = normalize_angle(theta_s - theta_c)
                        if best_abs is None or abs(d) < best_abs:
                            best_abs = abs(d)
                            best = theta_s
                    if best is not None:
                        delta = normalize_angle(best - theta_c)
                        theta_vals = np.linspace(theta_c, theta_c + delta, RING2_ARC_SEGMENTS)
                        arc_pts = [(float(ring_radius * np.cos(t)), float(ring_radius * np.sin(t))) for t in theta_vals]
                        msp.add_lwpolyline(
                            arc_pts,
                            dxfattribs={"layer": layer_au, "const_width": 2.0},
                        )
    ring12_radius = float(ring_radii[11])
    ring12_mid_circle_by_sector = {}
    for sector_index in range(SECTOR_DIVISIONS):
        if sector_index % 2 == 0:
            continue
        theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
        tx = -np.sin(theta_center)
        ty = np.cos(theta_center)
        base_x = ring12_radius * np.cos(theta_center)
        base_y = ring12_radius * np.sin(theta_center)
        for offset in (-RING12_TRIPLE_CENTER_SPACING_UM, 0.0, RING12_TRIPLE_CENTER_SPACING_UM):
            cx = base_x + offset * tx
            cy = base_y + offset * ty
            msp.add_circle(
                (float(cx), float(cy)),
                float(CIRCLE_DIAMETER_UM / 2.0),
                dxfattribs={"layer": layer_ring6_12_pairs},
            )
            add_pad_annulus_at(msp, float(cx), float(cy), layer_rings, SEGMENTS_PER_RING)
            if offset == 0.0:
                ring12_mid_circle_by_sector[sector_index] = (float(cx), float(cy), float(theta_center))
                continue
            if 12 in line_start_by_ring:
                side_key = "pos" if offset < 0 else "neg"
                line_map = line_start_by_ring[12][side_key]
                if line_map:
                    theta_c = np.arctan2(cy, cx)
                    best = None
                    best_abs = None
                    for sx, sy in line_map.values():
                        theta_s = np.arctan2(sy, sx)
                        d = normalize_angle(theta_s - theta_c)
                        if best_abs is None or abs(d) < best_abs:
                            best_abs = abs(d)
                            best = theta_s
                    if best is not None:
                        delta = normalize_angle(best - theta_c)
                        theta_vals = np.linspace(theta_c, theta_c + delta, RING2_ARC_SEGMENTS)
                        arc_pts = [(float(ring12_radius * np.cos(t)), float(ring12_radius * np.sin(t))) for t in theta_vals]
                        msp.add_lwpolyline(
                            arc_pts,
                            dxfattribs={"layer": layer_au, "const_width": 2.0},
                        )
    ring12_extra_line_start_by_sector = {}
    if 12 in line_segment_by_ring and RING12_EXTRA_LINE_SIDE in line_segment_by_ring[12]:
        side_sign = 1.0 if RING12_EXTRA_LINE_SIDE == "pos" else -1.0
        for sector_index, seg in line_segment_by_ring[12][RING12_EXTRA_LINE_SIDE].items():
            sx, sy, ex2, ey2 = seg
            theta_center = 2.0 * np.pi * (sector_index + 0.5) / SECTOR_DIVISIONS
            nx = -np.sin(theta_center)
            ny = np.cos(theta_center)
            shift = side_sign * RING12_EXTRA_LINE_OFFSET_UM
            s_shift_x = sx + shift * nx
            s_shift_y = sy + shift * ny
            e_shift_x = ex2 + shift * nx
            e_shift_y = ey2 + shift * ny
            s_half_x = 0.5 * (s_shift_x + e_shift_x)
            s_half_y = 0.5 * (s_shift_y + e_shift_y)
            ring12_extra_line_start_by_sector[sector_index] = (float(s_half_x), float(s_half_y))
            msp.add_lwpolyline(
                [(float(s_half_x), float(s_half_y)), (float(e_shift_x), float(e_shift_y))],
                dxfattribs={"layer": layer_au, "const_width": 2.0},
            )
    if 12 in line_segment_by_ring and RING12_EXTRA_LINE_SIDE in line_segment_by_ring[12]:
        for sector_index, mid_info in ring12_mid_circle_by_sector.items():
            target_sector = None
            for cand in ((sector_index - 1) % SECTOR_DIVISIONS, (sector_index + 1) % SECTOR_DIVISIONS):
                if cand in line_segment_by_ring[12][RING12_EXTRA_LINE_SIDE] and cand in ring12_extra_line_start_by_sector:
                    target_sector = cand
                    break
            if target_sector is None:
                continue
            sx0, sy0, ex0, ey0 = line_segment_by_ring[12][RING12_EXTRA_LINE_SIDE][target_sector]
            mx, my, theta_center = mid_info
            full_len = np.hypot(ex0 - sx0, ey0 - sy0)
            half_len = 0.5 * full_len
            ux = np.cos(theta_center)
            uy = np.sin(theta_center)
            mid_end_x = mx + half_len * ux
            mid_end_y = my + half_len * uy
            exs, eys = ring12_extra_line_start_by_sector[target_sector]
            msp.add_lwpolyline(
                [(float(mx), float(my)), (float(mid_end_x), float(mid_end_y))],
                dxfattribs={"layer": layer_au, "const_width": 2.0},
            )
            msp.add_lwpolyline(
                [(float(mid_end_x), float(mid_end_y)), (float(exs), float(eys))],
                dxfattribs={"layer": layer_au, "const_width": 2.0},
            )
            msp.add_circle(
                (float(mid_end_x), float(mid_end_y)),
                float(RING2_START_DOT_DIAMETER_UM / 2.0),
                dxfattribs={"layer": layer_au},
            )
            msp.add_circle(
                (float(exs), float(eys)),
                float(RING2_START_DOT_DIAMETER_UM / 2.0),
                dxfattribs={"layer": layer_au},
            )
    doc.saveas(output_path)


if __name__ == "__main__":
    output_file = os.path.join(RESULTS_DIR, "concentric_12_rings.dxf")
    draw_concentric_rings_dxf(output_file)
    print(f"已生成: {output_file}")
    merged_file = os.path.join(RESULTS_DIR, "concentric_12_rings_merged.dxf")
    draw_concentric_rings_dxf(merged_file, merged_two_layer=True)
    print(f"已生成: {merged_file}")
