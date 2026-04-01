#!/usr/bin/env python
import argparse
import math
import os
import numpy as np
import ezdxf
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def _geom_to_path_d(geom, to_svg_xy):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        ext = np.array(geom.exterior.coords)[:-1]
        if len(ext) < 3:
            return None
        d = "M " + " L ".join(f"{to_svg_xy(x,y)[0]} {to_svg_xy(x,y)[1]}" for x, y in ext) + " Z"
        for interior in geom.interiors:
            pts = np.array(interior.coords)[:-1]
            if len(pts) >= 3:
                d += " M " + " L ".join(f"{to_svg_xy(x,y)[0]} {to_svg_xy(x,y)[1]}" for x, y in pts) + " Z"
        return d
    if geom.geom_type == "MultiPolygon":
        parts = [_geom_to_path_d(g, to_svg_xy) for g in geom.geoms]
        return " ".join(p for p in parts if p)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dxf", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--scale", type=float, default=50.0)
    p.add_argument("--r-inner", type=float, default=None)
    p.add_argument("--r-outer", type=float, default=None)
    args = p.parse_args()

    dxf_path = args.dxf or os.path.join(RESULTS_DIR, "kirigami_pattern_woio.dxf")
    out_path = args.out or os.path.join(RESULTS_DIR, "kirigami_woio_50x.svg")

    if not os.path.isfile(dxf_path):
        raise SystemExit(f"DXF not found: {dxf_path}")

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    scale = args.scale

    by_layer = {}
    for entity in msp:
        if entity.dxftype() == "LWPOLYLINE" and entity.closed:
            layer = entity.dxf.layer
            pts = list(entity.get_points())
            if len(pts) < 3:
                continue
            scaled = [(float(p[0]) * scale, float(p[1]) * scale) for p in pts]
            by_layer.setdefault(layer, []).append(scaled)

    order = [
        "Ring",
        "Center_square_circle",
        "Square_980_40",
        "Circle_line_g1",
        "Circle_line_g2",
        "Circle_line_g3",
        "Circle_line_g4",
        "Bridge_spiral_0",
        "Bridge_spiral_90",
        "Bridge_spiral_180",
        "Bridge_spiral_270",
    ]
    path_parts = []
    if all(lay in by_layer for lay in order):
        all_pts = []
        for layer in order:
            for pts in by_layer[layer]:
                all_pts.extend(pts)
        if all_pts:
            all_pts = np.array(all_pts)
            minx, miny = all_pts[:, 0].min(), all_pts[:, 1].min()
            maxx, maxy = all_pts[:, 0].max(), all_pts[:, 1].max()
            pad = max((maxx - minx), (maxy - miny)) * 0.02 or 1
            minx, miny = minx - pad, miny - pad
            maxx, maxy = maxx + pad, maxy + pad
            w, h = maxx - minx, maxy - miny

            def to_svg_xy(x, y):
                return (x - minx, maxy - y)

            def area(pts):
                try:
                    p = Polygon(pts).buffer(0)
                    return p.area if not p.is_empty else 0
                except Exception:
                    return 0

            def centroid(pts):
                n = len(pts)
                if n == 0:
                    return (0, 0)
                return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)

            def poly_contains_pt(poly, pt):
                try:
                    return not poly.is_empty and poly.contains(Point(pt))
                except Exception:
                    return False

            for layer in order:
                polys_pts = []
                for pts in by_layer[layer]:
                    try:
                        p = Polygon(pts).buffer(0)
                        if p and not p.is_empty:
                            polys_pts.append((pts, p, area(pts)))
                    except Exception:
                        pass
                if not polys_pts:
                    continue
                if layer in ("Ring", "Center_square_circle") and len(polys_pts) > 1:
                    polys_pts.sort(key=lambda x: -x[2])
                    exterior_pts = polys_pts[0][0]
                    exterior_poly = Polygon(exterior_pts).buffer(0)
                    holes = []
                    for pts, p, _ in polys_pts[1:]:
                        c = centroid(pts)
                        if poly_contains_pt(exterior_poly, c):
                            holes.append(pts)
                    try:
                        geom = Polygon(exterior_pts, holes=[list(h) for h in holes]).buffer(0)
                    except Exception:
                        geom = exterior_poly
                        for h in holes:
                            try:
                                geom = geom.difference(Polygon(h).buffer(0))
                            except Exception:
                                pass
                else:
                    geoms = [p for (_, p, _) in polys_pts]
                    geom = unary_union(geoms)
                d = _geom_to_path_d(geom, to_svg_xy)
                if d:
                    path_parts.append(d)
    if not path_parts:
        polylines = []
        for entity in msp:
            if entity.dxftype() == "LWPOLYLINE" and entity.closed:
                pts = list(entity.get_points())
                if len(pts) < 3:
                    continue
                scaled = [(float(p[0]) * scale, float(p[1]) * scale) for p in pts]
                polylines.append(scaled)
        if not polylines:
            raise SystemExit("No closed polylines to export")

        def centroid(pts):
            n = len(pts)
            if n == 0:
                return (0, 0)
            return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)

        def area(pts):
            try:
                p = Polygon(pts).buffer(0)
                return p.area if not p.is_empty else 0
            except Exception:
                return 0

        def contains(pts_outer, pts_inner):
            try:
                po = Polygon(pts_outer).buffer(0)
                if po.is_empty:
                    return False
                return po.contains(Point(centroid(pts_inner)))
            except Exception:
                return False

        areas = [area(pts) for pts in polylines]
        max_area = max(areas)
        area_threshold = max_area * 0.05
        large = [(i, polylines[i]) for i in range(len(polylines)) if areas[i] >= area_threshold]
        small_holes = [polylines[i] for i in range(len(polylines)) if areas[i] < area_threshold]

        exterior = None
        gap_holes = []
        for i, pts in large:
            others_centroids = [centroid(polylines[j]) for (j, _) in large if j != i]
            if not others_centroids:
                exterior = pts
                break
            if all(contains(pts, polylines[j]) for (j, _) in large if j != i):
                exterior = pts
                gap_holes = [polylines[j] for (j, _) in large if j != i]
                break
        if exterior is None:
            exterior = large[0][1]
            gap_holes = [p for (_, p) in large[1:]]

        all_holes = gap_holes + small_holes
        try:
            material = Polygon(exterior, holes=[list(h) for h in all_holes]).buffer(0)
            if material.is_empty:
                material = Polygon(exterior).buffer(0).difference(unary_union([Polygon(h).buffer(0) for h in all_holes]))
        except Exception:
            material = Polygon(exterior).buffer(0)
            for h in all_holes:
                try:
                    material = material.difference(Polygon(h).buffer(0))
                except Exception:
                    pass

        if material.is_empty:
            raise SystemExit("Material geometry is empty")

        all_pts = np.array([p for pts in polylines for p in pts])
        minx, miny = all_pts[:, 0].min(), all_pts[:, 1].min()
        maxx, maxy = all_pts[:, 0].max(), all_pts[:, 1].max()
        pad = max((maxx - minx), (maxy - miny)) * 0.02 or 1
        minx, miny = minx - pad, miny - pad
        maxx, maxy = maxx + pad, maxy + pad
        w, h = maxx - minx, maxy - miny

        def to_svg_xy(x, y):
            return (x - minx, maxy - y)

        path_parts.append(_geom_to_path_d(material, to_svg_xy))

    if not path_parts:
        raise SystemExit("No geometry to export")

    all_pts = []
    for layer in (order if all(lay in by_layer for lay in order) else []):
        for pts in by_layer.get(layer, []):
            all_pts.extend(pts)
    if not all_pts and by_layer:
        for layer, pt_list in by_layer.items():
            all_pts.extend([p for pts in pt_list for p in pts])
    all_pts = np.array(all_pts)
    minx, miny = all_pts[:, 0].min(), all_pts[:, 1].min()
    maxx, maxy = all_pts[:, 0].max(), all_pts[:, 1].max()
    pad = max((maxx - minx), (maxy - miny)) * 0.02 or 1
    minx, miny = minx - pad, miny - pad
    maxx, maxy = maxx + pad, maxy + pad
    w, h = maxx - minx, maxy - miny

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        '<g fill="black" stroke="none">',
    ]
    for d in path_parts:
        if d:
            lines.append(f'  <path d="{d}" fill-rule="evenodd"/>')
    lines.append("</g>")
    lines.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(out_path)


if __name__ == "__main__":
    main()
