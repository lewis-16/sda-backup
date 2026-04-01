import numpy as np
import ezdxf
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _polyline_to_points(entity):
    pts = []
    if entity.dxftype() == "LWPOLYLINE":
        for p in entity.get_points():
            pts.append((p[0], p[1]))
    elif entity.dxftype() == "POLYLINE":
        for v in entity.vertices:
            pts.append((v.dxf.location.x, v.dxf.location.y))
    elif entity.dxftype() == "LINE":
        pts.append((entity.dxf.start.x, entity.dxf.start.y))
        pts.append((entity.dxf.end.x, entity.dxf.end.y))
    return pts


def _points_to_polygon(pts, buffer_zero=False):
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
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


def _polys_to_polygon_with_holes(polys):
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    sorted_polys = sorted(polys, key=lambda g: g.area, reverse=True)
    outer = sorted_polys[0]
    holes = []
    for p in sorted_polys[1:]:
        if p.area < 1e-10:
            continue
        if outer.contains(p.representative_point()):
            holes.append(p.exterior.coords[:-1])
    if not holes:
        return outer
    try:
        return Polygon(outer.exterior.coords[:-1], holes=holes)
    except Exception:
        return outer


def dxf_layers_to_material(dxf_path, material_layer="Kirigami_platte", fallback_layers=None):
    if fallback_layers is None:
        fallback_layers = []
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    layer_names = [l.dxf.name for l in doc.layers]
    polys = []
    if material_layer in layer_names:
        ents = msp.query(f'*[layer=="{material_layer}"]')
        for e in ents:
            pts = _polyline_to_points(e)
            p = _points_to_polygon(pts, buffer_zero=True)
            if p is not None:
                polys.append(p)
        if polys:
            result = _polys_to_polygon_with_holes(polys)
            if result is not None and hasattr(result, "make_valid"):
                result = result.make_valid()
            return result
    elif fallback_layers:
        outline_geoms = []
        bridge_geoms = []
        kirigami_geoms = []
        for layer in fallback_layers:
            if layer in layer_names:
                ents = msp.query(f'*[layer=="{layer}"]')
                for e in ents:
                    pts = _polyline_to_points(e)
                    p = _points_to_polygon(pts, buffer_zero=True)
                    if p is not None:
                        if layer == "bridge_shank":
                            bridge_geoms.append(p)
                        elif layer == "KIRIGAMI":
                            kirigami_geoms.append(p)
                        else:
                            outline_geoms.append(p)
        if "KIRIGAMI" in layer_names and "KIRIGAMI" in fallback_layers:
            ents = msp.query('*[layer=="KIRIGAMI"]')
            for e in ents:
                pts = _polyline_to_points(e)
                p = _points_to_polygon(pts, buffer_zero=False)
                if p is not None:
                    kirigami_geoms.append(p)
        if outline_geoms or bridge_geoms:
            outline_plus_bridge = unary_union(outline_geoms + bridge_geoms)
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
            if not material.is_empty:
                if material.geom_type == "Polygon":
                    polys = [material]
                elif material.geom_type == "MultiPolygon":
                    polys = list(material.geoms)
                else:
                    polys = []
            else:
                polys = []
        else:
            polys = []
    if not polys:
        return None
    result = unary_union(polys) if len(polys) > 1 else polys[0]
    if hasattr(result, "make_valid"):
        result = result.make_valid()
    return result
