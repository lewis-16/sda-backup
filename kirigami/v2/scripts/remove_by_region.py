import os
import sys
import ezdxf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
from ezdxf.addons import Importer


def get_entity_center_x(entity):
    try:
        if hasattr(entity, "bbox") and entity.bbox() is not None:
            bbox = entity.bbox()
            return (bbox.extmin.x + bbox.extmax.x) / 2
    except Exception:
        pass
    try:
        if entity.dxftype() in ("CIRCLE", "ARC"):
            return entity.dxf.center.x
        if entity.dxftype() == "LINE":
            return (entity.dxf.start.x + entity.dxf.end.x) / 2
        if entity.dxftype() == "LWPOLYLINE":
            pts = list(entity.vertices_in_wcs()) if hasattr(entity, "vertices_in_wcs") else list(entity.get_points("xy"))
            if pts:
                return sum(p[0] if hasattr(p, "__getitem__") else p.x for p in pts) / len(pts)
    except Exception:
        pass
    return 0


def remove_by_region(input_path, output_path, keep_x_max=0):
    sdoc = ezdxf.readfile(input_path)
    tdoc = ezdxf.new("R2010")
    msp = sdoc.modelspace()
    keep_ents = []
    remove_count = 0
    for entity in msp:
        cx = get_entity_center_x(entity)
        if cx <= keep_x_max:
            keep_ents.append(entity)
        else:
            remove_count += 1
    if keep_ents:
        importer = Importer(sdoc, tdoc)
        importer.import_entities(keep_ents, tdoc.modelspace())
        importer.finalize()
    tdoc.saveas(output_path)
    print(f"已保留 {len(keep_ents)} 个实体，删除 {remove_count} 个 (x>{keep_x_max})")
    print(f"保存至 {output_path}")


if __name__ == "__main__":
    input_path = os.path.join(RESULTS_DIR, "128array_0927_2_1shank.dxf")
    output_path = os.path.join(RESULTS_DIR, "test.dxf")
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    remove_by_region(input_path, output_path, keep_x_max=0)
