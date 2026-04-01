import os
import sys
import ezdxf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
from ezdxf.addons import Importer


def extract_layer(input_path, output_path, layer_name):
    sdoc = ezdxf.readfile(input_path)
    tdoc = ezdxf.new("R2010")
    msp = sdoc.modelspace()
    ents = msp.query(f'*[layer=="{layer_name}"]')
    count = len(ents)
    if count > 0:
        importer = Importer(sdoc, tdoc)
        importer.import_entities(ents, tdoc.modelspace())
        importer.finalize()
    tdoc.saveas(output_path)
    print(f"已提取图层 '{layer_name}' 的 {count} 个实体至 {output_path}")


if __name__ == "__main__":
    input_path = os.path.join(ROOT_DIR, "128array_0927_2.dxf")
    output_path = os.path.join(RESULTS_DIR, "128array_0927_2_1shank.dxf")
    layer_name = "5_Top SU8"
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        layer_name = sys.argv[3]
    extract_layer(input_path, output_path, layer_name)
