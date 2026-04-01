import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def list_dxf_layers(dxf_path):
    try:
        import ezdxf
    except ImportError:
        print("请先安装 ezdxf: pip install ezdxf")
        sys.exit(1)
    doc = ezdxf.readfile(dxf_path)
    layers = [l.dxf.name for l in doc.layers]
    print(f"文件: {dxf_path}")
    print(f"图层数量: {len(layers)}")
    for i, name in enumerate(layers):
        print(f"  {i+1}. {name}")


if __name__ == "__main__":
    path = os.path.join(ROOT_DIR, "128array_0927_2.dxf")
    if len(sys.argv) > 1:
        path = sys.argv[1]
    list_dxf_layers(path)
