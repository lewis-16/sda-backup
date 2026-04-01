import os
import sys


def dwg_to_dxf(dwg_path, dxf_path=None):
    if not os.path.exists(dwg_path):
        print(f"文件不存在: {dwg_path}")
        sys.exit(1)
    if dxf_path is None:
        base, _ = os.path.splitext(dwg_path)
        dxf_path = base + ".dxf"

    try:
        from ezdxf.addons import odafc
        if odafc.is_installed():
            odafc.convert(dwg_path, dxf_path, version="R2010", replace=True)
            print(f"已转换 (ODA): {dwg_path} -> {dxf_path}")
            return dxf_path
    except Exception:
        pass

    err_ezdwg = None
    try:
        import ezdwg
        ezdwg.to_dxf(dwg_path, dxf_path, dxf_version="R2010")
        print(f"已转换 (ezdwg): {dwg_path} -> {dxf_path}")
        return dxf_path
    except Exception as e:
        err_ezdwg = e

    print("转换失败。可用方案:")
    print("1. 安装 ODA File Converter: https://www.opendesign.com/guestfiles/oda_file_converter")
    print("   安装后本脚本将自动使用 ODA 进行转换")
    print("2. 使用 AutoCAD 或 LibreCAD 等软件手动另存为 DXF")
    if err_ezdwg:
        print(f"   原错误 (ezdwg): {err_ezdwg}")
    sys.exit(1)


if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "128array_0927_2.dwg")
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    dwg_to_dxf(input_path, output_path)
