"""
Generate one URDF per OBJ for use with Ravens put-object-in-basket task.
Put this script in ravens/scripts/ and run from ravens repo root.

Usage:
  python scripts/generate_urdf_from_obj.py

Expects: ravens/environments/assets/put_basket/*.obj
Creates: ravens/environments/assets/put_basket/*.urdf (same name, .urdf)
Mesh path in URDF is relative (e.g. "apple.obj") so URDF and OBJ must be in the same folder.
"""

import os

RAVENS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_PUT_BASKET = os.path.join(RAVENS_ROOT, 'ravens', 'environments', 'assets', 'put_basket')
TEMPLATE_PATH = os.path.join(ASSETS_PUT_BASKET, 'obj_to_urdf_template.urdf')

DEFAULT_SCALE = "0.02 0.02 0.02"


def main():
  if not os.path.isdir(ASSETS_PUT_BASKET):
    os.makedirs(ASSETS_PUT_BASKET)
    print(f"Created {ASSETS_PUT_BASKET}. Put your .obj files there and run again.")
    return

  with open(TEMPLATE_PATH, 'r') as f:
    template = f.read()

  objs = [f for f in os.listdir(ASSETS_PUT_BASKET) if f.lower().endswith('.obj')]
  if not objs:
    print(f"No .obj in {ASSETS_PUT_BASKET}. Put apple.obj, pear.obj, etc. there.")
    return

  for obj in sorted(objs):
    base = os.path.splitext(obj)[0]
    urdf_content = template.replace('MESHNAME', obj)
    urdf_content = urdf_content.replace('SCALEX SCALEY SCALEZ', DEFAULT_SCALE)
    urdf_path = os.path.join(ASSETS_PUT_BASKET, base + '.urdf')
    with open(urdf_path, 'w') as f:
      f.write(urdf_content)
    print(f"Wrote {urdf_path} (mesh={obj})")

  print("Done. Use object_config with urdf in put_object_in_basket task.")


if __name__ == '__main__':
  main()
