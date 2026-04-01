import os
import re
import shutil
import sys

from absl import app
from absl import flags

flags.DEFINE_string('data_dir', None, 'Dataset root (contains put-object-in-basket-{mode}-worker-*)')
flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_integer('n_per_worker', 200, 'Episodes per worker (must match collect n_per_object)')
FLAGS = flags.FLAGS

FIELDS = ('color', 'depth', 'action', 'reward', 'info')
NUM_OBJECTS = 6
PAT = re.compile(r'^(\d{6})-(\d+)\.pkl$')


def main(_):
  data_dir = os.path.abspath(FLAGS.data_dir or '.')
  mode = FLAGS.mode
  base_name = f'put-object-in-basket-{mode}'
  out_dir = os.path.join(data_dir, base_name)
  if os.path.isdir(out_dir) and not os.path.isdir(os.path.join(data_dir, f'{base_name}-worker-0')):
    print(f'{out_dir} already exists and no worker dirs found; abort to avoid overwrite')
    sys.exit(1)
  os.makedirs(out_dir, exist_ok=True)
  for f in FIELDS:
    os.makedirs(os.path.join(out_dir, f), exist_ok=True)

  n_per = FLAGS.n_per_worker
  total = 0
  for worker_id in range(NUM_OBJECTS):
    wdir = os.path.join(data_dir, f'{base_name}-worker-{worker_id}')
    if not os.path.isdir(wdir):
      print(f'Skip missing {wdir}')
      continue
    ep_offset = worker_id * n_per
    for field in FIELDS:
      path = os.path.join(wdir, field)
      if not os.path.isdir(path):
        continue
      for fname in sorted(os.listdir(path)):
        m = PAT.match(fname)
        if not m:
          continue
        local_ep, seed = int(m.group(1)), m.group(2)
        global_ep = ep_offset + local_ep
        new_name = f'{global_ep:06d}-{seed}.pkl'
        src = os.path.join(path, fname)
        dst = os.path.join(out_dir, field, new_name)
        shutil.copy2(src, dst)
    n_eps = len([f for f in os.listdir(os.path.join(wdir, 'action')) if PAT.match(f)])
    total += n_eps
    if n_eps:
      print(f'Worker {worker_id}: {n_eps} episodes -> global {ep_offset}-{ep_offset + n_eps - 1}')
    else:
      print(f'Worker {worker_id}: 0 episodes')
  print(f'Merged {total} episodes to {out_dir}')


if __name__ == '__main__':
  app.run(main)
