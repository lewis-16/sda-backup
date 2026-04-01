import os
import pickle
import sys

from absl import app
from absl import flags
import numpy as np

flags.DEFINE_string('export_dir', None, 'Dir from export_for_openvla.py (episode_*/ with instruction.txt, image_*.npy, actions.pkl)')
flags.DEFINE_string('out_dir', None, 'Output dir for RLDS TFRecord and metadata')
flags.DEFINE_integer('max_episodes', None, 'Max episodes to convert (default all)')
flags.DEFINE_string('split', 'train', 'train or test')
FLAGS = flags.FLAGS


def main(_):
  export_dir = os.path.abspath(FLAGS.export_dir or '.')
  out_dir = os.path.abspath(FLAGS.out_dir or (export_dir + '-rlds'))
  if not os.path.isdir(export_dir):
    print(f'export_dir not found: {export_dir}')
    sys.exit(1)
  try:
    import tensorflow as tf
  except ImportError:
    print('tensorflow required: pip install tensorflow')
    sys.exit(1)

  ep_dirs = sorted([d for d in os.listdir(export_dir) if os.path.isdir(os.path.join(export_dir, d)) and d.startswith('episode_')])
  if FLAGS.max_episodes is not None:
    ep_dirs = ep_dirs[:FLAGS.max_episodes]
  os.makedirs(out_dir, exist_ok=True)

  num_shards = max(1, (len(ep_dirs) + 499) // 500)
  records_per_shard = (len(ep_dirs) + num_shards - 1) // num_shards
  written = 0
  writer = None
  shard = -1

  for ep_dir_name in ep_dirs:
    ep_path = os.path.join(export_dir, ep_dir_name)
    instr_path = os.path.join(ep_path, 'instruction.txt')
    actions_path = os.path.join(ep_path, 'actions.pkl')
    if not os.path.isfile(instr_path) or not os.path.isfile(actions_path):
      continue
    with open(instr_path, 'r') as f:
      instruction = f.read().strip()
    with open(actions_path, 'rb') as f:
      actions = pickle.load(f)
    image_paths = sorted([f for f in os.listdir(ep_path) if f.startswith('image_') and f.endswith('.npy')])
    if not image_paths or len(actions) != len(image_paths) - 1:
      continue
    if written % records_per_shard == 0:
      if writer:
        writer.close()
      shard += 1
      out_name = f'put_basket_{FLAGS.split}-{shard:05d}-of-{num_shards:05d}.tfrecord'
      writer = tf.io.TFRecordWriter(os.path.join(out_dir, out_name))

    image_list = tf.train.FeatureList(feature=[])
    action_list = tf.train.FeatureList(feature=[])
    for step in range(len(image_paths)):
      img = np.load(os.path.join(ep_path, image_paths[step]))
      image_list.feature.append(tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.astype(np.uint8).tobytes()])))
      if step < len(actions):
        act = actions[step]
        p0 = np.asarray(act['pose0'][0], dtype=np.float32)
        p1 = np.asarray(act['pose1'][0], dtype=np.float32)
        action_list.feature.append(tf.train.Feature(float_list=tf.train.FloatList(value=np.concatenate([p0, p1]).tolist())))
      else:
        action_list.feature.append(tf.train.Feature(float_list=tf.train.FloatList(value=[0.0] * 6)))
    seq = tf.train.SequenceExample(
      context=tf.train.Features(feature={
        'language_instruction': tf.train.Feature(bytes_list=tf.train.BytesList(value=[instruction.encode('utf-8')])),
      }),
      feature_lists=tf.train.FeatureLists(feature_list={
        'image': image_list,
        'action': action_list,
      }))
    writer.write(seq.SerializeToString())
    written += 1
  if writer:
    writer.close()
  with open(os.path.join(out_dir, 'dataset_info.json'), 'w') as f:
    f.write('{"name": "put_basket", "image_size": [480, 640], "action_dim": 6}\n')
  print(f'Wrote {written} episodes to {out_dir}')


if __name__ == '__main__':
  app.run(main)
