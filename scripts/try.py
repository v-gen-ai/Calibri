import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from absl import app, flags
from ml_collections import config_flags


_CONFIG = config_flags.DEFINE_config_file("config", "configs/base.py", "Training configuration.")
# _MY_FLAG = flags.DEFINE_integer('flag', None, "flag help msg")

_MY_FLAG = [flags.DEFINE_integer(t, None, f"{t} help msg") for t in ["flag1", "flag2", "flag3"]]


def main(_):
  print(_CONFIG.value)
  print("---")
  print("---")
  print("---")
  print(_MY_FLAG[0].value)
  print(_MY_FLAG[1].value)
  print(_MY_FLAG[2].value)
  print("---")
  print("---")
  print("---")

if __name__ == '__main__':
  app.run(main)