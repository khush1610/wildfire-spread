
"""Utilities for working with local files."""
import os
from typing import Text


f_exists = os.path.exists
f_makedirs = os.makedirs


def maybe_make_dirs(path):
  """Creates the sub directories leading up to the given file.

  Args:
    path: The path to a file (i.e., the last word in the path is assumed to be a
      file, not a directory).

  Returns:
    The path generated.
  """
  dirname = os.path.dirname(path)
  if dirname and not f_exists(dirname):
    f_makedirs(dirname)
  return dirname