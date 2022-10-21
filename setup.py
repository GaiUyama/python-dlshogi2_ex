from distutils.core import setup
from setuptools import find_packages

setup(
  name='python-dlshogi2_ex',
  packages=find_packages(
    include=["checkpoints", "model", "pydlshogi2"]
  )
)
