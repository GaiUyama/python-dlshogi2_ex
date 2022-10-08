from distutils.core import setup
from setuptools import find_packages

setup(
  name='re.python-dlshogi2',
  packages=find_packages(
    include=["checkpoints", "model", "pydlshogi2"]
  )
)
