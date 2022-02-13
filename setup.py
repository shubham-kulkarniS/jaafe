# setup.py
from setuptools import setup

INSTALL_REQUIRES = [
    'numpy>=1.13.3',
    'scipy>=0.19.1',
    'scikit-learn>=0.22',
    'joblib>=0.11',
     # 'sounddevice',
      'librosa',
    #   'xgboost',
      'uuid',
      'h5py',
      'Sphinx'
]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'numpydoc',
        'matplotlib',
        'pandas',
    ]
}


setup(name="yaafelib",
      description="Yet another audio feature library",
      version="0.02",
      packages=["yaafelib"],
      keywords=['python', 'audio extraction'],
      classifiers= [
      "Development Status :: 3 - Alpha",
      "Intended Audience :: Education",
      "Programming Language :: Python :: 3",
      "Operating System :: Microsoft :: Windows",
        ],
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)

