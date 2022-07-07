from setuptools import setup
from setuptools import find_packages
# App dependencies
app_requirements = [
      'Deprecated>=1.2.6',
      'GitPython>=2.1.11',
      'numpy>=1.18.1',
      'pandas>=1.0.*',
      'pillow>=9.0.1',
      'plotly>=4.5.*',
      'pytest>=5.2.1',
      'pytest-cov>=2.5.1',
      'pytest-mock>=1.10.0',
      'pytest-pycharm>=0.5.0',
      'pyyaml>=5.4.1',
      'scikit-learn>=0.23.1',
      'scipy>=1.6.3',
      'torch==1.11.0',
      'torchvision>=0.12.0',
      'tox>=3.14.3',
      'tqdm>=4.37.0',
]

# Dev dependencies
dev_requirements = [
]

setup(name='pytorch_cifar100',
      description='PyTorch Models on cifar100',
      long_description='PyTorch Models on cifar100 dataset, for practices by Peter Li ',
      url='https://github.com/PeterrrrLi/pytorch_cifar100',
      license='All Rights Reserved Peter Li',
      packages=find_packages('.'),
      package_dir={'': '.'},
      zip_safe=False,
      install_requires=app_requirements,
      include_package_data=True,
      extras_require={'dev': dev_requirements}
      )
