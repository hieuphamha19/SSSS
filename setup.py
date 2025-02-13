from setuptools import setup, find_packages

required = [
    'torch>=1.7.0',
    'torchvision>=0.8.0',
    'numpy>=1.19.2',
    'pandas>=1.1.3',
    'PyYAML>=5.3.1',
    'tqdm>=4.50.2',
    'opencv-python>=4.4.0',
    'scikit-learn>=0.23.2',
]

extras_require = {
    'profiling': ['thop'],
    'medical': ['medpy'],
    'full': ['thop', 'medpy'],
}

setup(
    name="your_project",
    version="0.1",
    packages=find_packages(),
    install_requires=required,
    extras_require=extras_require,
) 