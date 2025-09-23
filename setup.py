from setuptools import setup, find_packages

setup(
        name='synful',
        version='0.1dev',
        description='Synaptic Partner Detection in 3D Microscopy Volumes.',
        url='https://github.com/funkelab/synful',
        license='MIT',
        packages=find_packages(),
        python_requires='>=3.8',
        install_requires=[
            'numpy>=1.20.0',
            'scipy>=1.7.0',
            'scikit-learn',
            'pandas',
            'pymongo',
            'zarr',
            'neuroglancer>=2.0.0',
            'cloud-volume',
            'navis',
        ],
        extras_require={
            'tensorflow': ['tensorflow>=2.12.0'],
            'tensorflow-gpu': ['tensorflow[and-cuda]>=2.12.0'],
            'dev': ['flake8', 'pytest', 'pytest-cov'],
        }
)
