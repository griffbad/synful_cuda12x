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
            'numpy>=1.21.0',
            'pymongo>=4.0.0',
            'zarr>=2.10.0',
            'scikit-learn>=1.0.0',
            'pandas>=1.3.0',
            'tensorflow>=2.12.0,<2.16.0',
        ],
)
