from setuptools import setup, find_packages

setup(
    name='synful',
    version='0.2.0',
    description='Synaptic Partner Detection in 3D Microscopy Volumes - CUDA 12.x Compatible',
    url='https://github.com/griffbad/synful_cuda12x',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
)
