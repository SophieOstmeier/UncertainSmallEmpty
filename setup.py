from setuptools import setup

setup(
    name='UncertainSmallEmpty',
    version='0.1',
    packages=['UncertainSmallEmpty'],
    url='https://github.com/SophieOstmeier/UncertainSmallEmpty.git',
    license='Apache License Version 2.0, January 2004',
    author='SophieOstmeier',
    author_email='sostm@stanford.edu',
    description='Evaluation for Medical Image Segemantion Models for Uncertain, Small and Empty Reference Annotations',
    install_requires=[
        "numpy",
        "MedPy",
        "scipy",
        "pandas",
        "SimpleITK",
        "batchgenerators",
        "nibabel",
        "natsort",
        "argparse",
        "ctg-surface-distance",
        "openpyxl"
    ]
)
