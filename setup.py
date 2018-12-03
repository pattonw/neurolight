#from distutils.core import setup
from setuptools import setup

setup(
        name='neurolight',
        version='0.1',
        description='Neuron reconstruction from light microscopy data.',
        url='https://github.com/maisli/neurolight',
        author='Lisa Mais',
        author_email='Lisa.Mais@mdc-berlin.de',
        license='MIT',
        packages=[
            'neurolight',
            'neurolight.gunpowder',
        ]
)
