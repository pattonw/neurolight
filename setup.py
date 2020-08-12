#from distutils.core import setup
from setuptools import setup, find_packages

setup(
        name='neurolight',
        version='0.1',
        description='Neuron reconstruction from light microscopy data.',
        url='https://github.com/maisli/neurolight',
        author='Lisa Mais',
        author_email='Lisa.Mais@mdc-berlin.de',
        license='MIT',
        entry_points='''
            [console_scripts]
            neurolight=neurolight.cli.cli:neurolight
        ''',
        packages=find_packages()
)
