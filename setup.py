
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

classifiers = """\
    Development Status :: 4 - Beta
    Operating System :: OS Independent
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.3
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Programming Language :: Python :: 3.6
"""
install_requires = [
    'numpy>=1.9', 
    'pandas>=0.17',
]

setup(
    name='cont3d',
    author='Maksim Imakaev',
    author_email='mimakaev@gmail.com',
    version='0.1.0',
    license='MIT',
    description='Tools for working with 3-point contact matrices',
    long_description=('''A starting point for the functions working with 3-point matrices
			Would include both dense and sparse utilities, and tools to create 
			them from polymer conformations'''),
    url='https://github.com/mimakaev/cont3d',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[s.strip() for s in classifiers.split('\n') if s],
    install_requires=install_requires,

)
