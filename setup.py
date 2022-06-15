# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='sudrabainiemakoni',
	version='0.01',
	description='Sudrabaino mākoņu sinhronās novērošanas datu apstrādes rīki',
	author='Juris Seņņikovs',
	author_email='jsenniko@latnet.lv',
	install_requires=['numpy',
		'pandas',
		'scipy',
		'matplotlib',
		'scikit-image','pyproj',
		'astropy','exifread','cameratransform','pymap3d','tilemapbase'],
	packages=['sudrabainiemakoni'],
	zip_safe=False)

__version__ = "0.1"