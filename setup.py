try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	'name': 'Parsuralia',
	'version': '0.1',
	'url': 'https://github.com/parnellj/parsuralia',
	'download_url': 'https://github.com/parnellj/parsuralia',
	'author': 'Justin Parnell',
	'author_email': 'parnell.justin@gmail.com',
	'maintainer': 'Justin Parnell',
	'maintainer_email': 'parnell.justin@gmail.com',
	'classifiers': [],
	'license': 'GNU GPL v3.0',
	'description': 'Generates random verse from my corpus of poems using Markov chains.',
	'long_description': 'Generates random verse from my corpus of poems using Markov chains.',
	'keywords': '',
	'install_requires': ['nose'],
	'packages': ['parsuralia'],
	'scripts': []
}
	
setup(**config)
