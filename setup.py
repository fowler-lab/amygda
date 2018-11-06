from setuptools import setup
from amygda import __version__

setup(
    name='amygda',
    version=__version__,
    author='Philip W Fowler',
    scripts=['bin/analyse-plate-with-amygda.py'],
    packages=['amygda'],
    package_data={'':['../config/*']},
    install_requires=[
        "numpy >= 1.13",
        "datreant >= 1.0.2",
        "opencv-python >= 3.4",
        "scipy >= 1.1.0",
        "matplotlib >= 2.2.2"
    ],
    license='University of Oxford (see LICENCE.md)',
    long_description=open('README.md').read(),
    zip_safe=False
)
