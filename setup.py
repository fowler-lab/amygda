from setuptools import setup

setup(
    name='amygda',
    version='1.1.0',
    author='Philip W Fowler',
    packages=['amygda'],
    install_requires=[
        "numpy >= 1.13",
        "datreant.core >= 0.7",
        "opencv-python >= 3.4"
    ],
    license='University of Oxford (see LICENCE.md)',
    long_description=open('README.md').read(),
)
