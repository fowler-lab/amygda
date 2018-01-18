from distutils.core import setup

setup(
    install_requires=[
        "numpy >= 1.13",
        "datreant >= 0.7",
        "opencv-python >= 3.4"
    ],
    name='amygda',
    version='0.2.0',
    author='Philip W Fowler',
    packages=['amygda'],
    license='University of Oxford (see LICENCE.md)',
    long_description=open('README.md').read(),
)
