from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()


setup(
   name='hyspeclib',
   version='1.0',
   description='Deep learning library for crop classification using hyperspectral image',
   author='Hetul V Patel',
   author_email='hetulvp@gmail.com',
   packages=['hyspeclib'],  #same as name
   install_requires=['pandas', 'spectral', 'tensorflow'], #external packages as dependencies
)
