from setuptools import setup, find_packages

setup(
   name='derp_learning',
   version='0.1',
   url='https://github.com/willsheffler/derp_learning',
   author='Will Sheffler',
   author_email='willsheffler@gmail.com',
   description='oof, you call this maching learning?',
   packages=find_packages(),
   install_requires=[
      'pytest',
      'tqdm',
   ],
)
