from setuptools import setup
from setuptools import find_packages

long_description = '''
ml-toolbox is built to provide the generic function for
building machine learning models. It has following features:
- data preprocessing
- feature engineering
- eda(in-progress)
- Machine Learning Models
- deep learning based models
These features is built using existed and my own implementation
of recent research projects.
More details about each of these features, can be find in 
README file.
'''

setup(name='ml-toolbox',
      version='1.0.0',
      description='Machine Learning and Data Science Strategy for data-modelling',
      long_description=long_description,
      author='Ankish Bansal',
      author_email='bansal.ankish1@gmail.com',
      url='https://github.com/ankishb/ml-toolbox',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'keras_applications>=1.0.6',
                        'keras_preprocessing>=1.0.5'],
      },
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Build tool',
      ],
      packages=find_packages())