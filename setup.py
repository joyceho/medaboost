from setuptools import setup

setup(name='medaboost',
      version='0.1',
      description='MedaBoost Model',
      url='http://github.com/joyceho/medaboost',
      author='Joyce Ho',
      author_email='joyce.c.ho@emory.edu',
      license='MIT',
      packages=['medaboost'],
      install_requires=[
                       'numpy',
                       'pandas',
                       'pymongo',
                       'pytest',
                       'scikit-learn',
                       'tqdm'
                     ],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      zip_safe=False)
