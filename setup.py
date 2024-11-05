from setuptools import setup

def read_long_description():
    with open('README.md', 'r') as open_file_description:
        return open_file_description.read()

def read_requirements():
    with open('requirements.txt', 'r') as open_file_requirements:
        return open_file_requirements.read()

setup(
    name='DBOpt',
    version='1.0.0',
    description='Bayesian optimized parameter selection for density-based clustering algorithms',
    long_description_content_type='text/markdown',
    long_description=read_long_description(),
    author='Joseph L. Hammer, Alexander J. Devanny',
    author_email='jhammer3018@gmail.com',
    url='https://github.com/Kaufman-Lab-Columbia/DBOpt',
    packages=['DBOpt'],
    license ="MIT",
    keywords = "cluster clusters clustering DBSCAN HDBSCAN OPTICS",
    install_requires=read_requirements(),
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering'
                 ]
)