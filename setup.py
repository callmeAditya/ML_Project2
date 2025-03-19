from setuptools import setup, find_packages

def get_packages(requirements_file):
    with open(requirements_file) as f:
        return f.read().splitlines()

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=get_packages("requirements.txt"),
    
)