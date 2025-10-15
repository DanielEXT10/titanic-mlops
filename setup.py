from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name ="titanic-mlops",
    version="0.1",
    author="DanielGP",
    packages=find_packages(),
    install_requires = requirements

)