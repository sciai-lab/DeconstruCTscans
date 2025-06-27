from setuptools import find_packages, setup

setup(
    name="deconstruct",
    version="0.1",
    description="Fully automated pipeline for parsing complex CT assemblies",
    author="Peter Lippmann",
    author_email="peter.lippmann@iwr.uni-heidelberg.de",
    license="GNU General Public License v2.0",
    url="https://github.com/sciai-lab/DeconstruCTscans",
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
)