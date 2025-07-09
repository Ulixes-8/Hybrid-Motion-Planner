from setuptools import setup, find_packages

setup(
    name="chimera",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'chimera': ['config/**/*.yaml'],
    },
)
