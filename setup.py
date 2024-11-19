from setuptools import setup, find_packages

setup(
    name='ot_bar',
    version='0.0.0',
    install_requires=[
        'requests',
        'importlib-metadata; python_version >= "3.9"',
    ],
    packages=find_packages()
)
