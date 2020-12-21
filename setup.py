from setuptools import setup, find_packages

setup(
    name='shapley',
    version='0.1.0',
    packages=[
        'shapley', 
        'shapley.apps', 
        'shapley.embedding',
        'shapley.loader',
        'shapley.measures',
        'shapley.models', 
        'shapley.utils', 
    ],
)