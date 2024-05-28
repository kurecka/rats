from setuptools import setup, find_packages

setup(
    name="raylite",
    version="2024.5.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'raylite=raylite.__main__:main',
        ],
    },
    install_requires=[
        "click",
        "omegaconf",
        "termcolor",
        "pyyaml",
    ],
    author="Formela",
    description="A simple replacement of a malfunction ray cluster commands.",
)