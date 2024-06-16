from setuptools import setup, find_packages

setup(
    name='rats_manhattan',
    version='2024.6.0',
    packages=find_packages(),
    install_requires=[
        'fimdpenv',
        'fimdp',
        'folium',
        'networkx',
    ],

    author="RATS",
    description='AEVEnv simulator, built on top of FiMDPEnv.',
)
