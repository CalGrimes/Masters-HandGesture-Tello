from setuptools import setup

setup(
    name='TelloEdgeML',
    version='0.1',
    description='Gesture Recognition and Face Tracking using Tello',
    author='CalGrimes',
    author_email='calgrimes29@gmail.com',
    url='https://github.com/CalGrimes/TelloEdgeML',
    packages=['TelloEdgeML'],
    install_requires=[
        'numpy',
        'mediapipe',
        'matplotlib',
        'opencv-python',
        'tensorflow',
        'scikit-learn',
        'tellopy',
        'pandas',
        'tqdm'
    ]
    )