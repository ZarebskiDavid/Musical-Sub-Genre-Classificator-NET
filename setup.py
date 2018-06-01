from setuptools import setup


def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()


setup(
    name="stylecheckerm",
    version="2018.01.01",
    description="Convolutional Neural Network for Metal Sub-genre classification based on artwork (Covers)",
    long_description=readfile('README.md'),
    author="David R. L. Zarebski",
    author_email="zarebskidavid@gmail.com",
    url="http://zarebski.io/",
    py_modules=['stylecheckerm'],
    license=readfile('LICENCE'),
    entry_points={
        'console_scripts': [
            'stylecheckerm = stylecheckerm:stylecheckerm'
        ]
    },
)
