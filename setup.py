import setuptools


setuptools.setup(
    name="nnde",
    version="0.0.7",
    author="Eric Winter",
    author_email="eric.winter62@gmail.com",
    description=("A package implementing a collection of neural networks to"
                 " solve ordinary and partial differential equations"),
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/elwinter/nnde",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        ("License :: OSI Approved :: GNU Lesser General Public License v3"
         " (LGPLv3)"),
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    tests_require=['nose', 'pytest'],
    test_suite='nose.collector',
    install_requires=['matplotlib', 'numpy', 'scipy']
)
