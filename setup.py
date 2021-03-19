import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnde",
    version="0.0.6",
    author="Eric Winter",
    author_email="eric.winter62@gmail.com",
    description=("A package implementing a collection of neural networks to"
                 " solve ordinary and partial differential equations"),
    long_description=long_description,
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
    tests_require=['nose'],
    test_suite='nose.collector',
)
