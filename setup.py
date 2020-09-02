import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnde",
    version="0.0.4",
    author="Eric Winter",
    author_email="ewinter@stsci.edu",
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
)
