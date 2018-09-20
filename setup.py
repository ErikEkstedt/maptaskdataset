import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="maptask",
    version="0.0.1",
    author="Erik",
    author_email="eeckee@gmail.com",
    description="The Maptask dataset for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ErikEkstedt/maptaskdataset",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux/MacOS",
    ],
)
