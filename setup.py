import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cv-labs",
    version="0.0.1",
    author="Adrian-Gabriel Bălănescu",
    author_email="balanescuadrian71@gmail.com",
    description="Some computer vision projects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adrianB3/cv_labs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['opencv-python', 'PyQt5']
)
