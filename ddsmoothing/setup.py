import os.path
import setuptools

repository_dir = os.path.dirname(__file__)

with open(os.path.join(repository_dir, "requirements.txt")) as fh:
    requirements = [line for line in fh.readlines()]

setuptools.setup(
    name="ddsmoothing",
    version=1.0,
    author="Motasem Alfarra",
    author_email="motasem.alfarra@kaust.edu.sa",
    python_requires=">=3.7",
    description="Data-dependent smoothing Python package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7"
    ],
    install_requires=requirements,
    include_package_data=True,
)
