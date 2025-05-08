from setuptools import setup, find_packages

setup(
    name="cryo_cooling_test",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "reportlab",
        "pytest",
        "pytest-cov"
    ],
) 