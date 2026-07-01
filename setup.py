from setuptools import find_packages, setup

setup(
    name="seguro-antirobos-mlops",
    version="1.0.0",
    description="Pipeline MLOps para el modelo de propensión al Seguro Antirrobos",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.11",
)
