from setuptools import setup, find_packages

setup(
    name="heart_disease_diagnosis",
    version="0.1.0",
    author="Nguyen Dinh Phuc",
    author_email="ndphuc3112@gmail.com.com",
    description="Ensemble Learning project for heart disease diagnosis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nphuc0311/Heart_Disease_Diagnosis.git",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "PyYAML",
    ],
    python_requires=">=3.12",
)
