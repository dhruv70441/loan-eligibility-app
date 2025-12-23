from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    
    HYPHEN_E_DOT = "-e ."
    with open(file_path, encoding='utf-8') as f:
        requirements=[
            line.strip()
            for line in f.read().splitlines()
            if line.strip() and line.strip() != HYPHEN_E_DOT
        ]
    return requirements


setup(
    name="loan-eligibility",
    version="0.1.0",
    description="Loan eligibility prediction system",
    author="Dhruv Parmar",
    author_email="dhruvparmar70441@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(file_path='requirements.txt')
)