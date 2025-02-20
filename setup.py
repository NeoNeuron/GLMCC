import setuptools

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="glmcc",
    author="Kai Chen",
    author_email="kchen513@sjtu.edu.cn",

    version="0.0.1",
    url="https://github.com/NeoNeuron/GLMCC",

    description="Package for GLMCC.",

    install_requires=requirements,
    packages=setuptools.find_packages(),
)