from setuptools import setup, find_packages

requirements = [
    "funcy",
    "matplotlib",
    "numpy==1.21",
    "shapely",
    "tgt",
    "torch",
    "torchaudio"
]

setup(
    name="vt_shape_gen",
    version="0.0.1",
    description="Automatic generation of the vocal tract shape from the sequence of phonemes to be articulated",
    author="Vinicius Ribeiro",
    author_email="vinicius.souza-ribeiro@loria.fr",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requirements=requirements
)
