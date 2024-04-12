from setuptools import setup, find_packages


def get_version():
    from pathlib import Path

    version_path = Path(__file__).parent / "VERSION"
    with version_path.open() as version_file:
        return version_file.read().strip()


setup(
    name="meteothek",
    version=get_version(),
    packages=find_packages(),
    install_requires=["numpy", "pandas", "xarray", "cartopy", "matplotlib"],
)
