import setuptools

setuptools.setup(
    name="dataloader",
    version="1.0.0",
    description="DataLoader for ML applications",
    install_requires=[
        'arrow',
        'docopt',
        'joblib',
        'numpy',
        'pandas',
        'progressbar2',
        'PyYAML',
        'scikit-learn',
        'scipy'
    ]
)
