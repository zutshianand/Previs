from setuptools import setup

setup(
    name='previs',
    packages=['previs'],
    version='1.01',
    description='PREVIS stands for PREprocessing and VIsualisation. '
                'This package consists of essential code snippets required for preprocessing and '
                'visualisation of data science tasks.',
    author='Anand Zutshi (zutshianand)',
    author_email='123anandji@gmail.com',
    url='https://github.com/zutshianand/Previs.git',
    download_url='https://github.com/zutshianand/Previs/archive/1.01.tar.gz',
    keywords=['dataloaders', 'preprocessors', 'featuregeneration', 'convenience', 'visualisation', 'pytorch', 'models',
              'deep-learning', 'dataset', 'snippets'],
    classifiers=[],
    install_requires=[
        'autogluon.tabular>=0.8.0',
    ],
)
