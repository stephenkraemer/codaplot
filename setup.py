from setuptools import find_packages, setup

setup(
        name='codaplot',
        version='0.1.0',
        description='description',
        long_description=__doc__,
        url='http://...',
        author='Stephen Kraemer',
        author_email='stephenkraemer@gmail.com',
        license='MIT',
        classifiers=(
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ),
        keywords='keyword1 keyword2',

        # explicit may be ok for small packages
        # packages=['mqc', 'mqc.pileup'],
        package_dir={'': 'src'},
        packages = find_packages(where='src', exclude=['contrib', 'docs', 'tests*']),

        # additional files, often data
        package_data = {},
        data_files = [],

        install_requires=[
            'matplotlib>=3.0.0',
            'pandas>=0.23',
            'seaborn>=0.9.0',
            # currently needs private fixed version, still need to submit PR
            'dynamicTreeCut',
            # only for python 3.6, improve this
            'dataclasses',
            'numpy',
            'numba',
            'scipy',
            'toolz',
            'more_itertools',
        ],
        python_requires='>=3.6',
        extras_require={
            'dev': [
                'pytest',
            ]
        }

)
