from setuptools import find_packages, setup

setup(
        name='complex_heatmap',
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
            'pandas',
            'numpy',
            'pytest',
            'pytest-mock',
            'matplotlib',
        ],
        python_requires='>=3.6',

)
