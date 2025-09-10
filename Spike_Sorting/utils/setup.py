from setuptools import setup, find_packages

setup(
    name="spike_sorting_utils",          # 包名称
    version="0.1.1",            # 版本号
    author="Lewis-16",         # 作者名
    description="Utils of spike sorting",
    packages=find_packages(),   # 自动查找包
    install_requires=[          # 声明依赖
        'numpy>=1.24.0',        # 最小版本号
        'pandas>=2.0.0',
        'scipy>=1.14.0',
        'scikit-learn>=1.5.0'
    ],
    python_requires='>=3.8',    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)