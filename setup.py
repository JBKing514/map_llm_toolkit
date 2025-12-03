from setuptools import setup, find_packages

setup(
    name="map-llm-toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "matplotlib",
        "scikit-learn",
    ],
    author="Yunchong Tang",
    author_email="d232901@st.tohtech.ac.jp",
    description="A minimal toolkit for extracting, projecting, and visualizing MAP reasoning trajectories in LLMs.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JBKing514/map_llm_toolkit",
    license="MIT",
    python_requires=">=3.8",
)
