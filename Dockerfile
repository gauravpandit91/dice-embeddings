FROM python:3.10
#RUN pip install dicee==0.0.4
RUN pip3 install "pandas>=1.5.1" "torch>=2.0.0" "polars>=0.16.14" "scikit-learn>=1.2.2" "pyarrow>=11.0.0" "pytest>=7.2.2" "gradio>=3.23.0" "psutil>=5.9.4" "pytorch-lightning==1.6.4" "pykeen==1.10.1" "zstandard>=0.21.0"
WORKDIR /dicee
ADD . .
CMD ./main.py
