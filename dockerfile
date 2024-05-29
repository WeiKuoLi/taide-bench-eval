# Start from a base image with conda pre-installed
FROM continuumio/anaconda3

# Install git to clone the repository
RUN apt-get update && apt-get install -y git


RUN apt-get update && apt-get install -y sudo
RUN useradd -m myuser && adduser myuser sudo

USER root
RUN apt-get update && apt-get install -y vim
USER myuser


# Set the working directory
WORKDIR /home/myuser

# Clone the repository
RUN git clone https://github.com/WeiKuoLi/taide-bench-eval.git

# Change to the repository directory
WORKDIR /home/myuser/taide-bench-eval

# Create the conda environment using libmamba solver
RUN conda env create -f environment.yml --solver libmamba

# Activate the environment
SHELL ["conda", "run", "-n", "taide-bench", "/bin/bash", "-c"]

# Keep the container running
CMD ["/bin/bash"]
