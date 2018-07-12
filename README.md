# Band structure and g-factor numerical calculations in Ge/Si core/shell nanowires

This repository contains simulation codes and raw data of numerical calculations included in [1806.01815](https://arxiv.org/abs/1806.01815).




# Reproducible research environment

We provide a reproducible research environment in form of a [Docker](https://www.docker.com/) image.
Please refer to Docker [documentation](https://docs.docker.com/install/) for the installation instruction.
We provide image that is based on [jupyter/minimal-notebook](https://hub.docker.com/r/jupyter/minimal-notebook/) image.

Once Docker is installed, the pre-built our docker image can be obtained and run with

    docker pull rafalskolasinski/sige_wires:v1.0.0
    docker run --rm -d -p 8888:8888 -v $PWD:/home/jovyan/work --name sige_wires rafalskolasinski/sige_wires:v1.0.0

This will mount current working directory ($PWD) under "~/work" directory inside the container.
Please either "cd" into the directory with the simulation codes and data prior to executing above command or provide path to it instead of "($PWD)".

The container will expose a jupyter notebook to the local port 8888.

Running

    docker logs sige_wires

will show information with "token" required to access jupyter notebook.
The last remainig step is to open web browser and go to the given link that looks similar to

    http://localhost:8888/?token=<TOKEN>
