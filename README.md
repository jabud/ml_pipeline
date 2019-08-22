# Pipeline Example

Machine learning pipeline challenge.

The Challenge consists on the following:

1.- Create a Docker container with a DB and store with some dataset in it.
2.- Create a 2nd Docker container with all the procedure tu create a ML model and train one using the previous dataset.
3.- Create a 3rd Docker container with a 2nd Database for storing results from our model.

The dataset chosen was the classic titanic dataset and we are going to create a pipeline for a classification solution.

I am using PostgreSql for the DBs, python3.5, Docker and a bunch of libraries specified in the requirements.txt inside the folder `prod`.


## How to run

1.- Open 2 terminals build and run each container to get our DBs started.

BUILD container 1:

`cd database`

`docker build --rm=true -t jfreek/postgresql:1 .`

BUILD container 2:

`cd results`

`docker build --rm=true -t jfreek/postgresql:2 .`

RUN DBs:

`docker run -i -t -p 5432:5432 jfreek/postgresql:1`

`docker run -i -t -p 5433:5432 jfreek/postgresql:2`


2.- Run Pipeline:

Build and run Container for our Production code:

`cd prod`

Build:
`docker build -t mlpipeline:1 .`

Run:
`docker run --rm --network=host -it mlpipeline:1 /bin/bash -c "python pipeline.py"`

I used prints to show each step of the pipeline, in the real world you may want to use logs.

3-. Check results

Open another 2 terminals and connect to the DBs and check for the content of the tables created.

* Check Dataset we used for training:

`psql -h localhost -p 5432 -U pguser -W pgdb`

password: `pguser`

cmd:
`select * from titanic;`


To check the Table with predictions:

`psql -h localhost -p 5433 -U pguser -W results`

password: `pguser`

cmd:
`select * from predictions;`

And that is it!