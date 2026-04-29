# Clinical Trial Matching

## Installation guide

### Install clinical trial data:

1. Download data dump from https://aact.ctti-clinicaltrials.org/downloads
2. In the Terminal, set up Docker container:

* Pull Postgres image: `docker pull postgres:16`
* Copy the dump downloaded earlier to `/tmp` in container: `docker cp path_to_dump/postgres.dmp ctdm-db:/tmp/postgres.dmp`
* Populate the data: `docker exec -it ctdm-db pg_restore -U postgres -e -v -O -x -d aact --no-owner /tmp/postgres.dmp`
* (If needed) Drop schema: `docker exec -it ctdm-db psql -U postgres -d aact -c "DROP SCHEMA ctgov CASCADE;"`

3. In pgAdmin (or other UI to view Postgres db):

* View the top 100 rows:

```
SELECT * FROM ctgov.eligibilities
ORDER BY id ASC LIMIT 100
```

* Connect with database:

```postgres://[username]:[password]@db:5432/aact```

* Store the URL as `DATABASE_URL` in `.env`

### Install Python Virtual Environment for dependencies

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Training

### Clinical Trials BIO Tagging

1. Train the model and view results:

```
cd clinical-trial-processor
python3 encoder.py
```

A new model will be saved in either `models`, directory `ncbi_disease` or `tner_bc5cdr` based on your default database (can set in `constants.py`), along with the losses and evaluation.

2. Update the database with extracted graph info:

```
python3 process.py
```

### Download scispaCy models
```
pip install scispacy
pip install en_core_sci_sm
```