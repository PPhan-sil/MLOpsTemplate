# MLOps Template

- Rename `.env_example` or create your own `.env`

## Commmands
- Build: `docker compose up --build`
- Spin up: `docker compose up -d`
- Run training code: `docker exec -it ml_template_dev python train.py`
- Run evaluation code: `docker exec -it ml_template_dev python eval.py`
- Run script: `docker exec -it ml_template_dev scripts/run.sh`
