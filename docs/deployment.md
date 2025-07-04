# Deployment

The portal is built on a backend and a frontend, both of which are dockerized. The instructions below will launch both components.

## Development Servers

In order to create development images, first copy `.env.example` to `.env` and adjust the settings as needed.

Run the following commands to create the docker images for the frontend and backend, start them, and follow the logs:

```bash
make build-images && make run-images && make follow-logs
```
Note that the frontend image will merge the environment.yml and environment-frontend.yml for its requirements. Similarly for the backend image with environment-backend.yml.

Both the backend and frontend have auto-reloading enabled in development mode, but it may take a few seconds for these to propagate. Notably, due to a bug in Gunicorn, the backend server will completely restart on any file change in the ``/src/`` folder.

## Production Servers

For a production server, you can start the images as follows:

```bash
docker compose up -d --build frontend backend && docker compose logs -f --tail 10
```

The differences between the production and development images are as follows:

- Auto-reloading is turned off.
- The local pip module is installed in non-editable mode. (``pip install --no-deps .`` instead of ``pip install --no-deps -e .``).

Otherwise, the images remain identical, and, unless specified otherwise in your .env file, will use the same ports by default.

## S3-like storage

The backend will write to a S3-like storage if the ``BUCKET_URL`` and ``BUCKET_NAME`` environment variables are given. A credentials file ``credentials.json`` must also be passed in order to have the backend.

It expects some sort of read-write policy looking like this:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetBucketLocation",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::portail-ing",
                "arn:aws:s3:::portail-ing/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:*"
            ],
            "Resource": [
                "arn:aws:s3:::portail-ing/workspace/*"
            ]
        }
    ]
}
```
