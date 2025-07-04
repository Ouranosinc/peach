# Backend storage

Data used for backend computations and their results are stored on a MinIO server running on PAVICS. MinIO is a storage server that provides a scalable, high-performance object storage system. It is API-compatible with Amazon S3 cloud storage service.

% The admin panel of the MinIO server can be found at http://notos.ouranos.ca:9401

MinIO organizes data in buckets. Each bucket is a top-level namespace for objects. The project's  bucket is called ``portail-ing``. Each bucket may have its own access rules and policies (access rights, maximum size, versioning, etc.)

% Use the admin panel to manage buckets and grant read-write permissions.

## Linux client

% The client is already installed on doris (``module load minio``).

The easiest way to read and write on a MinIO instance is through the `mc` command-line client. To configure the client, run the following command:

```bash
mc alias set pavics https://minio.ouranos.ca
```

providing your access key and secret key when prompted. Ask your admin for these keys. Note that read-only access is possible, just press Enter Enter.

Then you can list the buckets with `mc ls`:

```bash
mc ls pavics/portail-ing
```


To copy files to the bucket using, assuming you have write access, use `mc cp`:

```bash
mc cp file.txt pavics/portail-ing/file_copy.txt
```
Add `-r` flag for recursive copy within a directory.

## Programmatic access for developers

The MinIO server can be accessed programmatically in Python using ``s3fs``. Here is an example of how to write and read from a bucket.

```python

def write_to_minio(local_path="data.nc", root="test/data.zarr"):
    """Open netCDF dataset from local path and write to MinIO as a Zarr object."""
    import s3fs
    import xarray as xr

    # Open connection to MinIO server
    ACCESS_KEY = "<YOUR_API_KEY"
    SECRET_KEY = "<YOUR_SECRET_KEY>"

    s3 = s3fs.S3FileSystem(anon=False, key=ACCESS_KEY, secret=SECRET_KEY, use_ssl=False,
                           client_kwargs={"endpoint_url": "http://minio.ouranos.ca"})

    # Create store from bucket name / object name
    store = s3fs.S3Map(root=root, s3=s3, check=False)

    # Open local dataset
    ds = xr.open_dataset(local_path).chunk()

    # Write to MinIO
    ds.to_zarr(store=store, mode="w", consolidated=True)

def read_from_minio(root="test/data.zarr"):
    """Read Zarr object from MinIO and return as xarray dataset."""
    import s3fs
    import xarray as xr

    s3r = s3fs.S3FileSystem(anon=True, use_ssl=False, client_kwargs={"endpoint_url": "http://minio.ouranos.ca"})
    store = s3fs.S3Map(root=root, s3=s3r, check=False)
    return xr.open_zarr(store=store)
```
