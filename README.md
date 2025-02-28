### Runpod deploy

#### Docker build

```sh
docker build -t runpod_flux .
docker tag runpod_flux docker_name
docker push docker_name
```
#### Deploy
- create template in runpod
- create pod
