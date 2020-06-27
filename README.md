# noisy_student

1. To run in aws - Create an Ubuntu EC2 instance
You can use the following machine Ubuntu Server 16.04 LTS (HVM), SSD Volume Type - ami-0f2ed58082cb08a4d (64-bit x86)
Make sure the volume available to the EC2 instance is greater 150GB.

2. Run the following commands to install Docker - taken from [docker installation on ubuntu guide](https://docs.docker.com/engine/install/ubuntu/)
```sh
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) \
  stable"
sudo apt-get update
sudo apt-get install docker-ce docker-compose
```

3. check Client docker engine version >= 19.03 using the following command:
`docker version`

4. On Ubuntu 18.04 or 16.04 machines with Nvidia GPUs, run the following commands - taken from [nvidia-docker](https://github.com/NVIDIA/nvidia-docker])
```sh
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-cuda-toolkit
sudo systemctl restart docker
```

5. Make sure that the following command is running:
`nvidia-smi`
If not, try to reboot the machine using the reboot command:
`reboot`

6. pull the docker files code:
`git clone --branch noisy-student-flow https://github.com/MatanAvitan/noisy_student.git`

7. Optional: run the following command in tmux - run in shell:
`tmux`

8. cd to the noisy_student directory:
`cd noisy_student`

9. Create your AWS Credentials (AWS_ACCESS_ID and AWS_ACCESS_ID) - create them as local environment variables on your instance.

10. Build the docker file:
`sudo docker-compose build --build-arg AWS_ACCESS_ID=$AWS_ACCESS_ID --build-arg AWS_ACCESS_KEY=$AWS_ACCESS_KEY`

11. Create an S3 bucket for results, add the aws config file or credentials file to your env (see configuration guide [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#guide-configuration)) - these two are needed to run the algorithm.

12. Run the docker file:
`sudo docker run --gpus all \
                 --shm-size=100gb \
                 --name noisystudent \
                 --env MOCK_RUN=0 \
                 --env MOCK_ONE_MODEL=0 \
                 --env NUM_TRAIN_EPOCHS=151 \
                 --env ANNOTATIONS_SCORE_THRESH=0.6 \
                 --env S3_BUCKET_NAME=<your bucket name> \
                 --env EXPERIMENT_NAME=<your experiment name> \
                 -p 6006:6006 \
                bestteam/noisystudent:latest`

Note: we did not use docker-compose in this stage since docker compose does not suppor NVIDIA GPUs yet - see [the following issue](https://github.com/docker/compose/issues/6691)

13. For Mock-Run use the following command:
`sudo docker run --gpus all \
                 --shm-size=100gb \
                 --name noisystudent \
                 --env MOCK_RUN=1 \
                 --env MOCK_ONE_MODEL=0 \
                 --env S3_BUCKET_NAME=<your bucket name> \
                 --env EXPERIMENT_NAME=<your experiment name> \
                 -p 6006:6006 \
                 bestteam/noisystudent:latest`

14. If you want to mock the creation of once model only, change --env MOCK_ONE_MODEL=1 in the run command (change 1 instead of 0)

15. To find the trained models or logs in the docker - start the noisystudent container, and check the outputs directory:
```sh
sudo docker start noisystudent
sudo docker exec -it noisystudent bash
```
and run inside docker:
`ls outputs`

(To exit the container use ctrl-D)

16. To stop the noisystudent container run:
`sudo docker stop noisystudent`

17. To clear the containers run:
`sudo docker container rm $(sudo docker container ls -aq)`

18. To pull code from git, clear containers, build noisy student and run noisy student:
`git pull && sudo docker container rm $(sudo docker container ls -aq) && sudo docker-compose build --build-arg AWS_ACCESS_ID=$AWS_ACCESS_ID --build-arg AWS_ACCESS_KEY=$AWS_ACCESS_KEY && <your docker run command - see sections 12 and 13>`

19. To run TensorBoard while the container is running:
Make sure that in your run command you've added `-p 6006:6006`
Now run the following:
`sudo docker exec -it noisystudent bash`
And inside docker shell run:
`tensorboard --logdir src/openpifpaf/tb_logs/ --bind_all`
And in your local machine run the following to forward the port: `ssh -i <your aws key> -NL 6006:localhost:6006 <your aws instance>`
