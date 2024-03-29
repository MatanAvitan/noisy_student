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
`git clone --branch master https://github.com/MatanAvitan/noisy_student.git`

7. Optional: run the following command in tmux - run in shell:
`tmux`

8. cd to the noisy_student directory:
`cd noisy_student`

9. Create your AWS Credentials (AWS_ACCESS_ID and AWS_ACCESS_KEY) - create them as local environment variables on your instance.

10. There are 2 options for running the noisy_student algorithm: either docker or shell script.

**Using Docker - see sections 10 to 19:**

10. Build the docker file:
`sudo docker-compose build --build-arg AWS_ACCESS_ID=$AWS_ACCESS_ID --build-arg AWS_ACCESS_KEY=$AWS_ACCESS_KEY`

11. Create an S3 bucket for results, add the aws config file or credentials file to your env (see configuration guide [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#guide-configuration)) - these two are needed to run the algorithm.

12. Run the docker file:
`sudo docker run --gpus all \
                 --shm-size=100gb \
                 --name noisystudent \
                 --env MOCK_RUN=FALSE \
                 --env MOCK_ONE_MODEL=FALSE \
                 --env RUN_FULL_MODEL=FALSE \
                 --env NUM_TRAIN_EPOCHS=140 \
                 --env ANNOTATIONS_SCORE_INITIAL_THRESH=0.65 \
                 --env STUDENT_TEACHER_LOOP=15 \
                 --env S3_BUCKET_NAME=<your bucket name> \
                 --env EXPERIMENT_NAME=<your experiment name> \
                 -p 6006:6006 \
                bestteam/noisystudent:latest`

Note: we did not use docker-compose in this stage since docker compose does not suppor NVIDIA GPUs yet - see [the following issue](https://github.com/docker/compose/issues/6691)

13. To run the docker from starting a previous model as a teacher:
`sudo docker run --gpus all \
                 --shm-size=100gb \
                 --name noisystudent \
                 --env MOCK_RUN=FALSE \
                 --env MOCK_ONE_MODEL=FALSE \
                 --env RUN_FULL_MODEL=FALSE \
                 --env NUM_TRAIN_EPOCHS=140 \
                 --env ANNOTATIONS_SCORE_INITIAL_THRESH=0.4 \
                 --env INITIAL_MODEL=<model_name> \
                 --env INITIAL_MODEL_BUCKET=<bucket_name_of_initial_model> \
                 --env BLUR_MAX_SIGMA=<blur_max_sigma, for example 3> \
                 --env STUDENT_TEACHER_LOOP=15 \
                 --env S3_BUCKET_NAME=<your bucket name> \
                 --env EXPERIMENT_NAME=<your experiment name> \
                 -p 6006:6006 \
                bestteam/noisystudent:latest`

14. For Mock-Run use the following command:
`sudo docker run --gpus all \
                 --shm-size=100gb \
                 --name noisystudent \
                 --env MOCK_RUN=TRUE \
                 --env MOCK_ONE_MODEL=FALSE \
                 --env RUN_FULL_MODEL=FALSE \
                 --env STUDENT_TEACHER_LOOP=2 \
                 --env S3_BUCKET_NAME=<your bucket name> \
                 --env EXPERIMENT_NAME=<your experiment name> \
                 -p 6006:6006 \
                 bestteam/noisystudent:latest`

15. If you want to mock the creation of once model only, change --env MOCK_ONE_MODEL=TRUE in the run command

16. To find the trained models or logs in the docker - start the noisystudent container, and check the outputs directory:
```sh
sudo docker start noisystudent
sudo docker exec -it noisystudent bash
```
and run inside docker:
`ls outputs`

(To exit the container use ctrl-D)

17. To stop the noisystudent container run:
`sudo docker stop noisystudent`

18. To clear the containers run:
`sudo docker container rm $(sudo docker container ls -aq)`

19. To pull code from git, clear containers, build noisy student and run noisy student:
`git pull origin master && sudo docker container rm $(sudo docker container ls -aq) && sudo docker-compose build --build-arg AWS_ACCESS_ID=$AWS_ACCESS_ID --build-arg AWS_ACCESS_KEY=$AWS_ACCESS_KEY && <your docker run command - see sections 12 and 13>`

20. To run TensorBoard while the container is running:
Make sure that in your run command you've added `-p 6006:6006`
Now run the following:
`sudo docker exec -it noisystudent bash`
And inside docker shell run:
`tensorboard --logdir src/openpifpaf/tb_logs/ --bind_all`
And in your local machine run the following to forward the port: `ssh -i <your aws key> -NL 6006:localhost:6006 <your aws instance>`

**Using Shell scripts - see sections 20 to 22:**

21. Create additional environment variables, depending on your run configuration, for example for a full mock run (including creating images):
`export S3_BUCKET_NAME=<your bucket name> ; export EXPERIMENT_NAME=<your experiment name> ; export MOCK_RUN=TRUE ; export MOCK_ONE_MODEL=FALSE;`
And for a full run :
`export S3_BUCKET_NAME=<your bucket name> ; export EXPERIMENT_NAME=<your experiment name> ; export MOCK_RUN=FALSE ; export MOCK_ONE_MODEL=FALSE; export ANNOTATIONS_SCORE_INITIAL_THRESH=0.65 ; export NUM_TRAIN_EPOCHS=140`

22. run: `./install_collect_data.sh`

23. run `./run_noisy_student.sh`
