# Overview
This repo contains two things: data collection program, and source files for a website to display the collected data.

The program is meant to track line statistics of to-go restaurants through a live video feed processed via AI vision with the YOLO AI vision library. The program assumes you are running it on a computer with Python and other project requirements installed, that the computer is receiving the video feed through a some connection, and that the computer is connected to the internet. If the computer has a monitor, the video feed with overlays conveying what the system is track is displayed. Our implementation was done on MacBooks, and while the program should theoretically work regardless of operating system, some parts of the program *may* need to be altered to work on other operating systems.

Project requirements:
- Python 3
- Libraries in requirements.txt
- AWS CLI

To set up a local copy of the project for the first time...
- Make sure Python and the AWS CLI are installed on your machine
- In a terminal, navigate to the project root
- Set up the Python virtual environment with `python3 -m venv .venv`
- Install the project requirements with `pip install -r requirements.txt`
  - if this fails, try updating pip with `pip install –upgrade pip`

The website is used to display the estimated wait times and line lengths for all the stores you are monitoring with this program. This project assumes the data for the website is being retrieved from an S3 bucket. This repo has a GitHub actions workflow that syncs the website source files with the S3 bucket upon pushes to main. The workflow assumes the access key and secret access key for an IAM user/role with the proper permissions are stored in the repo as repository secrets.


# Video Processing Program
The video processing program uses the AI vision library called YOLO to track the actions of each customer in the store. When a customer enters a region predefined as the line, the system notes down a timestamp, increments the number of people in line, and continues tracking them. When a customer who was in the line picks up their food (defined as their wrists being near the food within the predefined pickup zone), the system calculates how much time has passed since they entered the line, decrements the line count, and stops tracking them. The system takes the average of all wait time durations within the past five minutes to produce an estimated wait time. When setting up the system for yourself, you must define the line and pickup zones within the system to match your setup, and you must train the AI model to recognize the food of the given store by giving it 100-1000 pictures as training data.

To run the video processing program...
- set the VIDEO_SOURCE in config.py to be the correct video source for your camera setup
  - If you are running on a MacBook with an iPhone associated with it, set the video source to 0 to use the iPhone as the camera
- If you haven't run the program on this machine before...
  - Run `AWS configure` and enter the access key and secret access key for an IAM user/role with the proper permissions, set the default region to be `us-east-2` and set the default output format to `json`
  - Set the S3 URI in the config file to match that of the S3 bucket you will be uploading to
- Open a terminal and run `source .venv/bin/activate` to make all Python commands use this projects Python installation for the duration of the terminal session
- Then run `python3 -m vision.worker` to start the program


# Website

If all the supporting systems (GitHub actions, AWS resources, etc.) hae been set up correctly, the website should simply work.

To demo the website on your local machine, change into the website directory and run `python3 -m http.server 8000`. Then, open your browser and go to http://localhost:8000. When you are done demoing the website, go back to the terminal and kill the process with ctrl + c (regardless of OS).