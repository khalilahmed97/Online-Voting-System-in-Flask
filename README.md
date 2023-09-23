# PROJECT SCREENSHOTS

![display_img](https://user-images.githubusercontent.com/101020879/213549690-b88d9cc4-0b45-45ad-8f4e-0d994eb54119.png)


![image](https://user-images.githubusercontent.com/101020879/213549755-09df36c4-5bfa-44ea-b0f2-2fd27e29ae34.png)


![image](https://user-images.githubusercontent.com/101020879/213549799-b9809847-9a86-4d2e-a86b-296d968155d5.png)


![image](https://user-images.githubusercontent.com/101020879/213550252-3d46f238-17b4-4464-a44e-7e309901dd48.png)


![image](https://user-images.githubusercontent.com/101020879/213550487-12a85107-da5f-43aa-b790-7ce954d01086.png)

# STEPS

In order to run the system you have to follow these steps:
1)- After downloading all the libraries Run the command (python train.py) to train the chatbot data
2) Run the command (python app.py) to run the server

## DATABASE

DATABASE NAME = online_voting_system

NUMBER OF TABLES IN A DATABASE = 3

NAME OF THE TABLES:
1)- voters
2) candidates
3) votes

## DATABASE QUERIES FOR THE CREATION OF THE TABLES

1) CREATE TABLE voters(ID INT PRIMARY KEY auto_increment, pic varchar(1024), name varchar(50), cnic varchar(50), age INT, password varchar(50), voted varchar(50));

2) CREATE TABLE candidates(ID INT PRIMARY KEY auto_increment, pic varchar(1024), name varchar(50), cnic varchar(50), age INT, password varchar(50), party_symbol varchar(50));

3) CREATE TABLE votes(ID INT PRIMARY KEY auto_increment, pic varchar(1024), candidate_name varchar(50), party_symbol varchar(50), number_of_votes INT);
