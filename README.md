In order to run the system you have follow these steps:
1)- After downloading all the libraries Run the command (python train.py) to train the chatbot data
2) Run the command (python app.py) to run the server

------------------------ DATABASE ------------------------

DATABASE NAME = online_voting_system
NUMBER OF TABLES IN A DATABASE = 3
NAME OF THE TABLES = 1)- voters, 2) candidates, 3) votes

------------------------ DATABASE QUERIES FOR THE CREATION OF THE TABLES ----------------------

1) CREATE TABLE voters(ID INT PRIMARY KEY auto_increment, pic varchar(1024), name varchar(50), cnic varchar(50), age INT, password varchar(50), voted varchar(50));

2) CREATE TABLE candidates(ID INT PRIMARY KEY auto_increment, pic varchar(1024), name varchar(50), cnic varchar(50), age INT, password varchar(50), party_symbol varchar(50));

3) CREATE TABLE votes(ID INT PRIMARY KEY auto_increment, pic varchar(1024), candidate_name varchar(50), party_symbol varchar(50), number_of_votes INT);
