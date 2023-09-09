CREATE SCHEMA HUAWEI;

USE HUAWEI;

CREATE TABLE Users (
	PRIMARY KEY (user_id),
    user_id SMALLINT(5) UNSIGNED AUTO_INCREMENT,
    user_name VARCHAR(20),
    user_password VARCHAR(20),
    premium_yn TINYINT(1) UNSIGNED
);

CREATE TABLE Questions (
	PRIMARY KEY (question_id),
    question_id SMALLINT(5) UNSIGNED AUTO_INCREMENT,
    file_name VARCHAR(255),
    correct_answer SMALLINT(2) UNSIGNED
);


CREATE TABLE Mistakes (
	PRIMARY KEY (mistake_id),
    mistake_id SMALLINT(5) UNSIGNED AUTO_INCREMENT,
    user_id SMALLINT(5) UNSIGNED,
    question_id SMALLINT(5) UNSIGNED
);

CREATE TABLE Favorites (
	PRIMARY KEY (favorite_id),
    favorite_id SMALLINT(5) UNSIGNED AUTO_INCREMENT,
    user_id SMALLINT(5) UNSIGNED,
    question_id SMALLINT(5) UNSIGNED
);

CREATE TABLE test (
	PRIMARY KEY (question_id),
    question_id SMALLINT(5) UNSIGNED AUTO_INCREMENT,
    correct_answer SMALLINT(2) UNSIGNED
);

INSERT INTO test (question_id, correct_answer) VALUES (1, 1), (2, 10), (3, 12), (4, 38), (5, 12);

        
