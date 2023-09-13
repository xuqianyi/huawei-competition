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
    picture_url VARCHAR(255),
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

INSERT INTO Questions(picture_url, correct_answer) VALUES 
('https://obs-dc4c.obs.ap-southeast-3.myhuaweicloud.com/ECGS/row_1.png', 1), 
('https://obs-dc4c.obs.ap-southeast-3.myhuaweicloud.com/ECGS/row_2.png', 10), 
('https://obs-dc4c.obs.ap-southeast-3.myhuaweicloud.com/ECGS/row_3.png', 12), 
('https://obs-dc4c.obs.ap-southeast-3.myhuaweicloud.com/ECGS/row_4.png', 38), 
('https://obs-dc4c.obs.ap-southeast-3.myhuaweicloud.com/ECGS/row_5.png', 12);
