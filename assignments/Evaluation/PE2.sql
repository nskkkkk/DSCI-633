
CREATE DATABASE IF NOT EXISTS PE2;

USE PE2;

CREATE TABLE IF NOT EXISTS BOOK (
    title            VARCHAR(50),
    isbn13Number     CHAR(13)  NOT NULL AUTO_INCREMENT PRIMARY KEY,
    author           VARCHAR(255),
    numberOfPages    INT,
    releaseDate      DATE
);


