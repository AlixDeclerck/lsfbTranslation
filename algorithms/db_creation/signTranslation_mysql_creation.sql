DROP DATABASE signTranslation;

CREATE DATABASE signTranslation;
use signTranslation;

CREATE TABLE ENVIRONMENT (
                             envId int NOT NULL AUTO_INCREMENT,
                             type varchar(8) UNIQUE NOT NULL,
                             PRIMARY KEY (envId)
);

INSERT INTO ENVIRONMENT (type) VALUES ("train");
INSERT INTO ENVIRONMENT (type) VALUES ("dev");
INSERT INTO ENVIRONMENT (type) VALUES ("test");

CREATE TABLE PARALLEL_ITEM (
                               itemId int NOT NULL AUTO_INCREMENT,
                               env_type int NOT NULL,
                               file varchar(255),
                               gloss varchar(255) NOT NULL,
                               skels MEDIUMTEXT,
                               text varchar(1024) NOT NULL,
                               PRIMARY KEY (itemId),
                               FOREIGN KEY (env_type) REFERENCES ENVIRONMENT(envId)
);

GRANT Alter ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Create ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Create view ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Delete ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Drop ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Grant option ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Index ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Insert ON signTranslation.* TO 'myosotis'@'localhost';
GRANT References ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Select ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Show view ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Trigger ON signTranslation.* TO 'myosotis'@'localhost';
GRANT Update ON signTranslation.* TO 'myosotis'@'localhost' WITH GRANT OPTION;
