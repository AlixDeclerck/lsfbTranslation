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

