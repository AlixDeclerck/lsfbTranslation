DROP DATABASE signTranslation;

CREATE DATABASE signTranslation;
use signTranslation;

-- SEVERAL ELEMENTS (FOLDER, ENUM) ARE USING THE TEST, TRAINING AND VALIDATION NAMES
-- ID SHOULD NOT BE CHANGED
CREATE TABLE ENVIRONMENT (
    type varchar(8) NOT NULL,
    PRIMARY KEY (type)
);

INSERT INTO ENVIRONMENT (type) VALUES ("train");
INSERT INTO ENVIRONMENT (type) VALUES ("test");
INSERT INTO ENVIRONMENT (type) VALUES ("val");

-- AUTHOR GIVE A (SIGNING) STYLE
-- AUTHOR NAME CAN BE CHANGED
CREATE TABLE AUTHOR (
    authorId int NOT NULL AUTO_INCREMENT,
    name varchar(32) UNIQUE NOT NULL,
    PRIMARY KEY (authorId)
)

INSERT INTO AUTHOR (name) VALUES ("Les Frères Grimm");

-- STORIES
-- FILE NAME (ID) SHOULD NOT BE CHANGED (because of the files themselves)
CREATE TABLE STORY (
    file_name varchar(48) NOT NULL,
    title varchar(128),
    id_author int,
    PRIMARY KEY (file_name),
    FOREIGN KEY (id_author) REFERENCES AUTHOR(authorId)
);

INSERT INTO STORY (file_name, title, id_author) VALUES (
                                                "001_prince_frog",
                                                "Le Roi-grenouille ou Henri-le-Ferré",
                                                1
                                            );

INSERT INTO STORY (file_name, title, id_author) VALUES (
                                                "002_cat_and_mouse",
                                                "Chat et souris associés",
                                                1
                                            );

INSERT INTO STORY (file_name, id_author) VALUES ("003_marie",1);
INSERT INTO STORY (file_name, id_author) VALUES ("004_fear",1);
INSERT INTO STORY (file_name, id_author) VALUES ("005_wolf_and_lambs",1);
INSERT INTO STORY (file_name, id_author) VALUES ("006_thrusty_john",1);
INSERT INTO STORY (file_name, id_author) VALUES ("008_wonderfull_musicien",1);
INSERT INTO STORY (file_name, id_author) VALUES ("009_12_brothers",1);
INSERT INTO STORY (file_name, id_author) VALUES ("010_ragamuffins",1);
INSERT INTO STORY (file_name, id_author) VALUES ("011_brother_and_sister",1);
INSERT INTO STORY (file_name, id_author) VALUES ("012_rapunzel",1);
INSERT INTO STORY (file_name, id_author) VALUES ("013_three_little_men",1);
INSERT INTO STORY (file_name, id_author) VALUES ("014_tree_spinners",1);
INSERT INTO STORY (file_name, id_author) VALUES ("015_hansel_and_gretel",1);
INSERT INTO STORY (file_name, id_author) VALUES ("016_three_snakes_leaves",1);
INSERT INTO STORY (file_name, id_author) VALUES ("017_white_snakes",1);
INSERT INTO STORY (file_name, id_author) VALUES ("021_cinderella",1);
INSERT INTO STORY (file_name, id_author) VALUES ("026_red_hood",1);

-- PARALLELS
CREATE TABLE PARALLEL_ITEM (
   itemId int NOT NULL AUTO_INCREMENT,
   story_name varchar(48),
   FR MEDIUMTEXT NOT NULL,
   GLOSS_LSF varchar(512),
   GENERATED_LSF varchar(512),
   TENSE varchar(128),
   GLOSS_LSFB varchar(512),
   EN MEDIUMTEXT,
   env_type varchar(8) NOT NULL,
   PRIMARY KEY (itemId),
   FOREIGN KEY (story_name) REFERENCES STORY(file_name),
   FOREIGN KEY (env_type) REFERENCES ENVIRONMENT(type)
);

-- OTHER REQUESTS
-- DROP TABLE PARALLEL_ITEM
-- DELETE FROM PARALLEL_ITEM WHERE story_name = "001_prince_frog"
-- SELECT * FROM PARALLEL_ITEM
-- ALTER TABLE PARALLEL_ITEM MODIFY COLUMN TENSE VARCHAR(128);
