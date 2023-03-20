DROP DATABASE signTranslation;

CREATE DATABASE signTranslation;
use signTranslation;

CREATE TABLE ENVIRONMENT (
    type varchar(8) NOT NULL,
    PRIMARY KEY (type)
);

INSERT INTO ENVIRONMENT (type) VALUES ("train");
INSERT INTO ENVIRONMENT (type) VALUES ("dev");
INSERT INTO ENVIRONMENT (type) VALUES ("test");

CREATE TABLE AUTHOR (
    authorId int NOT NULL AUTO_INCREMENT,
    name varchar(32) UNIQUE NOT NULL,
    PRIMARY KEY (authorId)
)

INSERT INTO AUTHOR (name) VALUES ("Les Frères Grimm");

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

INSERT INTO STORY (file_name) VALUES ("003_marie");
INSERT INTO STORY (file_name) VALUES ("004_fear");
INSERT INTO STORY (file_name) VALUES ("005_wolf_and_lambs");
INSERT INTO STORY (file_name) VALUES ("006_thrusty_john");
INSERT INTO STORY (file_name) VALUES ("008_wonderfull_musicien");
INSERT INTO STORY (file_name) VALUES ("009_12_brothers");
INSERT INTO STORY (file_name) VALUES ("010_ragamuffins");
INSERT INTO STORY (file_name) VALUES ("011_brother_and_sister");
INSERT INTO STORY (file_name) VALUES ("012_rapunzel");
INSERT INTO STORY (file_name) VALUES ("013_three_little_men");
INSERT INTO STORY (file_name) VALUES ("014_tree_spinners");
INSERT INTO STORY (file_name) VALUES ("015_hansel_and_gretel");
INSERT INTO STORY (file_name) VALUES ("016_three_snakes_leaves");
INSERT INTO STORY (file_name) VALUES ("017_white_snakes");
INSERT INTO STORY (file_name) VALUES ("021_cinderella");
INSERT INTO STORY (file_name) VALUES ("026_red_hood");

CREATE TABLE PARALLEL_ITEM (
   itemId int NOT NULL AUTO_INCREMENT,
   story_name varchar(48),
   FR MEDIUMTEXT NOT NULL,
   GLOSS_LSF varchar(512),
   GENERATED_LSF varchar(512),
   TENSE varchar(32),
   GLOSS_LSFB varchar(512),
   EN MEDIUMTEXT,
   env_type varchar(8) NOT NULL,
   PRIMARY KEY (itemId),
   FOREIGN KEY (story_name) REFERENCES STORY(file_name),
   FOREIGN KEY (env_type) REFERENCES ENVIRONMENT(type)
);
