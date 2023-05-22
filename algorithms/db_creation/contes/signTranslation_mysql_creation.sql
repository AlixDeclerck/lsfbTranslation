DROP DATABASE signTranslation;

CREATE DATABASE signTranslation;
use signTranslation;

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

INSERT INTO STORY (file_name, title, id_author) VALUES ("001_prince_frog","Le Roi-grenouille ou Henri-le-Ferré",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("002_cat_and_mouse","Chat et souris associés",1);
INSERT INTO STORY (file_name, id_author) VALUES ("003_marie",1);
INSERT INTO STORY (file_name, id_author) VALUES ("004_fear",1);
INSERT INTO STORY (file_name, id_author) VALUES ("005_wolf_and_lambs",1);
INSERT INTO STORY (file_name, id_author) VALUES ("006_thrusty_john",1);
INSERT INTO STORY (file_name, id_author) VALUES ("007_good_bargain",1);
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
INSERT INTO STORY (file_name, id_author) VALUES ("027_bremen_town_musicians",1);
INSERT INTO STORY (file_name, id_author) VALUES ("028_singing_bones",1);
INSERT INTO STORY (file_name, id_author) VALUES ("029_devil_with_golden_hairs",1);
INSERT INTO STORY (file_name, id_author) VALUES ("030_louse_and_flea",1);
INSERT INTO STORY (file_name, id_author) VALUES ("031_girl_whithout_hand",1);
INSERT INTO STORY (file_name, id_author) VALUES ("032_clever_hans",1);
INSERT INTO STORY (file_name, id_author) VALUES ("033_three_languages",1);
INSERT INTO STORY (file_name, id_author) VALUES ("034_clever_else",1);
INSERT INTO STORY (file_name, id_author) VALUES ("035_tailor_in_heaven",1);
INSERT INTO STORY (file_name, id_author) VALUES ("036_wishing_table",1);
INSERT INTO STORY (file_name, id_author) VALUES ("037_tom_thumb",1);
INSERT INTO STORY (file_name, id_author) VALUES ("038_wedding_ms_fox",1);
INSERT INTO STORY (file_name, id_author) VALUES ("039_the_elves",1);
INSERT INTO STORY (file_name, id_author) VALUES ("040_robber",1);
INSERT INTO STORY (file_name, id_author) VALUES ("050_sleeping_beauty",1);
INSERT INTO STORY (file_name, id_author) VALUES ("053_snow_white",1);
INSERT INTO STORY (file_name, id_author) VALUES ("055_rumpelstilskin",1);
INSERT INTO STORY (file_name, id_author) VALUES ("060_two_brothers",1);
INSERT INTO STORY (file_name, id_author) VALUES ("063_three_feathers",1);
INSERT INTO STORY (file_name, id_author) VALUES ("064_golden_goose",1);
INSERT INTO STORY (file_name, id_author) VALUES ("066_rabbits_bride",1);
INSERT INTO STORY (file_name, id_author) VALUES ("069_jorinda_and_joringel",1);
INSERT INTO STORY (file_name, id_author) VALUES ("070_three_children_of_fortune",1);
INSERT INTO STORY (file_name, id_author) VALUES ("071_six_soldiers_of_fortune",1);
INSERT INTO STORY (file_name, id_author) VALUES ("072_the_wolf_and_the_man",1);
INSERT INTO STORY (file_name, id_author) VALUES ("073_the_wolf_and_the_fox",1);
INSERT INTO STORY (file_name, id_author) VALUES ("078_the_old_man_and_his_grandson",1);
INSERT INTO STORY (file_name, id_author) VALUES ("079_the_water_nix",1);

-- SEVERAL ELEMENTS (FOLDER, ENUM) ARE USING THE TEST, TRAINING AND VALIDATION NAMES
-- ID SHOULD NOT BE CHANGED
CREATE TABLE ENVIRONMENT (type varchar(8) NOT NULL, PRIMARY KEY (type));
INSERT INTO ENVIRONMENT (type) VALUES ("train");
INSERT INTO ENVIRONMENT (type) VALUES ("test");

CREATE TABLE LANG (name varchar(16) NOT NULL, PRIMARY KEY (name));
INSERT INTO LANG (name) VALUES ("FR");
INSERT INTO LANG (name) VALUES ("EN");
INSERT INTO LANG (name) VALUES ("GLOSS_LSF");

-- PARALLELS are generalized,
-- a text and a generated text for each LANG.name
CREATE TABLE PARALLEL_ITEM (
   story_name varchar(48) NOT NULL,
   lang varchar(32) NOT NULL,
   txt MEDIUMTEXT NOT NULL,
   txt_generated varchar(512),
   txt_generation_date DATE,
   tense varchar(128),
   num int NOT NULL,
   score varchar(64),
   env_type varchar(8) NOT NULL,
   PRIMARY KEY (story_name, num, lang),
   FOREIGN KEY (story_name) REFERENCES STORY(file_name),
   FOREIGN KEY (lang) REFERENCES LANG(name),
   FOREIGN KEY (env_type) REFERENCES ENVIRONMENT(type)
);

-- USED REQUESTS
-- DROP TABLE PARALLEL_ITEM
-- DELETE FROM PARALLEL_ITEM WHERE story_name = "001_prince_frog"
-- ALTER TABLE PARALLEL_ITEM MODIFY COLUMN TENSE VARCHAR(128);
-- select  p2.txt as txt_fr, p2.txt_generated as fr_generated,
--         p1.txt as txt_en, p1.txt_generated as en_generated,
--         p3.txt as txt_gloss, p3.txt_generated as gloss_generated,
--         p1.num as num, p1.story_name as story_name
-- from PARALLEL_ITEM p1
--          inner join PARALLEL_ITEM p2 on p1.num = p2.num and p1.story_name = p2.story_name
--          inner join PARALLEL_ITEM p3 on p2.num = p3.num and p2.story_name = p3.story_name
-- where p1.lang = 'EN' and p2.lang = 'FR' and p3.lang = 'GLOSS_LSF' and p1.env_type = 'TRAIN'
