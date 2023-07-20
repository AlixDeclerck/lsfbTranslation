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

INSERT INTO STORY (file_name, title, id_author) VALUES ("001_prince_frog","Le Roi-grenouille ou Henri-le-Ferré (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("002_cat_and_mouse","Chat et souris associés (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("003_marie","L'enfant de Marie (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("004_fear","Conte de celui qui partit pour apprendre la peur (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("005_wolf_and_lambs","Le loup et les sept chevreaux (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("006_thrusty_john","Le fidèle Jean (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("007_good_bargain","The good bargain",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("008_wonderfull_musicien","Le violon merveilleux (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("009_12_brothers","Les douze frères (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("010_ragamuffins","Les vagabonds (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("011_brother_and_sister","Frérot et soeurette (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("012_rapunzel","Raiponce (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("013_three_little_men","Les trois petits hommes de la forêt (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("014_tree_spinners","Les trois fileuses",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("015_hansel_and_gretel","Hansel et Gretel (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("016_three_snakes_leaves","Les trois feuilles du serpent (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("017_white_snakes","Le serpent blanc (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("021_cinderella","Cendrillon (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("026_red_hood","Le petit Chaperon Rouge (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("027_bremen_town_musicians","The Bremen town musicians",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("028_singing_bones","The singing bone",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("029_devil_with_golden_hairs","The devil with the three golden hairs",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("030_louse_and_flea","The louse and the flea",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("031_girl_whithout_hand","The girl without hands",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("032_clever_hans","Clever Hans",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("033_three_languages","The three languages",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("034_clever_else","Clever Else",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("035_tailor_in_heaven","The tailor in heaven",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("036_wishing_table","The Wishing-Table, the Gold-Ass, and the Cudgel in the Sack",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("037_tom_thumb","Tom Thumb",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("038_wedding_ms_fox","The wedding of Mrs. Fox",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("039_the_elves","The elves",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("040_robber","The robber bridegroom",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("041_herr_korbes","Herr Korbes",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("042_godfather","The godfather",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("043_frau_trude","Frau Trude",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("044_godfather_death","Godfather Death",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("045_thrumblings_travel","Thumbling's Travels",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("046_fitchers_bird","Fitcher's Bird",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("047_almond_tree","The almond tree",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("048_old_sultan","Old Sultan",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("049_six_swans","The six swans",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("050_sleeping_beauty","La belle au bois dormant (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("051_fundevogel","Bird-foundling",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("052_king_thursbeard","King Thrushbeard",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("053_snow_white","Blanche-Neige (LSF)",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("054_knapsack_hat_horn","The knapsack, the hat, and the horn",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("055_rumpelstilskin","Nain Tracassin",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("060_two_brothers","Les deux frangins",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("063_three_feathers","Les trois plumes",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("064_golden_goose","L'oie d'or",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("066_rabbits_bride","La fiancée du petit lapin",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("069_jorinda_and_joringel","Jorinde et Joringel",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("070_three_children_of_fortune","Les trois enfants gâtés de la fortune",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("071_six_soldiers_of_fortune","Les six compagnons qui viennent à bout de tout",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("072_the_wolf_and_the_man","Le loup et l'homme",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("073_the_wolf_and_the_fox","Le loup et le renard",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("078_the_old_man_and_his_grandson","Le vieux grand-père et son petit-fils",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("079_the_water_nix","L'ondine",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("148_lord_animal_devil","The Lord's animals and the Devil's",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("149_beam","The beam",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("150_old_begar_woman","The old beggar-woman",1);
INSERT INTO STORY (file_name, title, id_author) VALUES ("151_three_sluggards","The three sluggards",1);

-- SEVERAL ELEMENTS (FOLDER, ENUM) ARE USING THE TEST, TRAINING AND VALIDATION NAMES
-- ID SHOULD NOT BE CHANGED
CREATE TABLE ENVIRONMENT (type varchar(8) NOT NULL, PRIMARY KEY (type));
INSERT INTO ENVIRONMENT (type) VALUES ("train");
INSERT INTO ENVIRONMENT (type) VALUES ("test");

CREATE TABLE LANG (name varchar(16) NOT NULL, PRIMARY KEY (name));
INSERT INTO LANG (name) VALUES ("FR");
INSERT INTO LANG (name) VALUES ("EN");
INSERT INTO LANG (name) VALUES ("GLOSS_LSF");

-- translations (8590 phrases inserted),
-- a text and a generated text for each LANG.name
CREATE TABLE TRANSLATION (
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
-- DROP TABLE TRANSLATION
-- DELETE FROM TRANSLATION WHERE story_name = "001_prince_frog"
-- ALTER TABLE TRANSLATION MODIFY COLUMN TENSE VARCHAR(128);
-- select  p2.txt as txt_fr, p2.txt_generated as fr_generated,
--         p1.txt as txt_en, p1.txt_generated as en_generated,
--         p3.txt as txt_gloss, p3.txt_generated as gloss_generated,
--         p1.num as num, p1.story_name as story_name
-- from TRANSLATION p1
--          inner join TRANSLATION p2 on p1.num = p2.num and p1.story_name = p2.story_name
--          inner join TRANSLATION p3 on p2.num = p3.num and p2.story_name = p3.story_name
-- where p1.lang = 'EN' and p2.lang = 'FR' and p3.lang = 'GLOSS_LSF' and p1.env_type = 'TRAIN'
