CREATE TABLE ENVIRONMENT (
                             envId int PRIMARY KEY NOT NULL,
                             type varchar(8) UNIQUE NOT NULL
)

INSERT INTO ENVIRONMENT (envId, type) VALUES (1, "dev");
INSERT INTO ENVIRONMENT (envId, type) VALUES (2, "test");
INSERT INTO ENVIRONMENT (envId, type) VALUES (3, "train");

CREATE TABLE PARALLEL_ITEM (
                               itemId int NOT NULL,
                               env_type int NOT NULL,
                               file varchar(255) NOT NULL,
                               gloss varchar(255) NOT NULL,
                               skels varchar(255),
                               text varchar(255) NOT NULL,
                               PRIMARY KEY (itemId),
                               FOREIGN KEY (env_type) REFERENCES ENVIRONMENT(envId)
);

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (1,
                                                                               "dev/11August_2010_Wednesday_tagesschau-2",
                                                                               "DRUCK TIEF KOMMEN",
                                                                               "coord1",
                                                                               "tiefer luftdruck bestimmt in den nächsten tagen unser wetter .",
                                                                               1
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (2,
                                                                               "dev/11August_2010_Wednesday_tagesschau-3",
                                                                               "ES-BEDEUTET VIEL WOLKE UND KOENNEN REGEN GEWITTER KOENNEN",
                                                                               "coord2",
                                                                               "das bedeutet viele wolken und immer wieder zum teil kräftige schauer und gewitter .",
                                                                               1
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (3,
                                                                               "dev/11August_2010_Wednesday_tagesschau-8",
                                                                               "WIND MAESSIG SCHWACH REGION WENN GEWITTER WIND KOENNEN",
                                                                               "coord3",
                                                                               "meist weht nur ein schwacher wind aus unterschiedlichen richtungen der bei schauern und gewittern stark böig sein kann .",
                                                                               1
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (4,
                                                                               "dev/25October_2010_Monday_tagesschau-22",
                                                                               "MITTWOCH REGEN KOENNEN NORDWEST WAHRSCHEINLICH NORD STARK WIND",
                                                                               "coord4",
                                                                               "am mittwoch hier und da nieselregen in der nordwesthälfte an den küsten kräftiger wind .",
                                                                               1
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (5,
                                                                               "dev/05May_2011_Thursday_tagesschau-25",
                                                                               "JETZT WETTER WIE-AUSSEHEN MORGEN FREITAG SECHSTE MAI ZEIGEN-BILDSCHIRM",
                                                                               "coord5",
                                                                               "und nun die wettervorhersage für morgen freitag den sechsten mai .",
                                                                               1
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (6,
                                                                               "test/25October_2010_Monday_tagesschau-17",
                                                                               "REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN SEHEN",
                                                                               "coord6",
                                                                               "regen und schnee lassen an den alpen in der nacht nach im norden und nordosten fallen hier und da schauer sonst ist das klar .",
                                                                               2
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (7,
                                                                               "test/25October_2010_Monday_tagesschau-24",
                                                                               "DONNERSTAG NORDWEST REGEN REGION SONNE WOLKE WECHSELHAFT DANN FREITAG AEHNLICH WETTER",
                                                                               "coord7",
                                                                               "am donnerstag regen in der nordhälfte in der südhälfte mal sonne mal wolken ähnliches wetter dann auch am freitag .",
                                                                               2
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (8,
                                                                               "test/15December_2010_Wednesday_tagesschau-37",
                                                                               "KRAEFTIG AB MORGEN FRUEH MEISTENS SCHNEE SCHNEIEN KALT REGEN",
                                                                               "coord8",
                                                                               "vom nordmeer zieht ein kräftiges tief heran und bringt uns ab den morgenstunden heftige schneefälle zum teil auch gefrierenden regen .",
                                                                               2
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (9,
                                                                               "test/10March_2011_Thursday_heute-58",
                                                                               "WOCHENENDE SONNE SAMSTAG SCHOEN TEMPERATUR BIS SIEBZEHN GRAD REGION",
                                                                               "coord9",
                                                                               "sonnig geht es auch ins wochenende samstag ein herrlicher tag mit temperaturen bis siebzehn grad hier im westen .",
                                                                               2
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (10,
                                                                               "test/14August_2009_Friday_tagesschau-62",
                                                                               "DEUTSCH LAND MORGEN HOCH DRUCK KOMMEN WOLKE AUFLOESEN",
                                                                               "coord10",
                                                                               "deutschland liegt morgen unter hochdruckeinfluss der die wolken weitgehend vertreibt .",
                                                                               2
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (11,
                                                                               "train/11August_2010_Wednesday_tagesschau-1",
                                                                               "JETZT WETTER MORGEN DONNERSTAG ZWOELF FEBRUAR",
                                                                               "coord11",
                                                                               "und nun die wettervorhersage für morgen donnerstag den zwölften august .",
                                                                               3
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (12,
                                                                               "train/11August_2010_Wednesday_tagesschau-4",
                                                                               "ORT REGEN DURCH REGEN KOENNEN UEBERSCHWEMMUNG KOENNEN",
                                                                               "coord12",
                                                                               "mancherorts regnet es auch länger und ergiebig auch lokale überschwemmungen sind wieder möglich .",
                                                                               3
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (13,
                                                                               "train/11August_2010_Wednesday_tagesschau-5",
                                                                               "NORDWEST HEUTE NACHT TROCKEN BLEIBEN SUEDWEST KOENNEN REGEN ORT GEWITTER DAZU",
                                                                               "coord13",
                                                                               "im nordwesten bleibt es heute nacht meist trocken sonst muss mit teilweise kräftigen schauern gerechnet werden örtlich mit blitz und donner .",
                                                                               3
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (14,
                                                                               "train/11August_2010_Wednesday_tagesschau-6",
                                                                               "TAGSUEBER OFT REGEN GEWITTER KOENNEN MANCHMAL REGEN VIEL REGEN",
                                                                               "coord14",
                                                                               "auch am tag gibt es verbreitet zum teil kräftige schauer oder gewitter und in manchen regionen fallen ergiebige regenmengen .",
                                                                               3
                                                                              );

INSERT INTO PARALLEL_ITEM (itemId, file, gloss, skels, text, env_type) VALUES (15,
                                                                               "train/11August_2010_Wednesday_tagesschau-7",
                                                                               "WOLKE LOCH SPEZIELL NORDWEST",
                                                                               "coord15",
                                                                               "größere wolkenlücken finden sich vor allem im nordwesten .",
                                                                               3
                                                                              );