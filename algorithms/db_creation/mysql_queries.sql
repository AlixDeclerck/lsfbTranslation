
select p.text, p.gloss, e.type from PARALLEL_ITEM as p inner join ENVIRONMENT as e on p.env_type = e.envId;

INSERT INTO PARALLEL_ITEM (text, gloss, env_type) VALUES ("text", "gloss", 3)
DELETE FROM PARALLEL_ITEM WHERE text = "text";

select * from ENVIRONMENT as e order by e.envId
