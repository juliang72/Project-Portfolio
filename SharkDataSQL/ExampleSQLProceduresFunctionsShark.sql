 use sharkdb;
 
-- Write a function mostAttacks() that returns the township id of the 
-- township thay had the most number of attacks. The function accepts no arguments. 
-- If there are more than 1 town with the maximum value, return any town with the maximum value.

DELIMITER // 
CREATE FUNCTION mostAttacks()
RETURNS INT 
DETERMINISTIC READS SQL DATA
BEGIN DECLARE most_attacks_var INT;

SELECT max(location) INTO most_attacks_var FROM attack;

RETURN most_attacks_var;
END// 

SELECT mostAttacks();

-- Write a procedure named sharkLenGTE(length_p)  that accepts a length for a 
-- shark and  returns a result set that contains the shark id, shark name, shark length, 
-- shark sex , and the number of detections for that shark for all sharks with a length greater 
-- than or equal to the passed length. 

DROP PROCEDURE IF EXISTS sharkLenGTE;
DELIMITER $$
CREATE PROCEDURE sharkLenGTE(IN length_p FLOAT)
BEGIN
SELECT shark.sid, shark.name, shark.length, shark.sex, shark.detections FROM shark WHERE shark.length>=length_p;
END $$
DELIMITER ;

CALL sharkLenGTE(16);


