CREATE DATABASE  IF NOT EXISTS `sharkdb` /*!40100 DEFAULT CHARACTER SET utf8 */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `sharkdb`;
-- MySQL dump 10.13  Distrib 8.0.26, for macos11 (x86_64)
--
-- Host: localhost    Database: sharkdb
-- ------------------------------------------------------
-- Server version	8.0.26

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `attack`
--

DROP TABLE IF EXISTS `attack`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `attack` (
  `shark` int DEFAULT NULL,
  `victim` int NOT NULL,
  `fatal` char(1) NOT NULL,
  `date` date NOT NULL,
  `activity` varchar(64) DEFAULT NULL,
  `description` varchar(64) DEFAULT NULL,
  `location` int NOT NULL,
  PRIMARY KEY (`victim`,`date`),
  UNIQUE KEY `victim_UNIQUE` (`victim`),
  UNIQUE KEY `date_UNIQUE` (`date`),
  KEY `fk_attach_victim_idx` (`victim`),
  KEY `fk_attack_shark_idx` (`shark`),
  KEY `fk_attack_location` (`location`),
  CONSTRAINT `fk_attack_location` FOREIGN KEY (`location`) REFERENCES `township` (`tid`),
  CONSTRAINT `fk_attack_shark` FOREIGN KEY (`shark`) REFERENCES `shark` (`sid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_attack_victim` FOREIGN KEY (`victim`) REFERENCES `victim` (`vid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `attack`
--

LOCK TABLES `attack` WRITE;
/*!40000 ALTER TABLE `attack` DISABLE KEYS */;
INSERT INTO `attack` VALUES (NULL,1,'0','1996-07-21','Swimming','Lacerations to left leg and right foot',1),(NULL,2,'0','2010-08-23','Scuba diving','No injury, shark bit camera',2),(NULL,3,'0','2012-07-30','Body surfing','Lacerations to both legs below the knees',1),(NULL,4,'0','2017-08-23','Paddleboarding','No injury, shark bit board',4),(NULL,5,'0','2018-08-15','Swimming','Lacerations to the leg, hip and hand',1),(NULL,6,'1','2018-09-15','Boogie boarding','Attacked and killed, died in hospital',4),(NULL,7,'0','2019-07-22','Surfing','Foot bite',5),(NULL,8,'0','2014-09-03','Kayaking','No injuries, boat damages, both kayakers knocked into water',13);
/*!40000 ALTER TABLE `attack` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `bay`
--

DROP TABLE IF EXISTS `bay`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `bay` (
  `name` varchar(64) NOT NULL,
  PRIMARY KEY (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `bay`
--

LOCK TABLES `bay` WRITE;
/*!40000 ALTER TABLE `bay` DISABLE KEYS */;
INSERT INTO `bay` VALUES ('Buzzards Bay'),('Cape Cod Bay'),('Duxbury Bay'),('Massachusetts Bay'),('Plymouth Bay');
/*!40000 ALTER TABLE `bay` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `gender`
--

DROP TABLE IF EXISTS `gender`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `gender` (
  `sex` varchar(7) NOT NULL,
  PRIMARY KEY (`sex`),
  UNIQUE KEY `sex_UNIQUE` (`sex`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `gender`
--

LOCK TABLES `gender` WRITE;
/*!40000 ALTER TABLE `gender` DISABLE KEYS */;
INSERT INTO `gender` VALUES ('Female'),('Male'),('Unknown');
/*!40000 ALTER TABLE `gender` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `receiver`
--

DROP TABLE IF EXISTS `receiver`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `receiver` (
  `rid` int NOT NULL AUTO_INCREMENT,
  `location` int DEFAULT NULL,
  `sponsor` varchar(45) DEFAULT NULL,
  `area` varchar(64) NOT NULL,
  `deployed` varchar(64) NOT NULL,
  `hauled` varchar(64) DEFAULT NULL,
  `detections` int DEFAULT NULL,
  `individual_sharks_detected` int DEFAULT NULL,
  `bayside` varchar(64) DEFAULT NULL,
  PRIMARY KEY (`rid`),
  KEY `location` (`location`),
  KEY `bayside` (`bayside`),
  KEY `fk1` (`sponsor`),
  CONSTRAINT `fk1` FOREIGN KEY (`sponsor`) REFERENCES `sponsor` (`sponsor_name`) ON UPDATE CASCADE ON DELETE CASCADE,
  CONSTRAINT `receiver_ibfk_1` FOREIGN KEY (`location`) REFERENCES `township` (`tid`) ON UPDATE CASCADE ON DELETE CASCADE,
  CONSTRAINT `receiver_ibfk_2` FOREIGN KEY (`bayside`) REFERENCES `bay` (`name`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=72 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `receiver`
--

LOCK TABLES `receiver` WRITE;
/*!40000 ALTER TABLE `receiver` DISABLE KEYS */;
INSERT INTO `receiver` VALUES (1,3,NULL,'Martins Ledge B','2019-07-16','2020-07-21',0,0,NULL),(2,7,'Thomas Hense','Hingham','2020-06-19','2020-12-11',0,0,'Massachusetts Bay'),(3,8,NULL,'Hull 2','2020-06-17','2020-11-15',10,1,'Massachusetts Bay'),(4,8,'Clarks Sharks','Hull 1','2020-06-17','2020-11-15',7,2,'Massachusetts Bay'),(5,9,'Jenkens Elementry School','Scituate 2','2020-05-22','2020-12-11',39,5,'Massachusetts Bay'),(6,9,'Tim Hanson','Scituate','2020-05-22','2020-12-11',31,5,'Massachusetts Bay'),(7,10,'Town of Marshfield','Marshfield 5','2020-07-02','2020-11-08',0,0,'Massachusetts Bay'),(8,10,'Christina Friel','Marshfield 3','2020-06-08','2020-11-08',41,6,'Massachusetts Bay'),(9,10,'Tim Hanson','Marshfield 2','2020-06-08','2020-11-08',81,6,'Massachusetts Bay'),(10,10,'Town of Marshfield','Marshfield 4','2020-07-02','2020-11-08',111,7,'Massachusetts Bay'),(11,10,'Tim Hanson','Marshfield 1','2020-06-08','2020-11-08',57,4,'Massachusetts Bay'),(12,11,'Bayside Marine Corp','Duxbury Beach','2020-06-02','2020-12-22',107,10,'Duxbury Bay'),(13,12,NULL,'Kingston','2020-07-13','2020-10-22',0,0,'Plymouth Bay'),(14,13,NULL,'Plymouth Beach','2020-07-10','2020-12-22',103,6,'Plymouth Bay'),(15,13,'The Hawley Cottage','Manomet Point','2020-06-02','2020-12-22',7026,16,'Plymouth Bay'),(16,13,NULL,'Ellisville','2020-06-02','2020-12-22',441,15,'Cape Cod Bay'),(17,14,'William Scottii III','Scusset Beach','2020-06-02','2020-12-22',125,13,'Cape Cod Bay'),(18,14,'David Perry','Sandwich','2020-06-16','2021-01-21',132,11,'Cape Cod Bay'),(19,15,'Wave Bait','Scorton Ledge','2020-06-16','2021-01-21',171,16,'Cape Cod Bay'),(20,15,'Lance R. Wachenheim','Sandy Neck','2020-06-16','2021-01-21',370,19,'Cape Cod Bay'),(21,15,NULL,'Barnstable Harbor West','2020-06-16','2021-01-21',352,23,'Cape Cod Bay'),(22,16,NULL,'Dennis','2020-06-12','2020-01-21',926,21,'Cape Cod Bay'),(23,17,NULL,'Billingsgate Shoal Offshore','2020-06-12','2021-01-21',388,30,'Cape Cod Bay'),(24,18,NULL,'Brewster','2020-06-12','2021-01-21',747,27,'Cape Cod Bay'),(25,19,'Bill & Laura Clendenen','Rock Harbor','2020-06-12','2021-01-21',1061,32,'Cape Cod Bay'),(26,17,'Golden Age Spirits','Billingsgate Shoal','2020-06-12','2021-01-21',963,34,'Cape Cod Bay'),(27,17,'Cape and Islands Charm','First Encounter Beach','2020-06-12','2021-01-21',1250,26,'Cape Cod Bay'),(28,4,'Bill & Laura Clendenen','Jeremy Point','2020-06-12','2021-01-21',673,18,'Cape Cod Bay'),(29,4,NULL,'Billingsgate Shoal North','2020-06-12','2021-01-21',651,32,'Cape Cod Bay'),(30,4,NULL,'Great Island','2020-06-12','2021-01-21',290,26,'Cape Cod Bay'),(31,1,'Mark Wisotzky','Pamet South','2020-06-12','2021-01-21',261,28,'Cape Cod Bay'),(32,1,'Thomas D. Brown Real Estate','Pamet North','2020-06-12','2021-01-21',348,22,'Cape Cod Bay'),(33,20,'Kristen Schneider','Provincetown','2020-06-12','2021-01-14',255,22,'Cape Cod Bay'),(34,20,'JW Constronction','Herring Cove','2020-06-12','2021-01-14',473,39,'Cape Cod Bay'),(35,20,'Flash II','Race Point','2020-06-23','2021-01-14',1192,40,NULL),(36,20,'Shark Bitten','Provincetown Airport','2020-06-23','2021-01-14',4175,58,NULL),(37,1,'Brian Grimm','Truro','2020-06-23','2021-01-14',9024,64,NULL),(38,1,NULL,'Peaked Hill 4','2020-05-27','2021-01-14',775,51,NULL),(39,1,NULL,'Peaked Hill 3','2020-05-27','2021-01-14',809,55,NULL),(40,1,NULL,'Peaked Hill 2','2020-05-27','2021-01-14',2641,53,NULL),(41,1,NULL,'Peaked Hill 1','2020-05-27','2021-01-14',7453,74,NULL),(42,1,NULL,'Head of the Meadow RT','2020-08-06','2020-11-07',6530,61,NULL),(43,1,'Lance R. Wachenheim','Highland Beach','2020-06-23','2021-01-14',4885,70,NULL),(44,1,'The Desisto Family','Longnook Beach','2020-06-23','2021-01-14',5189,72,NULL),(45,1,'Tonja Jiran & John Fiori','Ballston Beach','2020-06-23','2021-01-14',3247,76,NULL),(46,4,NULL,'Newcomb Hollow RT1','2020-07-21','2020-08-16',2354,34,NULL),(47,4,'Cheryl Noroian','Beachcomber','2020-06-23','2021-01-14',3482,74,NULL),(48,4,'Barbara & Guangzhi Chen','LeCount Hollow','2020-06-23','2021-01-14',2157,78,NULL),(49,4,NULL,'LeCount Hollow RT','2020-08-10','2020-09-20',1754,43,NULL),(50,4,'Pamela Wildman','Marconi','2020-03-10','2020-07-21',67,10,NULL),(51,17,'Bob & Chris Morgan','Nauset Light','2020-06-23','2021-01-13',3415,83,NULL),(52,19,'Nauset Disposal','Nauset Beach North','2020-06-23','2021-01-13',2729,77,NULL),(53,19,NULL,'Nauset Trail 1 RT','2020-08-13','2020-09-14',1550,41,NULL),(54,19,'Mainsail Events & Marketing','Orleans','2020-06-25','2021-01-13',2521,73,NULL),(55,3,'Michael Degnan','Chatham North Inlet','2020-06-25','2021-01-13',4980,80,NULL),(56,3,NULL,'North Beach Island RT2','2020-08-31','2020-10-19',2143,54,NULL),(57,3,'Chatham Bars Inn','Chatham Bars Inn','2019-06-10','2021-01-13',NULL,NULL,NULL),(58,3,'Randy Fusaro & Tom Coughlin','Chatham Harbor Inlet North','2020-06-25','2021-01-13',7275,82,NULL),(59,3,'Marie McCormack','North Beach Offshore','2020-05-28','2020-10-22',6250,67,NULL),(60,3,'Warren E. Dalton Jr.','Chatham Harbor Inlet South','2020-06-25','2021-01-13',10700,89,NULL),(61,3,'Driscoll','North Shark Cove','2020-06-25','2021-01-13',6920,80,NULL),(62,3,'Gilman Corp','South Shark Cove','2020-06-25','2021-01-13',7136,77,NULL),(63,3,'Seb Corri','South Monomoy','2020-06-25','2021-01-13',1919,67,NULL),(64,21,NULL,'Hyannis','2020-06-19','2020-12-09',NULL,NULL,NULL),(65,22,'Benny Ross','Marthass Vinyard','2019-07-04','2020-11-08',54,10,NULL),(66,22,'Benny Ross','Marthass Vinyard 2','2019-07-04','2020-11-08',32,8,NULL),(67,22,NULL,'Marthass Vinyard 3','2019-07-04','2020-07-27',4,2,NULL),(68,23,NULL,'Cleveland Ledge B','2020-05-09','2020-06-19',NULL,NULL,NULL),(69,23,NULL,'Sippiwisset Rocks B','2020-05-09','2020-06-19',NULL,NULL,NULL),(70,24,NULL,'Cuttyhunk B','2020-05-16','2020-06-15',7,1,NULL),(71,25,NULL,'Buzzards Bay Barge B','2020-05-16','2020-06-15',9,2,'Buzzards Bay');
/*!40000 ALTER TABLE `receiver` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `shark`
--

DROP TABLE IF EXISTS `shark`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `shark` (
  `sid` int NOT NULL AUTO_INCREMENT,
  `name` varchar(64) NOT NULL UNIQUE,
  `sex` varchar(7) NOT NULL,
  `length` double DEFAULT NULL,
  `detections` int DEFAULT NULL,
  PRIMARY KEY (`sid`),
  KEY `fk_gender` (`sex`),
  CONSTRAINT `fk_gender` FOREIGN KEY (`sex`) REFERENCES `gender` (`sex`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=265 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `shark`
--

LOCK TABLES `shark` WRITE;
/*!40000 ALTER TABLE `shark` DISABLE KEYS */;
INSERT INTO `shark` VALUES (1,'32310','Unknown',9,718),(2,'32312','Unknown',10.5,12),(3,'A-Okay','Female',7.5,2479),(4,'Agnes','Female',13,1572),(5,'Aidan','Male',12,989),(6,'Alex','Male',11,13206),(7,'Amy','Female',14,1125),(8,'Anarkali','Unknown',14,1570),(9,'Andale','Unknown',14,1832),(10,'Annie','Unknown',16,870),(11,'Archipelago','Female',10,2),(12,'Ashley Grace','Female',11,5148),(13,'Avery','Female',13,508),(14,'Bambi','Female',9.5,101),(15,'Barrister','Female',10,729),(16,'Batman','Male',10,108),(17,'Bella','Female',12,1817),(18,'Ben','Male',13,10167),(19,'Bertie','Male',12,0),(20,'Betsy','Female',12.5,374),(21,'Beverly Ann','Female',12,7),(22,'Big Bopper','Male',13,15255),(23,'Big Papi','Male',14,11314),(24,'Big Pete','Unknown',11,3612),(25,'Bob','Unknown',8,13),(26,'Bond','Female',10,2593),(27,'Bonnie','Female',9,1721),(28,'Bou','Male',11,0),(29,'Brady','Male',12,858),(30,'Braylon','Unknown',14,354),(31,'Broken Tail','Male',12,6484),(32,'Brucie Mac','Male',11,1421),(33,'Camden','Male',9,10),(34,'Candy Cane','Female',11,243),(35,'Caroline','Female',10,9),(36,'Casey','Male',13,332),(37,'Charli','Female',12.5,0),(38,'Charlotte','Female',14,1106),(39,'Cheeky','Unknown',17,0),(40,'Cheerio','Male',9,541),(41,'Chex','Male',12,7984),(42,'Chopper','Female',11,93),(43,'Cilla B','Unknown',15,2021),(44,'Cindy Lou','Female',15,524),(45,'Clown','Female',10,4),(46,'Commodore','Male',10,2795),(47,'Constellation','Male',12,2023),(48,'Cookie','Unknown',4,0),(49,'Cool Beans','Female',11,7938),(50,'Copy Cat','Male',11,0),(51,'Cow','Male',11,1376),(52,'Curly','Female',17.5,242),(53,'CurvEy Sloth','Male',12,0),(54,'Dale','Male',9.5,1218),(55,'Danny','Male',11,7337),(56,'Dexter','Male',12,704),(57,'Didi','Female',12,1022),(58,'Doogie','Male',12,0),(59,'Dylan','Male',11,0),(60,'Elephant','Male',9.5,0),(61,'Ellie','Female',12,0),(62,'Fangirl','Female',8.5,137),(63,'Fin Face','Female',11,3045),(64,'Fin of MO','Female',8.5,2511),(65,'Finn','Male',12,1533),(66,'FINomenal Me','Unknown',16,121),(67,'Flash','Female',12,4178),(68,'Flea','Male',12.5,0),(69,'Flopsy','Female',10,1524),(70,'Fluffy','Female',11,0),(71,'Frank','Male',7.5,0),(72,'Fruit Loops','Male',10,816),(73,'Fuzzy','Male',8.9,12),(74,'Gannon','Male',12,7195),(75,'Genie','Female',14.8,799),(76,'Ghost','Unknown',13,540),(77,'Gillie','Female',15,310),(78,'Glenn','Male',12,193),(79,'Godzilla','Female',11,3726),(80,'Gooz','Male',11,1212),(81,'Grace','Female',15,41),(82,'Gronk','Male',10,0),(83,'Hammerhead','Male',11,0),(84,'Happy Fins','Male',11,3),(85,'Headbutt','Male',11,0),(86,'Heady Chomper','Male',12,6025),(87,'Hen','Female',10,43),(88,'Hev','Female',12.5,180),(89,'Hilary','Female',10,3965),(90,'Hollywood Sam','Male',8.5,1688),(91,'Hunter','Male',9.8,0),(92,'Iceland','Male',11,0),(93,'Isabelle','Female',10,2555),(94,'Jack','Male',11,6422),(95,'Jack Henry','Male',8,453),(96,'Jacs','Unknown',13,785),(97,'James','Male',14,25140),(98,'Jamison','Male',7,7510),(99,'Janice Anne','Female',10,1756),(100,'JB the Duf','Male',10,5524),(101,'Jenna','Female',12,35),(102,'Jessie','Female',8.4,0),(103,'Joanie','Female',9.5,4204),(104,'Joseph Lockridge','Male',9,2174),(105,'Jules','Male',12,17321),(106,'Katharine','Female',14.1,1931),(107,'Katie','Female',10,3163),(108,'Keelie','Female',11,1387),(109,'Ken','Male',11,0),(110,'Kevin Cadden','Male',11,1224),(111,'Kibble','Female',8.5,2612),(112,'Kite','Female',11,93),(113,'Lady Razor','Unknown',14,0),(114,'Lange Syne','Female',16.4,0),(115,'Large Marge','Female',16,3155),(116,'Lawrence','Male',14,2884),(117,'Lee','Female',16,1428),(118,'Leonardo','Male',12,5414),(119,'Lily','Female',14,301),(120,'Lizzi','Female',13,630),(121,'Lockjaw','Female',8,780),(122,'Lost camera','Unknown',9,175),(123,'Lou Lou','Female',10,1327),(124,'Luci','Female',16.1,343),(125,'Luke','Male',13,20075),(126,'Lumpy','Male',10,1161),(127,'Luna','Unknown',11,1123),(128,'Lydia','Female',14.4,0),(129,'Lyndsay','Female',14,303),(130,'MA2003','Unknown',9,504),(131,'MA2014','Unknown',11.5,89),(132,'MA2107','Female',NULL,0),(133,'MA2110','Male',9,0),(134,'MA2111','Female',11,0),(135,'MA2113','Male',10.5,0),(136,'MA2114','Female',17,0),(137,'MA2118','Female',14,0),(138,'MA2120','Female',10.5,0),(139,'MA2123','Male',NULL,0),(140,'MA2124','Female',9,0),(141,'MA2125','Female',8.5,0),(142,'MA2126','Female',9,0),(143,'MA2128','Unknown',9,0),(144,'MA2131','Female',11.5,0),(145,'MA2132','Male',9,0),(146,'MA2135','Unknown',8,0),(147,'Machiella','Female',11,6380),(148,'Magneto','Female',13,0),(149,'Major BRO','Male',10,234),(150,'Major Mike Donahue','Male',12,2561),(151,'Major Moran','Male',12.5,653),(152,'Mallory','Female',11.6,0),(153,'Mara','Unknown',13,1298),(154,'Margaret','Female',9,1091),(155,'Marianne','Female',15,5218),(156,'Mark','Male',12,0),(157,'Marley','Female',10,78),(158,'Mary Lee','Female',16.1,0),(159,'Maxine','Female',10,500),(160,'Mayo','Male',10,1591),(161,'Meg','Female',14,1468),(162,'Miss Carolina','Female',8.2,1863),(163,'Mo','Female',8.5,539),(164,'Mocha Chip','Female',11,0),(165,'Monomoy','Unknown',16,523),(166,'Moon Lady','Female',14.5,217),(167,'Moose','Female',15,277),(168,'Mousie','Female',10,6898),(169,'Mr. Frisky','Male',11,8191),(170,'Mr. Spot Claw','Male',10,1223),(171,'Mrs. Broken Tail','Unknown',13,273),(172,'Mueller','Male',10,17822),(173,'Myra','Female',9,4985),(174,'Mystery','Male',13.1,264),(175,'Nan-Sea','Female',12,1884),(176,'Nathalie','Female',13,40),(177,'Natso','Female',15,0),(178,'Neptune','Unknown',10,1143),(179,'Newton','Female',9,4481),(180,'Ninja','Female',10,1346),(181,'Notcho Libre','Male',11,0),(182,'Ocean-Maia','Female',9,1441),(183,'Ocearch','Unknown',14,4540),(184,'Omar','Male',11,13113),(185,'Padre','Male',13,14553),(186,'Patagonia Rose','Female',9.5,442),(187,'Paulie','Male',12,0),(188,'Penny','Female',11,867),(189,'Pipit','Unknown',15,780),(190,'Popcorn','Female',11,0),(191,'Poppy','Female',11,1336),(192,'Pretty Sahara','Female',9,3710),(193,'Priya','Unknown',11,730),(194,'Pumpkin','Unknown',10,671),(195,'Puppum','Female',9,0),(196,'Redemption','Unknown',14,957),(197,'Redpoint','Female',8,755),(198,'Robert','Male',12,43),(199,'Rocinante','Male',12,107),(200,'Ronald Tarentino Jr.','Male',13,3572),(201,'Rose','Female',10,1),(202,'Ruby','Female',9,4507),(203,'Ryder','Male',10,26),(204,'Sage','Female',11,2096),(205,'Sage Fish','Female',12,73),(206,'Salacia','Female',14,1147),(207,'Salty','Male',12,9957),(208,'Sammi','Female',11.5,388),(209,'Sandy','Male',12,14646),(210,'Sassquatch','Female',13.5,138),(211,'SC1701','Male',12.1,3886),(212,'SC1801','Unknown',16,0),(213,'SC1802','Male',7,818),(214,'SC1901','Male',8.5,250),(215,'SC1905','Female',7.2,0),(216,'SC1906','Female',7.7,61),(217,'SC1907','Female',8.8,437),(218,'SC1908','Female',8.3,13),(219,'SC1909','Female',7.2,2244),(220,'SC2101','Female',12,0),(221,'SC2103','Female',16,0),(222,'SC2104','Male',11,0),(223,'SC2106','Male',11.6,0),(224,'Scarface','Male',12,9421),(225,'Scarred Face','Female',11.5,2),(226,'Scrapper','Female',10,324),(227,'Scratchy','Male',13,14275),(228,'Scyther','Unknown',15,477),(229,'Sean','Male',13,17909),(230,'Seth','Male',12,2586),(231,'Shark Tooth Tail','Male',12,1191),(232,'Shiny Penny','Female',16,843),(233,'Shipley','Female',11,2903),(234,'Sicklefin','Female',12,0),(235,'Skerry Shark','Male',13,9067),(236,'Snoop Dogg','Male',10,0),(237,'Snowflake','Female',11,1432),(238,'SPC Brian Arsenault','Male',13,7423),(239,'Spike','Male',13,2910),(240,'Squidward','Male',9,1689),(241,'Stitches','Female',13,90),(242,'Stormy','Female',11,2654),(243,'Sunburst','Female',9,5158),(244,'Super Shark','Male',11,273),(245,'Susan','Unknown',15,4915),(246,'Teddy Bear','Female',8,8312),(247,'Terp','Male',11,0),(248,'Tfaql','Unknown',15,24),(249,'Thompson','Male',11,1),(250,'Tony Hawk','Male',10.5,1725),(251,'Toothy','Male',8,1178),(252,'Top Dot UFO','Male',NULL,241),(253,'Turbette','Unknown',8,0),(254,'Turbo','Male',10,18592),(255,'Wally Cadet','Male',9,22),(256,'Web','Male',12,11685),(257,'Weezie','Female',13.1,290),(258,'Whitaker','Male',11.5,1839),(259,'White Eye','Female',10,0),(260,'White Spot','Female',13,2243),(261,'Wildcat','Female',13.8,3),(262,'Wilson','Unknown',14,1247),(263,'Winton','Female',13,146),(264,'Zero','Male',10.5,397);
/*!40000 ALTER TABLE `shark` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `sponsor`
--

DROP TABLE IF EXISTS `sponsor`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sponsor` (
  `sponsor_name` varchar(128) NOT NULL,
  PRIMARY KEY (`sponsor_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sponsor`
--

LOCK TABLES `sponsor` WRITE;
/*!40000 ALTER TABLE `sponsor` DISABLE KEYS */;
INSERT INTO `sponsor` VALUES ('Barbara & Guangzhi Chen'),('Bayside Marine Corp'),('Benny Ross'),('Bill & Laura Clendenen'),('Bob & Chris Morgan'),('Brian Grimm'),('Cape and Islands Charm'),('Chatham Bars Inn'),('Cheryl Noroian'),('Christina Friel'),('Clarks Sharks'),('David Perry'),('Driscoll'),('Flash II'),('Gilman Corp'),('Golden Age Spirits'),('Jenkens Elementry School'),('JW Constronction'),('Kristen Schneider'),('Lance R. Wachenheim'),('Mainsail Events & Marketing'),('Marie McCormack'),('Mark Wisotzky'),('Michael Degnan'),('Nauset Disposal'),('Pamela Wildman'),('Randy Fusaro & Tom Coughlin'),('Seb Corri'),('Shark Bitten'),('The Desisto Family'),('The Hawley Cottage'),('Thomas D. Brown Real Estate'),('Thomas Hense'),('Tim Hanson'),('Tonja Jiran & John Fiori'),('Town of Marshfield'),('Warren E. Dalton Jr.'),('Wave Bait'),('William Scottii III');
/*!40000 ALTER TABLE `sponsor` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `tag`
--

DROP TABLE IF EXISTS `tag`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tag` (
  `shark` int NOT NULL,
  `date` date NOT NULL,
  PRIMARY KEY (`shark`,`date`),
  CONSTRAINT `fk_tag_SHARK1` FOREIGN KEY (`shark`) REFERENCES `shark` (`sid`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `tag`
--

LOCK TABLES `tag` WRITE;
/*!40000 ALTER TABLE `tag` DISABLE KEYS */;
INSERT INTO `tag` VALUES (1,'2011-07-28'),(2,'2011-08-02'),(3,'2019-09-27'),(4,'2015-10-14'),(5,'2019-10-25'),(6,'2017-08-14'),(7,'2012-07-16'),(8,'2012-08-14'),(9,'2012-09-13'),(10,'2015-10-19'),(11,'2020-11-07'),(12,'2016-08-29'),(13,'2015-07-06'),(14,'2020-06-17'),(15,'2017-10-18'),(16,'2020-11-08'),(17,'2017-08-24'),(18,'2015-10-17'),(19,'2014-09-04'),(20,'2013-08-15'),(21,'2014-10-28'),(22,'2016-08-23'),(23,'2016-09-09'),(24,'2019-09-03'),(25,'2011-07-23'),(26,'2019-10-25'),(27,'2020-10-02'),(28,'2021-08-06'),(29,'2015-09-03'),(30,'2012-07-13'),(31,'2017-10-05'),(32,'2018-07-30'),(33,'2020-10-02'),(34,'2018-10-09'),(35,'2016-07-18'),(36,'2016-10-03'),(37,'2019-01-17'),(38,'2014-10-09'),(39,'2014-10-06'),(40,'2019-07-26'),(41,'2012-07-09'),(42,'2020-11-07'),(43,'2016-07-06'),(44,'2016-09-16'),(45,'2019-09-03'),(46,'2020-09-08'),(47,'2019-10-25'),(48,'2017-07-09'),(49,'2016-07-28'),(50,'2021-09-27'),(51,'2020-08-10'),(52,'2012-09-13'),(53,'2021-10-07'),(54,'2020-07-27'),(55,'2019-08-27'),(56,'2018-08-27'),(57,'2019-08-09'),(58,'2021-07-07'),(59,'2021-08-13'),(60,'2021-08-25'),(61,'2021-09-27'),(62,'2019-11-11'),(63,'2018-10-01'),(64,'2019-07-01'),(65,'2014-09-23'),(66,'2018-11-11'),(67,'2017-07-20'),(68,'2020-11-09'),(69,'2020-07-21'),(70,'2021-08-02'),(71,'2019-01-17'),(72,'2020-09-25'),(73,'2016-12-14'),(74,'2018-08-09'),(75,'2012-09-13'),(76,'2012-07-17'),(77,'2014-09-29'),(78,'2015-08-31'),(79,'2019-08-09'),(80,'2015-08-14'),(81,'2020-01-08'),(82,'2016-09-14'),(83,'2021-08-25'),(84,'2016-06-27'),(85,'2021-08-06'),(86,'2016-09-12'),(87,'2020-11-08'),(88,'2020-11-09'),(89,'2019-07-01'),(90,'2019-08-20'),(91,'2017-03-09'),(92,'2021-08-17'),(93,'2019-09-10'),(94,'2017-08-07'),(95,'2019-07-15'),(96,'2017-09-13'),(97,'2014-08-08'),(98,'2015-07-13'),(99,'2019-07-08'),(100,'2019-07-16'),(101,'2017-07-06'),(102,'2019-12-16'),(103,'2019-09-27'),(104,'2019-08-09'),(105,'2011-08-02'),(106,'2013-08-20'),(107,'2019-08-20'),(108,'2016-08-03'),(109,'2021-07-27'),(110,'2020-07-27'),(111,'2019-08-09'),(112,'2020-11-09'),(113,'2015-08-31'),(114,'2016-12-31'),(115,'2012-07-27'),(116,'2016-09-12'),(117,'2016-09-26'),(118,'2017-09-13'),(119,'2017-10-10'),(120,'2014-09-29'),(121,'2019-09-24'),(122,'2018-10-19'),(123,'2018-07-09'),(124,'2013-10-03'),(125,'2016-06-24'),(126,'2015-08-31'),(127,'2019-07-21'),(128,'2013-03-03'),(129,'2018-10-31'),(130,'2020-06-17'),(131,'2020-08-31'),(132,'2021-08-06'),(133,'2021-08-06'),(134,'2021-08-13'),(135,'2021-08-13'),(136,'2021-08-16'),(137,'2021-08-17'),(138,'2021-08-25'),(139,'2021-08-31'),(140,'2021-09-21'),(141,'2021-09-21'),(142,'2021-09-21'),(143,'2021-09-27'),(144,'2021-10-07'),(145,'2021-10-07'),(146,'2021-10-14'),(147,'2018-07-13'),(148,'2019-08-28'),(149,'2020-08-20'),(150,'2015-09-03'),(151,'2019-07-02'),(152,'2021-03-04'),(153,'2015-08-14'),(154,'2015-07-23'),(155,'2015-10-14'),(156,'2021-08-17'),(157,'2020-10-19'),(158,'2012-09-17'),(159,'2020-07-21'),(160,'2018-10-10'),(161,'2012-08-27'),(162,'2017-01-20'),(163,'2019-07-10'),(164,'2021-07-20'),(165,'2015-10-16'),(166,'2020-11-07'),(167,'2017-10-18'),(168,'2019-07-16'),(169,'2016-09-12'),(170,'2020-08-31'),(171,'2019-11-04'),(172,'2018-07-30'),(173,'2016-07-15'),(174,'2015-09-02'),(175,'2018-07-13'),(176,'2019-10-25'),(177,'2017-10-10'),(178,'2020-08-20'),(179,'2018-07-20'),(180,'2019-08-24'),(181,'2020-11-07'),(182,'2019-07-08'),(183,'2012-09-07'),(184,'2016-07-21'),(185,'2016-10-07'),(186,'2019-09-27'),(187,'2021-08-06'),(188,'2017-10-20'),(189,'2017-09-11'),(190,'2021-10-14'),(191,'2017-09-05'),(192,'2020-08-06'),(193,'2012-08-14'),(194,'2016-07-15'),(195,'2021-04-29'),(196,'2012-08-27'),(197,'2015-10-21'),(198,'2016-09-09'),(199,'2020-09-25'),(200,'2019-07-16'),(201,'2020-11-09'),(202,'2018-08-27'),(203,'2018-10-26'),(204,'2017-10-20'),(205,'2014-10-20'),(206,'2017-07-03'),(207,'2012-08-08'),(208,'2019-11-04'),(209,'2015-08-31'),(210,'2019-09-03'),(211,'2017-01-12'),(212,'2018-02-19'),(213,'2018-03-15'),(214,'2019-01-17'),(215,'2019-02-14'),(216,'2019-02-14'),(217,'2019-03-08'),(218,'2019-03-29'),(219,'2019-12-06'),(220,'2021-02-04'),(221,'2021-03-04'),(222,'2021-03-09'),(223,'2021-03-10'),(224,'2019-07-26'),(225,'2011-08-23'),(226,'2019-09-16'),(227,'2015-08-17'),(228,'2016-07-06'),(229,'2015-08-17'),(230,'2018-08-06'),(231,'2014-08-26'),(232,'2012-07-10'),(233,'2017-10-18'),(234,'2021-03-10'),(235,'2017-09-13'),(236,'2021-08-02'),(237,'2018-08-02'),(238,'2014-09-04'),(239,'2014-08-26'),(240,'2019-11-04'),(241,'2015-10-15'),(242,'2015-08-20'),(243,'2020-07-27'),(244,'2015-08-20'),(245,'2011-07-28'),(246,'2019-09-27'),(247,'2021-08-31'),(248,'2019-10-25'),(249,'2020-11-09'),(250,'2019-11-11'),(251,'2020-06-17'),(252,'2014-09-04'),(253,'2021-10-14'),(254,'2017-08-03'),(255,'2019-07-08'),(256,'2017-07-10'),(257,'2013-09-20'),(258,'2019-09-10'),(259,'2021-08-17'),(260,'2017-10-11'),(261,'2016-03-09'),(262,'2010-08-07'),(263,'2020-10-02'),(264,'2020-10-02');
/*!40000 ALTER TABLE `tag` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `township`
--

DROP TABLE IF EXISTS `township`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `township` (
  `tid` int NOT NULL AUTO_INCREMENT,
  `town` varchar(64) NOT NULL,
  `state` char(2) NOT NULL,
  PRIMARY KEY (`tid`)
) ENGINE=InnoDB AUTO_INCREMENT=27 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `township`
--

LOCK TABLES `township` WRITE;
/*!40000 ALTER TABLE `township` DISABLE KEYS */;
INSERT INTO `township` VALUES (1,'Truro','MA'),(2,'Eastport','ME'),(3,'Chatham','MA'),(4,'Wellfleet','MA'),(5,'South Kingstown','RI'),(6,'Cohasset','MA'),(7,'Hingham','MA'),(8,'Hull','MA'),(9,'Scituate','MA'),(10,'Marshfield','MA'),(11,'Duxbury','MA'),(12,'Kingston','MA'),(13,'Plymouth','MA'),(14,'Sandwich','MA'),(15,'Barnstable','MA'),(16,'Dennis','MA'),(17,'Eastham','MA'),(18,'Brewster','MA'),(19,'Orleans','MA'),(20,'Provincetown','MA'),(21,'Hyannis','MA'),(22,'Marthas Vineyarf','MA'),(23,'Falmouth','MA'),(24,'Cuttyhunk Island','MA'),(25,'Buzzards Bay','MA'),(26,'Yarmouth','MA');
/*!40000 ALTER TABLE `township` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `victim`
--

DROP TABLE IF EXISTS `victim`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `victim` (
  `vid` int NOT NULL AUTO_INCREMENT,
  `name` varchar(45) DEFAULT NULL,
  `age` int DEFAULT NULL,
  PRIMARY KEY (`vid`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `victim`
--

LOCK TABLES `victim` WRITE;
/*!40000 ALTER TABLE `victim` DISABLE KEYS */;
INSERT INTO `victim` VALUES (1,'James Orlowski',46),(2,'Scott MacNichol',30),(3,'Chris Myers',50),(4,'Cleveland Bigelow',69),(5,'William Lytton',61),(6,'Arthur Medici',26),(7,'Chase Clary',18),(8,'Several Grandmas',NULL);
/*!40000 ALTER TABLE `victim` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-01-12 10:22:10
