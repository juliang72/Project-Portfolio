-- Which are  the longest books (in pages)? Return the book’s title  in the result. 
-- Return all books with the maximum number of pages. 
SELECT title FROM book WHERE page_count = (SELECT max(page_count) FROM book);

-- Determine the number of members who have books checked out. Rename the count num_members. 
SELECT count(DISTINCT current_holder) AS num_members FROM book;

-- Return the member’s first name and last name who is a member in  all book clubs. 
SELECT member.first_name, member.last_name FROM member RIGHT OUTER JOIN reading_club_members 
ON member.username=reading_club_members.member_username GROUP BY reading_club_members.member_username 
HAVING count(DISTINCT reading_club_members.club_name)=(SELECT count(DISTINCT reading_club.name) FROM reading_club);
