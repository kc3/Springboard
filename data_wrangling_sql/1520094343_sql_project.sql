
/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */

SELECT name
FROM  `Facilities` 
WHERE membercost =0

A1:
Badminton Court
Table Tennis
Snooker Table
Pool Table

/* Q2: How many facilities do not charge a fee to members? */

SELECT COUNT( * ) 
FROM  `Facilities` 
WHERE membercost =0

A2:
4

/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

SELECT facid, name, membercost, monthlymaintenance
FROM  `Facilities` 
WHERE membercost >0
AND membercost < ( 0.2 * monthlymaintenance ) 

A3:
facid	name		membercost	monthlymaintenance	
0	Tennis Court 1	5.0		200
1	Tennis Court 2	5.0		200
4	Massage Room 1	9.9		3000
5	Massage Room 2	9.9		3000
6	Squash Court	3.5		80


/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */

SELECT * 
FROM  `Facilities` 
WHERE facid
IN ( 1, 5 ) 

A4:
name	        membercost	guestcost	facid	initialoutlay	monthlymaintenance	
Tennis Court 2	5.0	        25.0		1	8000		200
Massage Room 2	9.9	        80.0		5	4000		3000

/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */

SELECT name, monthlymaintenance, 
CASE WHEN monthlymaintenance >100
THEN  'expensive'
ELSE  'cheap'
END AS monthlymaintenancecategory
FROM  `Facilities` 

A5:
name		monthlymaintenance	monthlymaintenancecategory	
Tennis Court 1	200			expensive
Tennis Court 2	200			expensive
Badminton Court	50			cheap
Table Tennis	10			cheap
Massage Room 1	3000			expensive
Massage Room 2	3000			expensive
Squash Court	80			cheap
Snooker Table	15			cheap
Pool Table	15			cheap

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */

SELECT firstname, surname
FROM  `Members` 
ORDER BY joindate DESC 
LIMIT 1

A6:
firstname	surname	
Darren		Smith

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

SELECT DISTINCT B.name, CONCAT( C.firstname,  ' ', C.surname ) AS membername
FROM  `Bookings` AS A
INNER JOIN  `Facilities` AS B ON A.facid = B.facid
INNER JOIN  `Members` AS C ON A.memid = C.memid
WHERE B.name
IN (
'Tennis Court 1',  'Tennis Court 2'
)
ORDER BY membername
LIMIT 100

A7:
name	membername	
Tennis Court 2	Anne Baker
Tennis Court 1	Anne Baker
Tennis Court 1	Burton Tracy
Tennis Court 2	Burton Tracy
Tennis Court 2	Charles Owen
Tennis Court 1	Charles Owen
Tennis Court 2	Darren Smith
Tennis Court 2	David Farrell
Tennis Court 1	David Farrell
Tennis Court 2	David Jones
Tennis Court 1	David Jones
Tennis Court 1	David Pinker
Tennis Court 1	Douglas Jones
Tennis Court 1	Erica Crumpet
Tennis Court 2	Florence Bader
Tennis Court 1	Florence Bader
Tennis Court 2	Gerald Butters
Tennis Court 1	Gerald Butters
Tennis Court 1	GUEST GUEST
Tennis Court 2	GUEST GUEST
Tennis Court 2	Henrietta Rumney
Tennis Court 2	Jack Smith
Tennis Court 1	Jack Smith
Tennis Court 1	Janice Joplette
Tennis Court 2	Janice Joplette
Tennis Court 1	Jemima Farrell
Tennis Court 2	Jemima Farrell
Tennis Court 1	Joan Coplin
Tennis Court 2	John Hunt
Tennis Court 1	John Hunt
Tennis Court 1	Matthew Genting
Tennis Court 2	Millicent Purview
Tennis Court 1	Nancy Dare
Tennis Court 2	Nancy Dare
Tennis Court 1	Ponder Stibbons
Tennis Court 2	Ponder Stibbons
Tennis Court 1	Ramnaresh Sarwin
Tennis Court 2	Ramnaresh Sarwin
Tennis Court 2	Tim Boothe
Tennis Court 1	Tim Boothe
Tennis Court 1	Tim Rownam
Tennis Court 2	Tim Rownam
Tennis Court 2	Timothy Baker
Tennis Court 1	Timothy Baker
Tennis Court 1	Tracy Smith
Tennis Court 2	Tracy Smith

/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

SELECT CONCAT( firstname,  ' ', surname ) AS membername, C.name, 
CASE WHEN B.memid =0
THEN slots * guestcost
ELSE slots * membercost
END AS daycost
FROM  `Bookings` AS A
INNER JOIN  `Members` AS B ON A.memid = B.memid
AND DATE( A.starttime ) =  '2012-09-14'
INNER JOIN  `Facilities` AS C ON A.facid = C.facid
WHERE (

CASE WHEN B.memid =0
THEN slots * guestcost
ELSE slots * membercost
END
) >30
LIMIT 100

A8:
membername	name		daycost	
GUEST GUEST	Tennis Court 1	75.0
GUEST GUEST	Tennis Court 1	75.0
GUEST GUEST	Tennis Court 2	75.0
GUEST GUEST	Tennis Court 2	150.0
GUEST GUEST	Massage Room 1	160.0
GUEST GUEST	Massage Room 1	160.0
Jemima Farrell	Massage Room 1	39.6
GUEST GUEST	Massage Room 1	160.0
GUEST GUEST	Massage Room 2	320.0
GUEST GUEST	Squash Court	70.0
GUEST GUEST	Squash Court	35.0
GUEST GUEST	Squash Court	35.0

/* Q9: This time, produce the same result as in Q8, but using a subquery. */
SELECT * 
FROM (
SELECT CONCAT( firstname,  ' ', surname ) AS membername, C.name, 
CASE WHEN B.memid =0
THEN slots * guestcost
ELSE slots * membercost
END AS daycost
FROM  `Bookings` AS A
INNER JOIN  `Members` AS B ON A.memid = B.memid
AND DATE( A.starttime ) =  '2012-09-14'
INNER JOIN  `Facilities` AS C ON A.facid = C.facid
) AS daycosts
WHERE daycost >30
LIMIT 100

A9:
membername	name		daycost	
GUEST GUEST	Tennis Court 1	75.0
GUEST GUEST	Tennis Court 1	75.0
GUEST GUEST	Tennis Court 2	75.0
GUEST GUEST	Tennis Court 2	150.0
GUEST GUEST	Massage Room 1	160.0
GUEST GUEST	Massage Room 1	160.0
Jemima Farrell	Massage Room 1	39.6
GUEST GUEST	Massage Room 1	160.0
GUEST GUEST	Massage Room 2	320.0
GUEST GUEST	Squash Court	70.0
GUEST GUEST	Squash Court	35.0
GUEST GUEST	Squash Court	35.0


/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

SELECT C.name, SUM( 
CASE WHEN B.memid =0
THEN slots * guestcost
ELSE slots * membercost
END ) AS revenue
FROM  `Bookings` AS A
INNER JOIN  `Members` AS B ON A.memid = B.memid
INNER JOIN  `Facilities` AS C ON A.facid = C.facid
GROUP BY C.name
ORDER BY revenue DESC 
LIMIT 100

A10:
name		revenue	
Massage Room 1	50351.6
Massage Room 2	14454.6
Tennis Court 2	14310.0
Tennis Court 1	13860.0
Squash Court	13468.0
Badminton Court	1906.5
Pool Table	270.0
Snooker Table	240.0
Table Tennis	180.0