
Difference between Locking, Blocking and Deadlocking

    Locking: Locking occurs when a connection needs access to a piece of data in a database and it locks it for certain use so that no other transaction is able to access it.
    Blocking: Blocking occurs when a transaction tries to acquire an incompatible lock on a resource that another transaction has already locked. The blocked transaction remains blocked until the blocking transaction releases the lock.
    Deadlocking: Deadlocking occurs when two or more transactions have a resource locked, and each transaction requests a lock on the resource that another transaction has already locked. Neither of the transactions here can move forward, as each one is waiting for the other to release the lock


Delete duplicate data from table so that only first data remains constant

DELETE M1 from managers M1, managers M2 where M2.Name=M1.Name AND M1.Id>M2.Id;

Find the Emloyees who were hired in the Last n months

Select *, TIMESTAMPDIFF(month, Hiredate, current_date()) as 
DiffMonth from employees
where TIMESTAMPDIFF(month, Hiredate, current_date()) between 
1 and 5 order by Hiredate desc;


Find the Name of Employees where First Name, Second Name, and Last Name is given in the table. Some Name is missing such as First Name, Second Name and maybe Last Name. Here we will use COALESCE() function which will return first Non Null values. 
SELECT ID, COALESCE(FName, SName, LName) as Name FROM employees;


Find the Emloyees who hired in the Last n days

select *, DATEDIFF(current_date(), Hiredate)as 
DiffDay from employees
where DATEDIFF(current_date(), Hiredate) between
1 and 100 order by Hiredate desc; 


Find the Employees who were hired in the Last n years

select *, TIMESTAMPDIFF(year, Hiredate, current_date()) as 
DiffYear from employees
where TIMESTAMPDIFF(year, Hiredate, current_date()) 
between 1 and 4 order by Hiredate desc;











RDBMS 	MongoDB
It is a relational database. 	It is a non-relational and document-oriented database.
Not suitable for hierarchical data storage. 	Suitable for hierarchical data storage.
It is vertically scalable i.e increasing RAM. 	It is horizontally scalable i.e we can add more servers.
It has a predefined schema. 	It has a dynamic schema.
It is quite vulnerable to SQL injection. 	It is not affected by SQL injection.
It centers around ACID properties (Atomicity, Consistency, Isolation, and Durability). 	It centers around the CAP theorem (Consistency, Availability, and Partition tolerance).
It is row-based. 	It is document-based.
It is slower in comparison with MongoDB. 	It is almost 100 times faster than RDBMS.
Supports complex joins. 	No support for complex joins.
It is column-based. 	It is field-based.
It does not provide JavaScript client for querying. 	It provides a JavaScript client for querying.
It supports SQL query language only. 	It supports JSON query language along with SQL.






Cassandra would be an optimal choice in the following cases : 
- There are minimum updates in your application.
- There is a need for real-time data analytics and report generations.
- Use if you need to work on huge amount of data.
- Use if you have a requirement for fast writes.
- Use if there is less secondary index needs.
- Use if there is no need for joins or aggregates.
- Use if there is a requirement to integrate with Big Data, Hadoop, Hive, and Spark.
- Use if there is a need for a distributed application.

Cassandra won't be an optimal choice in the following cases:
- Do Not Use if you are not storing volumes of data across racks of clusters.
- Do Not Use if you have a strong requirement for ACID properties.
- Do Not Use if you want to use aggregate function.
- Do Not Use if you are not partitioning your servers.
- Do Not Use if you are application has more read requests than writes.
- Do Not Use if you require strong Consistency.










MySQL | PARTITION BY Clause

A PARTITION BY clause is used to partition rows of table into groups. It is useful when we have to perform a calculation on individual rows of a group using other rows of that group.

    It is always used inside OVER() clause.
    The partition formed by partition clause are also known as Window.
    This clause works on windows functions only. Like- RANK(), LEAD(), LAG() etc.
    If this clause is omitted in OVER() clause, then whole table is considered as a single partition.

Syntax:
The syntax for Partition clause is-

Window_function ( expression ) 
       Over ( partition by expr [order_clause] [frame_clause] ) 

Here, order_clause and frame_clause are optional.

expr can be column names or built-in functions in MySQL.

But, standard SQL permits only column names in expr.

Examples:

Consider, a table “Hacker“:
h_id    h_name  challenge_id    score
3   shubh   111     20
2   aayush  111     80
5   krithik     112     40
5   krithik     114     90
4   tushar  112     30
1   parth   112     40

We have to find the rank of hackers in each challenge. That means we have to list all participated hackers of a challenge along with their rank in that challenge.

Query:

select challenge_id, h_id, h_name, score, 
   dense_rank() over ( partition by challenge_id order by score desc ) 
       as "rank", from hacker;

Explanation:

In the above query, partition by clause will partition table into groups that are having same challenge_id.

order by will arrange the hackers of each partition in descending order by “scores”.

over() clause defines how to partition and order rows of table, which is to be processed by window function rank().

dense_rank() is a window function, which will assign rank in ordered partition of challenges. If two hackers have same scores then they will be assigned same rank.

Output:
challenge_id    h_id    h_name  score   rank
111     2   aayush  80  1
111     3   shubh   20  2
112     5   krithik     40  1
112     1   parth   40  1
112     4   tushar  30  2
114     5   krithik     90  1

Thus, we get list of all hackers along with their ranks in the individual challenges.






MySQL | Ranking Functions

The ranking functions in MySql are used to rank each row of a partition. The ranking functions are also part of MySQL windows functions list.

    These functions are always used with OVER() clause.
    The ranking functions always assign rank on basis of ORDER BY clause.
    The rank is assigned to rows in a sequential manner.
    The assignment of rank to rows always start with 1 for every new partition.

There are 3 types of ranking functions supported in MySQL-

    dense_rank():
    This function will assign rank to each row within a partition without gaps. Basically, the ranks are assigned in a consecutive manner i.e if there is a tie between values then they will be assigned the same rank, and next rank value will be one greater then the previous rank assigned.
    rank():
    This function will assign rank to each row within a partition with gaps. Here, ranks are assigned in a non-consecutive manner i.e if there is a tie between values then they will be assigned same rank, and next rank value will be previous rank + no of peers(duplicates).
    percent_rank():
    It returns the percentile rank of a row within a partition that ranges from 0 to 1. It tells the percentage of partition values less than the value in the current row, excluding the highest value.

In order to understand these functions in a better way.
Let consider a table “result”–
s_name  subjects    mark
Pratibha    Maths   100
Ankita  Science     80
Swarna  English     100
Ankita  Maths   65
Pratibha    Science     80
Swarna  Science     50
Pratibha    English     70
Swarna  Maths   85
Ankita  English     90

Queries:

    dense_rank() function-

    SELECT subjects, s_name, mark, dense_rank() 
    OVER ( partition by subjects order by mark desc ) 
    AS 'dense_rank' FROM result;

    Output-
    Subjects    Name    Mark    Dense_rank
    English     Swarna  100     1
    English     Ankita  90  2
    English     Pratibha    70  3
    Maths   Pratibha    100     1
    Maths   Swarna  85  2
    Maths   Ankita  65  3
    Science     Ankita  80  1
    Science     Pratibha    80  1
    Science     Swarna  50  2

    Explanation-

    Here, table is partitioned on the basis of “subjects”.

    order by clause is used to arrange rows of each partition in descending order by “mark”.

    dense_rank() is used to rank students in each subject.

    Note, for science subject there is a tie between Ankita and Pratibha, so they both are assigned same rank. The next rank value is incremented by 1 i.e 2 for Swarna.
    rank() function-

    SELECT subjects, s_name, mark, rank() 
    OVER ( partition by subjects order by mark desc ) 
    AS 'rank' FROM result;

    Output-
    Subjects    Name    Mark    rank
    English     Swarna  100     1
    English     Ankita  90  2
    English     Pratibha    70  3
    Maths   Pratibha    100     1
    Maths   Swarna  85  2
    Maths   Ankita  65  3
    Science     Ankita  80  1
    Science     Pratibha    80  1
    Science     Swarna  50  3

    Explanation-

    It’s output is similar to dense_rank() function.

    Except, that for Science subject in case of a tie between Ankita and Pratibha, the next rank value is incremented by 2 i.e 3 for Swarna.
    percent_rank() function-

    SELECT subjects, s_name, mark, percent_rank() 
    OVER ( partition by subjects order by mark ) 
    AS 'percent_rank' FROM result;

    Output-
    Subjects    Name    Mark    percent_rank
    English     Pratibha    70  0
    English     Ankita  90  0.5
    English     Swarna  100     1
    Maths   Ankita  65  0
    Maths   Swarna  85  0.5
    Maths   Pratibha    100     1
    Science     Swarna  50  0
    Science     Pratibha    80  0.5
    Science     Ankita  80  0.5

    Explanation:

    Here, the percent_rank() function calculate percentile rank in ascending order by “mark” column.

    percent_rank is calculated using following formula-

    (rank - 1) / (rows - 1)

    rank is the rank of each row of the partition resulted using rank() function.

    rows represent the no of rows in that partition.

    To, clear this formula consider following query-

    SELECT subjects, s_name, mark, rank() 
    OVER ( partition by subjects order by mark )-1 
    AS 'rank-1', count(*) over (partition by subjects)-1
    AS 'total_rows-1', percent_rank()
    OVER ( partition by subjects order by mark ) AS 'percenr_rank'
    FROM result;

    Output-
    Subjects    Name    Mark    rank-1  total_rows-1    percent_rank
    English     Pratibha    70  0   2   0
    English     Ankita  90  1   2   0.5
    English     Swarna  100     2   2   1
    Maths   Ankita  65  0   2   0
    Maths   Swarna  85  1   2   0.5
    Maths   Pratibha    100     2   2   1
    Science     Swarna  50  0   2   0
    Science     Ankita  80  1   2   0.5
    Science     Pratibha    80  1   2   0.5

    Note: While using ranking function, in MySQL query the use of order by clause is must otherwise all rows are considered as peers i.e(duplicates) and all rows are assigned same rank i.e 1.




The OVER clause combined with PARTITION BY is used to break up data into partitions. 
Syntax : function (...) OVER (PARTITION BY col1, Col2, ...)

The specified function operates for each partition.

For example : 
COUNT(Gender) OVER (PARTITION BY Gender) will partition the data by GENDER i.e there will 2 partitions (Male and Female) and then the COUNT() function is applied over each partition.

Any of the following functions can be used. Please note this is not the complete list.
COUNT(), AVG(), SUM(), MIN(), MAX(), ROW_NUMBER(), RANK(), DENSE_RANK() etc.



One way to achieve this is by including the aggregations in a subquery and then JOINING it with the main query as shown in the example below. Look at the amount of T-SQL code we have to write.
SELECT Name, Salary, Employees.Gender, Genders.GenderTotals,
        Genders.AvgSal, Genders.MinSal, Genders.MaxSal   
FROM Employees
INNER JOIN
(SELECT Gender, COUNT(*) AS GenderTotals,
          AVG(Salary) AS AvgSal,
         MIN(Salary) AS MinSal, MAX(Salary) AS MaxSal
FROM Employees
GROUP BY Gender) AS Genders
ON Genders.Gender = Employees.Gender

Better way of doing this is by using the OVER clause combined with PARTITION BY
SELECT Name, Salary, Gender,
        COUNT(Gender) OVER(PARTITION BY Gender) AS GenderTotals,
        AVG(Salary) OVER(PARTITION BY Gender) AS AvgSal,
        MIN(Salary) OVER(PARTITION BY Gender) AS MinSal,
        MAX(Salary) OVER(PARTITION BY Gender) AS MaxSal
FROM Employees







Row_Number function

    Introduced in SQL Server 2005
    Returns the sequential number of a row starting at 1
    ORDER BY clause is required
    PARTITION BY clause is optional
    When the data is partitioned, row number is reset to 1 when the partition changes

Syntax : ROW_NUMBER() OVER (ORDER BY Col1, Col2)

Row_Number function without PARTITION BY : In this example, data is not partitioned, so ROW_NUMBER will provide a consecutive numbering for all the rows in the table based on the order of rows imposed by the ORDER BY clause.

SELECT Name, Gender, Salary,
        ROW_NUMBER() OVER (ORDER BY Gender) AS RowNumber
FROM Employees



Row_Number function with PARTITION BY : In this example, data is partitioned by Gender, so ROW_NUMBER will provide a consecutive numbering only for the rows with in a parttion. When the partition changes the row number is reset to 1.

SELECT Name, Gender, Salary,
        ROW_NUMBER() OVER (PARTITION BY Gender ORDER BY Gender) AS RowNumber
FROM Employees









ID_STUDENT | ID_CLASS | GRADE | RANK
------------------------------------
    2      |    1     |  99   |  1
    1      |    1     |  90   |  2
    3      |    1     |  80   |  3
    4      |    1     |  70   |  4
    6      |    2     |  90   |  1
    1      |    2     |  80   |  2
    5      |    2     |  78   |  3
    7      |    3     |  90   |  1
    6      |    3     |  50   |  2


SELECT id_student, id_class, grade,
   @student:=CASE WHEN @class <> id_class THEN 0 ELSE @student+1 END AS rn,
   @class:=id_class AS clset
FROM
  (SELECT @student:= -1) s,
  (SELECT @class:= -1) c,
  (SELECT *
   FROM mytable
   ORDER BY id_class, id_student
  ) t

This works in a very plain way:

    Initial query is ordered by id_class first, id_student second.
    @student and @class are initialized to -1
    @class is used to test if the next set is entered. If the previous value of the id_class (which is stored in @class) is not equal to the current value (which is stored in id_class), the @student is zeroed. Otherwise is is incremented.
    @class is assigned with the new value of id_class, and it will be used in test on step 3 at the next row.



## if(condition, run if true, run if false)



use test
DROP TABLE IF EXISTS scores;
CREATE TABLE scores
(
    id int not null auto_increment,
    score int not null,
    primary key (id),
    key score (score)
);
INSERT INTO scores (score) VALUES
(50),(40),(75),(80),(55),
(40),(30),(80),(70),(45),
(40),(30),(65),(70),(45),
(55),(45),(83),(85),(60);

Let's load the sample data

mysql> DROP TABLE IF EXISTS scores;
Query OK, 0 rows affected (0.15 sec)

mysql> CREATE TABLE scores
    -> (
    ->     id int not null auto_increment,
    ->     score int not null,
    ->     primary key (id),
    ->     key score (score)
    -> );
Query OK, 0 rows affected (0.16 sec)

mysql> INSERT INTO scores (score) VALUES
    -> (50),(40),(75),(80),(55),
    -> (40),(30),(80),(70),(45),
    -> (40),(30),(65),(70),(45),
    -> (55),(45),(83),(85),(60);
Query OK, 20 rows affected (0.04 sec)
Records: 20  Duplicates: 0  Warnings: 0

Next, let initialize the user variables:

mysql> SET @rnk=0; SET @rank=0; SET @curscore=0;
Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Now, here is the output of the query:

mysql> SELECT score,ID,rank FROM
    -> (
    ->     SELECT AA.*,BB.ID,
    ->     (@rnk:=@rnk+1) rnk,
    ->     (@rank:=IF(@curscore=score,@rank,@rnk)) rank,
    ->     (@curscore:=score) newscore
    ->     FROM
    ->     (
    ->         SELECT * FROM
    ->         (SELECT COUNT(1) scorecount,score
    ->         FROM scores GROUP BY score
    ->     ) AAA
    ->     ORDER BY score DESC
    -> ) AA LEFT JOIN scores BB USING (score)) A;
+-------+------+------+
| score | ID   | rank |
+-------+------+------+
|    85 |   19 |    1 |
|    83 |   18 |    2 |
|    80 |    4 |    3 |
|    80 |    8 |    3 |
|    75 |    3 |    5 |
|    70 |    9 |    6 |
|    70 |   14 |    6 |
|    65 |   13 |    8 |
|    60 |   20 |    9 |
|    55 |    5 |   10 |
|    55 |   16 |   10 |
|    50 |    1 |   12 |
|    45 |   10 |   13 |
|    45 |   15 |   13 |
|    45 |   17 |   13 |
|    40 |    2 |   16 |
|    40 |    6 |   16 |
|    40 |   11 |   16 |
|    30 |    7 |   19 |
|    30 |   12 |   19 |
+-------+------+------+