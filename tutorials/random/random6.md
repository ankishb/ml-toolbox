 What's power? How to explain it to a non-statistics person? what's false positive and false negative?   

     How to compute an inverse matrix faster by playing around  with some computational tricks?


      How to perform a series of calculations without a calculator and your logic behind the steps.   



       How to builds ads model, basic algorithms.   



       1.Can you explain the Naive Bayes fundamentals? How did you set the threshold?
2.Can you explain what MapReduce is and how it works?  


--------
These are first level topics that are part of a general data science interview, where statistics is one of the skills being brushed over, but not the primary one. These are not for evaluating expertise in statistics, just familiarity and reasonable coherence to apply correctly.

Data exploration

    How do you summarize the distribution of your data?
    How do you handle outliers or data points that skew data?
    What assumptions can you make? Why and when? (i.e When is it safe to assume "normal")


Confidence intervals

    How they are constructed
    Why you standardize
    How to interpret 


Sampling

    Why and when?
    How do you calculate needed sample size? [Power analysis is advanced]
    Limitations
    Bootstrapping and resampling? 


Biases

    When you sample, what bias are you inflicting?
    How do you control for biases?
    What are some of the first things that come to mind when I do X in terms of biasing your data?


Modeling

    Can you build a simple linear model?
    How do you select features?
    How do you evaluate a model?


Experimentation

    How do test new concepts or hypotheses in....insert domain X? i.e. How would evaluate whether or not consumers like the webpage redesign or new food being served?
    How do you create test and control groups?
    How do you control for external factors?
    How do you evaluate results?
------



It really depends on the position you are applying for. I believe a good way to answer your question is to divide the data science positions into several categories. I’ll focus on the most popular ones: Business Intelligence Analyst, Data Analyst, and, of course, Data Scientist.

I’ll start with main responsibilities, so you can understand why these questions are relevant. Then I’ll provide you with my take on why they would ask you that specific question.

Business Intelligence Analyst. The BI Analyst has two defining traits: works with data (often inhouse data), and has a solid business foundation.

Let’s say you need to help a retail shop with its inventory management. You will need to gather the data, design the metrics, and eventually analyze it. Subsequently, you’ll probably need to visualize it in a manager friendly way.

So, FAQs at interviews for a BI Analyst?

    Describe the different parts of an SQL query.
    They may want to check if you are familiar with the SQL programming language. If the job involves SQL, be sure you’ll be asked a lot of those.
    We are thinking about implementing a BI solution, such as Tableau, or Power BI. How would you go about the process of implementation?
    Data science is rising, so many companies want to create a data science team but don’t really know how to approach it. They may rely on you for that. Alternatively, you may be asked about your experience with a BI solution, as they already have one and are looking for someone to utilize it.
    You get A views on a website you are advertising on. B people click on an ad. C people buy your product. What’s the conversion rate? How would you calculate the cost to acquire a customer?
    A BI Analyst may also be involved with the marketing efforts of a company. Still, some business preparation and sound logic will be sufficient for most questions of this type.

Data Analyst. A Data Analyst has different responsibilities; thus, you may expect different FAQ. You are expected to be able to work with data on many levels. From data cleansing, through the actual analysis, to presentation of findings. Often there is some modelling involved, like regression, factor, cluster analysis.

For instance, you may be asked to project the sales of a company, based on some common metrics (or trends). Maybe you will need to download the data, preprocess it, so it is useful for coding, and create a regression that will yield a prediction about the future sales.

So, FAQs:

    You have a 10x10x10 cube. What’s the outside surface area.
    You are expected to be a very analytical person. Often, interviewers will check your critical thinking through a brainteaser.
    What’s the difference between DELETE and TRUNCATE?
    Some SQL never hurts.
    Identify issues on a spreadsheet.
    Not only in data science, but also audit, and consulting, they may ask you to find errors. They are testing your attention to detail, and competence at the same time.
    What data would you ask for, if you need to predict client retention?
    They may not be able to check your regression skills at the interview, but they’ll surely make sure you know how to approach a problem.

Data Scientist. Probably your question was designed to address this job position. The data scientist position is hard to define exactly, but generally, he/she should understand the business processes, while having the technical preparation related to statistical skills, programming. More and more, machine learning seems to be a must.

If you are given a problem, for instance, to predict which clients will stay with the firm (client retention), you will most probably be the one to design a machine learning algorithm. You’ll need to be able to identify the data needed, gather it, preprocess it, and finally analyze it through some machine learning technique. If there is a whole data science team, you may want the data scientist to have management and leadership skills, too. Finally, presentational skills, storytelling and ability to explain a complicated problem in a simple way are also required.

FAQs:

    What does the term ‘data science’ mean to you?
    If you are applying for the position, you surely know what data science is. Still, since you need not only technical skills, but great communicational skills, this is a good way to see how you person thinks and structures his thoughts.
    What are the assumptions of a linear regression?
    Linear regressions are less and less used in the presence of machine learning algorithms, but still, this is some technical knowledge that these people should have (as that’s a starting point).
    Draw a graph relevant for a pay-per-click ad.
    This question checks your analytical skills, technical skills, and knowledge of business processes. Naturally, many data science problems are related to marketing, thus chances are that they’ll ask you something along those lines.
    How can you prove that an improvement you introduced to a model is actually working?
    This point refers to models that check if other models work. You want your data scientist to not only implement algorithms, but be able to measure the impact.
    How would you explain NNs / Random Forest / etc. to a non-technical person.
    Once more: presentational skills and ability to express complicated thoughts in a simple way.
    What types of machine learning are there? Explain the differences between them.
    That’s also an easy question for someone who has done machine learning, but is a very good way to check the structure of thoughts of the applicant.
    Give examples where false positive / false negative is more important.
    That’s a common question which checks statistical knowledge and model design knowledge.
    Tell us about some biases you are likely to encounter when … ?
    Biases are a huge deal in behavioral economics. This question refers to your intuition, knowledge, and ability to make hypotheses about your data, or interpret results, after you’ve analyzed the data.

So, these are some frequently asked questions. Obviously, it depends on the country, company, team, and position you are applying to, but generally, most questions are along those lines.

Hope that helps!


--------


 What Is a Z-Score?

A Z-score is a numerical measurement used in statistics of a value's relationship to the mean (average) of a group of values, measured in terms of standard deviations from the mean. If a Z-score is 0, it indicates that the data point's score is identical to the mean score. A Z-score of 1.0 would indicate a value that is one standard deviation from the mean. Z-scores may be positive or negative, with a positive value indicating the score is above the mean and a negative score indicating it is below the mean.
