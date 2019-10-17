---
layout: post
title: "What are the best predictors for interest in a book on Goodreads?"
author: "Dan Roth"
categories: datascience
tags: [linear-regression, web-scraping, data-science]
image: "reading-outdoor.jpg"
---

[Goodreads.com](http://goodreads.com) is essentially a social networking website for book lovers; the site introduces 
a slew of new social interactions related to reading books.  But how many of them are pertinent to the average 
Goodreads user?  In order to answer this question, I decided to explore the use of linear regression models to 
predict the total number of people that would want to read a book on Goodreads.  Thankfully, Goodreads tracks an 
insightful statistic called "want to read," which records the number of users that indicate their desire to read a book on a 
book's Goodreads page. 

![sample goodreads page](../assets/img/posts/goodreads/gr_sample_page.png "A Tale of Two Cities on Goodreads")

*<sub>A sample page from Goodreads with the "want to read" button highlighted in red.</sub>*

## Web Scraping Using BeautifulSoup4 and Selenium

In order to create a dataset of books and their attributes I chose to use the BeautifulSoup4 and Selenium packages in 
Python3.  There were several key features that I chose to collect from Goodreads' book pages.  They included a number of
different attributes such as the number of followers an author had, representing their popularity, to categorical features
such as book genre and binding.  I collected the number of trivia questions and "likes" that a quote from the book had 
as well as the Amazon and Kindle prices if available.  While most of the html information I was looking for could easily 
be scraped using BeautifulSoup4, there were some specific interactions that required Selenium, such as the ability to 
click the link to a book's Amazon page to scrape the price or logging in to access the special book stats section of a
Goodreads page.  You can examine the scrape code 
[here at my project repository](https://github.com/DanRothDataScience/goodreads_scrape_predict/blob/master/goodread_scrape.py).
The biggest trick to scraping was to find a way to prevent Goodreads from returning a 403 or 404 status code while using Selenium. 
Ultimately I decided on using a 30 second pause for every 5 entries I scraped in order to avoid being locked out of the
website.

## Initial EDA (Getting Acquainted with the Dataset)

I now imported my data and took a look at the initial statistics for my features.  I had scraped around 1800 raw data
points to start.  First, we import the libraries that will be used for this modeling exploration and then import the data
using pandas.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, collections, operator
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import scipy.stats as stats
```
```python
# load dataframe
book_df = pd.read_pickle("data/book_data.pkl")
```

I then got a feel for my raw dataframe.
```python
book_df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1801 entries, 0 to 1800
Data columns (total 24 columns):
index            1801 non-null int64
book_num         1801 non-null int64
title            1801 non-null object
author           1801 non-null object
followers        1801 non-null int64
pub_date         1543 non-null float64
og_pub_date      755 non-null float64
avg_rating       1801 non-null float64
genre            1784 non-null object
binding          1801 non-null object
pages            1444 non-null float64
language         1799 non-null object
perc_like        1801 non-null int64
trivia           1801 non-null int64
quote_likes      1801 non-null int64
rev_likes        1801 non-null int64
num_revs         1801 non-null int64
num_ratings      1801 non-null int64
kindle_price     51 non-null float64
amzn_price       124 non-null float64
total_added      1801 non-null int64
total_to_read    1801 non-null int64
avg_added        1801 non-null float64
avg_to_read      1801 non-null float64
dtypes: float64(8), int64(11), object(5)
memory usage: 337.8+ KB
```
I filtered out bad columns, such as those with both an unknown title and author from the scrape.  Sadly the Kindle
and Amazon price columns yielded few returns and so were not very useful.
```python
# filter out unknowns and bad columns
mask = ((book_df.title != 'Unknown') & (book_df.author != 'Unknown'))
book_df = book_df[mask].drop(['kindle_price', 'amzn_price', 'index'], axis=1)
book_df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1790 entries, 0 to 1800
Data columns (total 21 columns):
book_num         1790 non-null int64
title            1790 non-null object
author           1790 non-null object
followers        1790 non-null int64
pub_date         1541 non-null float64
og_pub_date      755 non-null float64
avg_rating       1790 non-null float64
genre            1773 non-null object
binding          1790 non-null object
pages            1443 non-null float64
language         1788 non-null object
perc_like        1790 non-null int64
trivia           1790 non-null int64
quote_likes      1790 non-null int64
rev_likes        1790 non-null int64
num_revs         1790 non-null int64
num_ratings      1790 non-null int64
total_added      1790 non-null int64
total_to_read    1790 non-null int64
avg_added        1790 non-null float64
avg_to_read      1790 non-null float64
dtypes: float64(6), int64(10), object(5)
memory usage: 307.7+ KB
```
I then cleaned several of the categorical features in order to aggregate the category values that were deemed to be too
rare.
```python
# clean binding category
book_df.binding = book_df.binding\
.replace(['Kindle Edition', 'e book', 'Nook'], 'ebook')\
.replace('Mass Market Paperback', 'Paperback')\
.replace(['Audiobook', 'Audio CD', 'Audio Cassette'], 'Audio')

binding_count = book_df.binding.value_counts()
other = list(binding_count[binding_count <= 8].index)
book_df.binding = book_df.binding.replace(other, 'Other')

book_df.binding.value_counts()
```
```python
Paperback          765
Hardcover          582
Unknown Binding    294
ebook               69
Other               44
Audio               36
Name: binding, dtype: int64
```
```python
# clean genre category
genre_count = book_df.genre.value_counts()
other = list(genre_count[genre_count <= 10].index)

book_df.genre = book_df.genre.replace(other, 'Other')
book_df.genre.value_counts()
```
```python
Unknown            1132
Other               220
Fiction              54
Nonfiction           48
Mystery              42
History              41
Childrens            36
Fantasy              26
Classics             23
Romance              20
Historical           19
Sequential Art       17
Science              17
Religion             15
Biography            15
Art                  14
Poetry               12
Philosophy           11
Science Fiction      11
Name: genre, dtype: int64
```
```python
# clean language category
lang_count = book_df.language.value_counts()
other = list(lang_count[lang_count < 4].index)

book_df.language = book_df.language.replace(other, 'Other')
book_df.language.value_counts()
```
```python
English       1182
Unknown        449
Spanish         49
German          32
French          25
Other           13
Indonesian       8
Italian          8
Polish           5
Portuguese       5
Japanese         4
Dutch            4
Arabic           4
Name: language, dtype: int64
```
I then examined the top performing books in order to get a sense of context for what was in my dataset.
```python
# print top performing books
book_df.sort_values(by='total_to_read', ascending=False).head()
```

|      | title                   | author             |   followers |   pub_date |   og_pub_date |   avg_rating |   total_to_read |
|-----:|:------------------------|:-------------------|------------:|-----------:|--------------:|-------------:|----------------:|
|   18 | Senkyūhyakuhachijūyonen | George Orwell      |       30197 |        nan |           nan |         4.17 |         1082134 |
|  259 | Crepúsculo              | Stephenie Meyer    |       58712 |       2006 |          2005 |         3.59 |          634932 |
| 1540 | A Tale of Two Cities    | Charles Dickens    |       21408 |       2008 |          1859 |         3.83 |          502501 |
|   48 | El dador de recuerdos   | Lois Lowry         |       16518 |       2010 |          1993 |         4.12 |          431672 |
| 1155 | Crime and Punishment    | Fyodor Dostoyevsky |       31233 |       1981 |          1866 |         4.21 |          410809 |

I cleared NaN values in order to make my dataset as robust as possible.  I chose logical values to replace NaNs
for each feature.
```python
# clear NaNs
book_df.og_pub_date = book_df.og_pub_date.fillna(book_df.pub_date)
book_df = book_df.dropna(subset=['pub_date'])
book_df.genre = book_df.genre.fillna('Unknown')
book_df.pages = book_df.pages.fillna(book_df.pages.mean())
book_df.language = book_df.language.fillna('Unknown')
book_df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1541 entries, 0 to 1800
Data columns (total 21 columns):
book_num         1541 non-null int64
title            1541 non-null object
author           1541 non-null object
followers        1541 non-null int64
pub_date         1541 non-null float64
og_pub_date      1541 non-null float64
avg_rating       1541 non-null float64
genre            1541 non-null object
binding          1541 non-null object
pages            1541 non-null float64
language         1541 non-null object
perc_like        1541 non-null int64
trivia           1541 non-null int64
quote_likes      1541 non-null int64
rev_likes        1541 non-null int64
num_revs         1541 non-null int64
num_ratings      1541 non-null int64
total_added      1541 non-null int64
total_to_read    1541 non-null int64
avg_added        1541 non-null float64
avg_to_read      1541 non-null float64
dtypes: float64(6), int64(10), object(5)
memory usage: 264.9+ KB
```
Not too bad, I ended up with 1541 samples after cleaning the dataframe!  Finally, I started some of the feature engineering 
process by adding some attributes I thought would be useful, such as a book's age or logarithmically scaling the amount of likes on book quotes 
in order to represent a quality factor of the quote interactions and to dampen the effect of overly influential outliers 
in this feature.
```python
# create age feature
book_df['age'] = book_df.pub_date - book_df.og_pub_date

# log processing number of likes of top quote
book_df['log_quote_likes'] = book_df.quote_likes.apply(lambda x: np.log1p(x))
```

## Creating the Modeling Pipeline 

In order to create and test models, I first selected the features that seemed to be the most effective in predicting my target
variable: the amount of people that would click the "want to read" button on a book.  You can see the features I ultimately 
settled on in the code below.  I then split the data into training and test sets for scoring the models.  I chose to use
20% of the data for a test set in this case.
```python
# select features and create train/test split
feature_select = ['total_added', 'avg_added', 'followers', 'pages',
                  'log_quote_likes', 'rev_likes', 'num_ratings',
                   'num_revs', 'trivia', 'age', 'perc_like']

remainder = book_df.columns.drop(feature_select)
target = book_df.total_to_read 
features = book_df[feature_select]

X_train, X_test, y_train, y_test = train_test_split(features, target, 
test_size=0.2, random_state=42)
```
I wanted to get a baseline score using the most basic linear regression model to start.  I created a KFold object in order
to standardize the results across several cross-validated models.
```python
lm = LinearRegression()

# create KFold object and cross-validate model
kf = KFold(n_splits=5, shuffle=True, random_state = 420)
scores = cross_val_score(lm, features, target, cv=kf, scoring='r2')
print('Linear Model Average 5-Fold CV Score:', np.mean(scores))
```
This returned a mean R<sup>2</sup> of 0.442.  This is not terrible for an initial model, but I definitely wanted to do 
better.  I wanted to examine the model behavior first though, so that I would be able to better quantify any improvements
that I arrived at.  Since the objective of this experiment was to find the most influential features that contributed to a model's predictive power, I decided to create a boxplot of the most influential attributes, ranked by their correlation coefficient
variance (both negative and positive).
```python
# plot feature coefficient variance
cv_results = cross_validate(lm, X_train, y_train, cv=5, return_estimator=True)

feat_coef = []
for model in cv_results['estimator']:
    feat_coef.append(list(zip(X_train.columns, model.coef_)))
flat_feat = [item for sublist in feat_coef for item in sublist]

d = defaultdict(list)
for k, v in flat_feat:
    d[k].append(v)
sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
x = ([feat[0] for feat in sorted_d[:2]]
     + [feat[0] for feat in sorted_d[4:6]]
    + [feat[0] for feat in sorted_d[-1:]])
y = ([coef[1] for coef in sorted_d[:2]]
    + [coef[1] for coef in sorted_d[4:6]]
    + [coef[1] for coef in sorted_d[-1:]])

plt.figure(figsize=(12, 6))
ax = sns.boxplot(x,y, linewidth=2, palette='Set1')
ax.set_title('Top 5 Predictive Features for Wanting to Read a Book on Goodreads (Linear)')
ax.set_xlabel('Top Features')
ax.set_ylabel('Correlation Coefficients')
ax.set_ylim(-500,2700)
ax.set(xticklabels=['# of Trivia Questions', 'Book Quote Quality', 
                    '# of Reviews', 'Like %', '# of Top Review Likes']);
```
![feature plot linear](../assets/img/posts/goodreads/feat_plot_lin2.png)

The plot indicates that trivia has both the strongest correlation coefficients and a fairly limited variance, so I was thinking
that trivia may have a significant impact on Goodreads user interactions.  Similarly, the quality of quotes taken from the 
book also had a strong positive effect on a user wanting to read a book.  Strangely enough, the number of likes a
top review on the book page received had a slightly negative correlation with the target variable, which may indicate that
negative reviews receive more likes.  

I then proceeded with residual and Q-Q plots in order to get a better sense of how my model was generalizing.
```python
# residual plot
book_df['predict'] = lm.predict(features)
book_df['res'] = abs(book_df.predict - target)

with sns.axes_style('white'):
    plot=book_df.plot(kind='scatter',
                  x='predict',y='res',alpha=0.3,figsize=(10,6))
plt.title('Linear Model Residual Plot');
```
![linear residual plot](../assets/img/posts/goodreads/linear_res.png)
```python
# diagnose/inspect residual normality using qqplot:
stats.probplot(book_df['res'], dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()
```
![linear Q-Q plot](../assets/img/posts/goodreads/linearQQ.png)

*<sub>The topmost points on this plot correspond with the top performing books previously explored above, such as a Japanese 
version of George Orwell's 1984 and A Tale of Two Cities by Charles Dickens.</sub>*

The residual plot revealed some very strong bias as the model was underfitting the larger predicted values in the data; the 
Q-Q plot confirmed this as the line of best fit was clearly failing to approximate what was clearly an exponential phenomenon.
This makes sense as the most famous books will likely confound any linear modeling of the majority of little-known books
on Goodreads.  These strong outliers are likely skewing the line of best fit up.

Now I proceeded to experiment with several different model enhancement techniques in order to eke out a better performing
model.  I finally settled on introducing 2<sup>nd</sup> degree polynomial terms and interactions in order to reduce the bias
I had observed earlier.  I attempted to add in categorical features such as genre or language but these actually had little effect
on my model, leading me to believe that they may not be discerning factors for a typical Goodreads user.  I then found that
some light Lasso regularization helped in order to tame the new complexity I had introduced with the more complex feature set.
I scaled the features before regularization in order to make sure the regularization acted evenly on the entire feature space.

```python
# scale and generate polynomial features for regularization
std = StandardScaler()
X_tr = std.fit_transform(X_train.values)
X_te = std.transform(X_test.values)
feat_transform = std.transform(features)
p = PolynomialFeatures(degree=2)
X_train_poly = p.fit_transform(X_tr)
feat_poly = p.transform(feat_transform)

lasso = Lasso(tol=0.1, alpha=0.01) # create Lasso model
lasso_fit = lasso.fit(X_train_poly, y_train)

lasso_scores = cross_val_score(lasso, feat_poly, target, cv=kf, scoring='r2')
print('Lasso Model Average 5-Fold CV Score::', np.mean(lasso_scores))
```
This yielded a final mean R<sup>2</sup> of 0.658, which I deemed satisfactory for attempting to explain this complex interaction
using linear regression.  Once again I plotted my most influential features and brought up analyses plots in order to draw
effective comparative conclusions with my first model.
```python
# plot feature coefficient variance
cv_results = cross_validate(lasso, X_train_poly, y_train, cv=5, return_estimator=True)

feat_coef_lasso = []
for model in cv_results['estimator']:
    feat_coef_lasso.append(list(zip(p.get_feature_names(features.columns), model.coef_)))
flat_feat_lasso = [item for sublist in feat_coef_lasso for item in sublist]

d_lasso = defaultdict(list)
for k, v in flat_feat_lasso:
    d_lasso[k].append(v)
sorted_d_lasso = sorted(d_lasso.items(), key=operator.itemgetter(1), reverse=True)
x = ([feat[0] for feat in sorted_d_lasso[1:3]] 
     + [feat[0] for feat in sorted_d_lasso[4:5]] 
     + [feat[0] for feat in sorted_d_lasso[-2:]])
y = ([coef[1] for coef in sorted_d_lasso[1:3]]
     + [coef[1] for coef in sorted_d_lasso[4:5]]
     + [coef[1] for coef in sorted_d_lasso[-2:]])
     
plt.figure(figsize=(14, 6))
ax = sns.boxplot(x,y, linewidth=2, palette='Set1')
ax.set_title('Top 5 Predictive Features for Wanting to Read a Book on Goodreads (Lasso Model)')
ax.set_xlabel('Top Features')
ax.set_ylabel('Correlation Coefficients')
ax.set(xticklabels=['# of Trivia Questions', 'Book Quote Quality', 
                    '# of Top Review Likes', 'Interaction Btwn Quotes 
			and Likes', '# of Ratings']);
```
![lasso feature plot](../assets/img/posts/goodreads/featplot_lasso.png)
```python
# residual plot
book_df['lasso_pred'] = lasso_fit.predict(feat_poly)
book_df['lasso_res'] = abs(book_df.lasso_pred - target)

with sns.axes_style('white'):
    plot=book_df.plot(kind='scatter',
                  x='lasso_pred',y='lasso_res',alpha=0.2,figsize=(10,6), ylim=(0,1e5))
plt.title('Lasso Model Residual Plot');
```
![lasso residual plot](../assets/img/posts/goodreads/lasso_res.png)
```python
# diagnose/inspect residual normality using qqplot:
stats.probplot(book_df['lasso_res'], dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()
```
![lasso QQ plot](../assets/img/posts/goodreads/lassoQQ.png)

We see that the final model has tamed a lot of the bias from the original model using polynomial features.  Some of the 
resultant variance was then pulled back with Lasso regularization.  The line of best fit is a much better approximation of the
phenomenon overall.  However, this model is still failing to generalize over the entire behavior of the target, and this is likely
due to the fact that we are still observing the exponential effect of particularly popular books.  

The feature plot is promising however, and confirms that trivia and quote interactions are still some of the most 
predictive qualities of whether or not a Goodreads user will want to read a book.  However, it is strange that the number 
of ratings is so negatively correlated with a user wanting to read a book.  This may mean that a large number of reviews
could be negative or simply not the most useful information for a Goodreads user to decide their interest in a book.  While
polynomial features have improved the model, they do hinder some of the interpretability of the resulting features in determining
the most influential factors.  Highly negatively correlated features like the interaction between the quality of quotes from
a book and the percentage of people that liked a book are a bit obscure to understand.  Perhaps if a lot of people like a book
but there is a lack of quotes this can hurt a user's interest in the book.  

## Final Thoughts

This was an interesting exercise in trying to fit a linear regression model to a clearly more complex interaction.  While
the final models are not the most robust, the process did confirm that factors like the presence of trivia and strong book
quotes drove better Goodreads user interactions with books.  However, it was strange to note that features like the average 
rating of a book or the number of ratings were either poorly predictive or negatively correlated with a user's interest in 
a book.  

My final recommendations to Goodreads are to experiment with book trivia and quotes sections.  It may help books
to put greater emphasis on these sections and to encourage book publishers to supply good trivia and quotes for their book
pages.  This makes sense as book readers may really enjoy learning facts about books and finding memorable lines from the text.  Goodreads should look further into their ratings system to examine why it may not be properly correlated to encouraging a user's interest in reading a book.  It may be of interest 
in the future to create separate models for a majority of books on Goodreads and some of the famous classics and outliers
(the Orwells and Dickens seen here) in order to better generalize this interaction and to draw even stronger conclusions.
Until then, happy reading!

You can access the full project repository [here](https://github.com/DanRothDataScience/goodreads_scrape_predict).


