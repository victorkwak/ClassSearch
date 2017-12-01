# ClassSearch
## Resources
* [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017/index.html)
* [StarSpace: Embed All The Things!](https://research.fb.com/publications/starspace-embed-all-the-things/)
* [Word2Vec Paper](https://arxiv.org/pdf/1301.3781.pdf)
* [A Comparative Study of Word Embeddings for Reading Comprehension](https://arxiv.org/pdf/1703.00993.pdf)
* [FastText Paper](https://arxiv.org/pdf/1607.01759.pdf)

## Data
[Reddit Posts Data](https://bigquery.cloud.google.com/dataset/fh-bigquery:reddit_posts?pli=1)

Sample Query:
```sql
SELECT title, score, subreddit
FROM [fh-bigquery:reddit_posts.2017_10], [fh-bigquery:reddit_posts.2017_09]
WHERE subreddit IN ("dailyprogrammer", "docker")
```