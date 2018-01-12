# ClassSearch
## Resources
* [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017/index.html)
* [StarSpace: Embed All The Things!](https://research.fb.com/publications/starspace-embed-all-the-things/)
* [Word2Vec Paper](https://arxiv.org/pdf/1301.3781.pdf)
* [A Comparative Study of Word Embeddings for Reading Comprehension](https://arxiv.org/pdf/1703.00993.pdf)
* [FastText Classifier Paper](https://arxiv.org/pdf/1607.01759.pdf)
* [Embeddings using Subword Information](https://arxiv.org/abs/1607.04606)

## Data
[Reddit Posts Data](https://bigquery.cloud.google.com/dataset/fh-bigquery:reddit_posts?pli=1)

Sample Query:
```sql
SELECT title, score, subreddit
FROM [fh-bigquery:reddit_posts.2017_10], [fh-bigquery:reddit_posts.2017_09]
WHERE subreddit IN ("dailyprogrammer", "docker")
```
## Documenation:

#### Summary
The idea of ClassSearch is to use a text classifier's prediction probabilities to implement a vertical search engine. The following API is a demo of how to use Facebook’s [fastText](https://github.com/facebookresearch/fastText) to create your own model and serve it on a REST API on Flask for this purpose. In this case we’ve used a year's worth of computer science-related subreddit categories and post titles. We chose this data as it serves as a good example of what ClassSearch can achieve: classification of searches (text) to documents. In a search engine, the first search result doesn't always need to be what the user is looking for and can be a few below that and the experience is still good enough. In this demo, we trained a text classifier using fastTExt and achieved a top 10 ~78% accuracy on 117 categories and 84% top 10 accuracy using a tuned fastText model on Keras. For simplicity, this demo uses the former. We believe this concept can be useful for other things as well such as if developers want to recommend users a category for their post, review, complaint, feedback, etc… so the user doesn’t have to choose one.

##### Authors:
- Victor Kwak - Created Jupyter Notebooks and Models
- Luis Magana  - Created API and Containerization

### Installation
1. Download or Clone repository
2. Create docker image and container
```linux
  docker build -t flask-cs410:latest . 
  docker run -p 5000:5000 flask-cs410       <- will run under http://0.0.0.0:5000 (for demo)
  or 
  docker run -d -p 5000:5000 flask-cs410    <- will run under http://{youripaddress}:5000 (for production) 
```
3. Access http://0.0.0.0:5000 to give the API a try. 
#### Notes 
- Install Instructions for Docker: https://docs.docker.com/engine/installation/ (available for Mac, Windows 10, and Linux)
- Space Requirements: ~2GB
- Cleanup Commands (After Grading):
```linux
  docker rm $(docker ps -a -q)
  docker stop $(docker ps -a -q)
  docker rmi $(docker images -q)
```
#### API Functionality
``` java
GET /classify_this_post_api HTTP/1.1
Host: localhost:5000
Accept: application/json, text/javascript
Form: {'post_title':'string'} 
```
##### Python example using requests:
``` python
In [1]: import requests
In [2]: url = 'http://0.0.0.0:5000/classify_this_post_api'
In [3]: requests.post(url,data={'post_title':'help safari crashes'}).content
Out[3]: b'{"osx": 0.19616183075456536, "linux4noobs": 0.00070690865025827931, "chrome": 0.0025314985868822707, "softwaregore": 0.0036844663189471759, "linuxquestions": 0.0012716215919381388, "ios": 0.099633552936388356, "mac": 0.6834370354720688, "macapps": 0.0034646836365399359, "iOSProgramming": 0.00077988063181481839, "AskNetsec": 0.0017493687826002848}'
```
