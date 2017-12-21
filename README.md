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
## Documenation:

#### Summary
The following API is a demo of how to use Facebook’s Fasttext to create your own model, in this case we’ve used a dataset containing sub-reddit categories and post titles. We trained the model on this dataset and achieved a ~80% accuracy, this can be useful for developers if they want to recommend users a category for their post, review, complaint, feedback, etc… so the user doesn’t have to choose one.

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
