import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """
        global reviews

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            query = parse_qs(environ.get('QUERY_STRING', ''))
            location = query.get('location', [None])[0]
            start_date = query.get('start_date', [None])[0]
            end_date = query.get('end_date', [None])[0]

            filtered_reviews = reviews
            if location:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]
            
            if start_date and end_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if start_date <= datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]
            
            elif start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date]
            
            elif end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]

            
            # Analyze sentiment for each review
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])
            
            # Sort reviews by compound sentiment score in descending order
            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            # Create the response body from the filtered reviews and convert to a JSON byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            try:
                size = int(environ.get('CONTENT_LENGTH', 0) or 0)
            except (ValueError):
                size = 0
        
            request_body = environ['wsgi.input'].read(size)
            if not request_body:
                status = '400 Bad Request'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                return [b'{"error": "Empty request body"}']
        
            try:
                content_type = environ.get('CONTENT_TYPE', '')
                if 'application/json' in content_type:
                    data = json.loads(request_body)
                elif 'application/x-www-form-urlencoded' in content_type:
                    data = parse_qs(request_body.decode('utf-8'))
                    # Convert lists to single values
                    data = {k: v[0] for k, v in data.items()}
                else:
                    raise ValueError('Unsupported Content-Type')
            except (json.JSONDecodeError, ValueError) as e:
                status = '400 Bad Request'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                error_response = json.dumps({"error": str(e), "received": request_body.decode('utf-8')})
                return [error_response.encode('utf-8')]
        
            review_body = data.get('ReviewBody')
            location = data.get('Location')

            valid_locations = [
                'Albuquerque, New Mexico',
                'Carlsbad, California',
                'Chula Vista, California',
                'Colorado Springs, Colorado',
                'Denver, Colorado',
                'El Cajon, California',
                'El Paso, Texas',
                'Escondido, California',
                'Fresno, California',
                'La Mesa, California',
                'Las Vegas, Nevada',
                'Los Angeles, California',
                'Oceanside, California',
                'Phoenix, Arizona',
                'Sacramento, California',
                'Salt Lake City, Utah',
                'San Diego, California',
                'Tucson, Arizona'
            ]

            if not review_body or not location or location not in valid_locations:
                status = '400 Bad Request'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                error_response = json.dumps({"error": "Invalid input", "received": data})
                return [error_response.encode('utf-8')]

            if not review_body or not location:
                status = '400 Bad Request'
                headers = [('Content-Type', 'application/json')]
                start_response(status, headers)
                return [b'{"error": "Invalid input"}']
        
            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
            new_review = {
                'ReviewId': review_id,
                'ReviewBody': review_body,
                'Location': location,
                'Timestamp': timestamp
            }
        
            reviews.append(new_review)
        
            response_body = json.dumps(new_review).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()