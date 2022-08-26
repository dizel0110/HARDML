import requests


secret_number = None


def init_external(url: str, max_retries: int = 20) -> None:
    n_retries = 0
    status_code = 0
    while status_code != 200:
        n_retries += 1
        if n_retries > max_retries:
            raise ConnectionError(f'Couldnt connect to {url}')
        
        response = requests.get(url)
        status_code = response.status_code
    
    global secret_number
    secret_number = response.json()['secret_number']
            