from bs4 import BeautifulSoup
import requests
from collections import defaultdict
import urllib.parse
import sys

start = sys.argv[1]

def parse_url(url) -> str:
    return str(urllib.parse.urlparse(url).geturl())

graph = defaultdict(list)
visited = set([start])
queue = [start]

it = 0
while queue:

    parent = queue.pop(0)
    it += 1

    try:
        print(parent)
        response = requests.get(parent)
        soup = BeautifulSoup(response.content)
    except Exception:
        continue

    links = soup.find_all('a') + soup.find_all('li')

    for link in links:
        child = link.get('href')

        if child is None:
            continue

        if child is '':
            continue

        if child is '/': # homepage, skip
            continue

        if child[0] is '/': # subpage
            child = urllib.parse.urljoin(parent, child)

        child = parse_url(child)

        if child not in visited:
            graph[parent].append(child)
            visited.add(child)
            queue.append(child)
