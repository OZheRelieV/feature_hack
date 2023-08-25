import ast
import gc
# import json
import warnings
from typing import List

import nest_asyncio
import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyngrok import ngrok
from sklearn.neighbors import DistanceMetric
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import wikipedia

gc.enable()
warnings.filterwarnings("ignore")
wikipedia.set_lang("ru")

TOP_K = 5
# DATABASE with sights
URL = "https://a20047-194f.s.d-f.pw/api/category/subcategory/sight"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class Graph():
    __slots__ = ("n_v", "graph",)

    def __init__(self, n_v):
        self.n_v = n_v
        self.graph = np.empty((n_v, n_v))

    def set_graph(self, graph):
        self.graph = graph

    # A function to find the vertex with minimum distance value,
    # from the set of vertices not yet included in shortest path tree
    def min_distance(self, dist, spt_set):
        min = 1e7
        # Search not nearest vertex not in the shortest path tree
        for v in range(self.n_v):
            if (dist[v] < min) and (spt_set[v] is False):
                min, min_index = dist[v], v
        return min_index

    def dijkstra(self, src):

        dist = [1e7] * self.n_v
        dist[src] = 0
        spt_set = [False] * self.n_v

        for _ in range(self.n_v):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.min_distance(dist, spt_set)

            # Put the minimum distance vertex in the shortest path tree
            spt_set[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.n_v):
                if (
                    (self.graph[u][v] > 0) and
                    (spt_set[v] is False) and
                    (dist[v] > dist[u] + self.graph[u][v])
                ):
                    dist[v] = dist[u] + self.graph[u][v]
        return dist


def get_res(res, top_k=TOP_K):
    fin_res = np.empty((top_k, ), dtype=np.uint16)
    for i, element in enumerate(res[:top_k]):
        fin_res[i] = element[0]
    return fin_res


def create_json(fin_res):
    if not isinstance(fin_res, pd.core.frame.DataFrame):
        raise ValueError("Invalid fin_res type")
    temp = [
        {
            "id": item[0],
            "name": item[1],
            "category": item[2],
            "subcategory": item[3],
            "rating": item[4],
            "coord_x": item[5],
            "coord_y": item[6],
            "short_descrip": item[7]
        }
        for item in fin_res.values
    ]
    return {
        "items": temp
    }


class DatabaseLoading():
    def __init__(self, url=URL):
        r = requests.get(url)
        if (r.status_code == 200):
            self.res = ast.literal_eval(r.text)

    def _sklearn_haversine(self, latitude, longitude):  # to weight graph edges
        haversine = DistanceMetric.get_metric("haversine")
        latlon = np.hstack((latitude[:, np.newaxis], longitude[:, np.newaxis]))
        dists = haversine.pairwise(latlon)
        return 6371 * dists  # multiplication by const to get distance in km

    def load(self):
        vals = [item.values() for item in self.res]
        data = pd.DataFrame(
            columns=[
                "id", "name", "category", "subcategory",
                "rating", "coord_x", "coord_y",
                ],
            data=vals
        )
        dist = self._sklearn_haversine(
            data["coord_x"].values,
            data["coord_y"].values
        )
        return data, dist


class RequestBody(BaseModel):
    subcategories: List[str]


@app.get("/")
async def index():
    return {"message": "It' working"}

# @app.on_event("startup")
# async def startup():
#     global data, dist
#     data, dist = DatabaseLoading().load()


class SummarizarionModel:
    model = None

    def load_model(self):
      self.tokenizer = T5TokenizerFast.from_pretrained("UrukHan/t5-russian-summarization")
      self.model = AutoModelForSeq2SeqLM.from_pretrained("UrukHan/t5-russian-summarization")

    def predict(self, name):
      text = wikipedia.summary(name, sentences=10)
      text = text[:text.find("=")].strip()
      input_sequences = [' '.join(text.split('\n')).strip()]

      if type(input_sequences) != list: input_sequences = [input_sequences]
      encoded = self.tokenizer(
        ["Spell correct: " + sequence for sequence in input_sequences],
        padding="longest",
        max_length=256,
        truncation=True,
        return_tensors="pt",
      )

      predicts = self.model.generate(encoded["input_ids"])
      ans = self.tokenizer.batch_decode(predicts, skip_special_tokens=True)
      if (len(ans) == 0):
         return name
      else:
        return ans


summarizarion_model = SummarizarionModel()


@app.post("/api/predict")
async def get_user_cats(user_cats: RequestBody):
    # json structure
    # {
    #   "subcategories": ["cat1", "cat2"]
    # }
    data, dist = DatabaseLoading().load()
    threshold = user_cats.subcategories
    print(threshold)
    if len(threshold) != 0:
        category_mask = (data["subcategory"].isin(threshold))
        data = data.loc[category_mask]
        data.reset_index(drop=True, inplace=True)

    g = Graph(data.shape[0])
    g.set_graph(dist)

    optimal_way_mmaping = {}
    for i in range(data.shape[0]):
        dijkstra_res = g.dijkstra(i)
        optimal_way_mmaping[i] = (
            sum(dijkstra_res), zip(range(data.shape[0]), dijkstra_res)
        )

    to_return = sorted(
        min(optimal_way_mmaping.items())[1][1],
        key=lambda x: x[1],
        reverse=False
    )

    fin_res = data.loc[get_res(to_return)]
    short_desrip = [
        summarizarion_model.predict(name)
        for name in fin_res["name"]
    ]
    fin_res["short_desrip"] = short_desrip
    resss = create_json(fin_res)

    return resss

@app.on_event("startup")
async def startup():
    summarizarion_model.load_model()


if __name__ == "__main__":

    port = 80
    ngrok_tunnel = ngrok.connect(port)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=port)
